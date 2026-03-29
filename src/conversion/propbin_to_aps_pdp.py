import gzip
import json
import os
import pickle
import random
import sys

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_PROPAGATION_DIR = os.path.join(CURRENT_DIR, "cpp_propagation")
if CPP_PROPAGATION_DIR not in sys.path:
    sys.path.append(CPP_PROPAGATION_DIR)

from propbin_reader import load_propbin


# ================= Configuration =================
PKL_PATH = "source_0_propagation_data.pkl.gz"
SERVER_DATA_DIR = os.environ.get("SERVER_DATA_DIR", r"/home/hjx/multipath-2d").strip()

OUT_DELAY_SEQ_TPL = "pdp_{name}.npy"
OUT_ANGLE_SEQ_TPL = "aps_{name}.npy"
OUTPUT_ROOT_DIR_FMT = "{map_id}_adps"

NUM_SAMPLES = 500
RECEIVER_SAMPLE_SEED = 2026
MAX_VALID_LOSS = 310.0

# If set, load sampled_rx_keys.json from this root instead of the output dir.
# Point this to the GT SERVER_DATA_DIR so all methods share the same RX sampling.
# e.g. SAMPLE_RECORD_SERVER_DIR=/data/gt  → reads /data/gt/map_{id}/{id}_adps/sampled_rx_keys.json
SAMPLE_RECORD_SERVER_DIR = os.environ.get("SAMPLE_RECORD_SERVER_DIR", "").strip()

SYS_CONF = {
    "fs": 200e6,
    "N": 16,
    "d_lambda": 0.5,
}

GRID_CONF = {
    "theta_min": -180,
    "theta_max": 180,
    "n_theta": 180,
    "tau_min": 0,
    "tau_max": 1000e-9,
    "n_tau": 128,
}

GLOBAL_MIN_DB = -200.0
SAMPLE_RECORD_FILENAME = "sampled_rx_keys.json"
# ================================================


def _tx_key(tx_loc):
    return f"{int(round(tx_loc[0]))}_{int(round(tx_loc[1]))}"


def load_sample_record(base_out, tx_loc):
    path = os.path.join(base_out, SAMPLE_RECORD_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    key = _tx_key(tx_loc)
    if key not in data:
        return None
    return set(map(tuple, data[key]))


def save_sample_record(base_out, tx_loc, samples):
    path = os.path.join(base_out, SAMPLE_RECORD_FILENAME)
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    key = _tx_key(tx_loc)
    data[key] = [
        (int(round(rec["location"][0])), int(round(rec["location"][1])))
        for rec in samples
    ]
    with open(path, "w") as f:
        json.dump(data, f)


def restore_original_format(optimized_data):
    print("   [system] restoring optimized data format...")
    original = {}
    if "source_location" in optimized_data:
        loc = optimized_data["source_location"]
        original["source_location"] = (
            float(loc[0]),
            float(loc[1]),
            float(loc[2]) if len(loc) > 2 else 0.0,
        )
    if "receivers" in optimized_data:
        original["receivers"] = {}
        receivers = optimized_data["receivers"]
        paths = optimized_data.get("paths", np.array([]))

        for receiver in receivers:
            receiver_key = f"receiver_{receiver['x']}_{receiver['y']}"
            path_info = []
            start_idx = receiver["path_start_idx"]
            end_idx = receiver["path_end_idx"]
            for j in range(start_idx, end_idx):
                if j < len(paths):
                    path = paths[j]
                    if "arrival_azimuth" in path.dtype.names:
                        arrival_angle = [
                            float(path["arrival_azimuth"]),
                            float(path["arrival_elevation"]),
                        ]
                    else:
                        arrival_angle = [0.0, 0.0]
                    path_dict = {
                        "level": int(path["level"]),
                        "arrival_angle": arrival_angle,
                        "distance": float(path["distance"]),
                        "loss": float(path["loss"]),
                        "delay": float(path["delay"]),
                    }
                    path_info.append(path_dict)
            original["receivers"][receiver_key] = {
                "total_intensity_dBm": float(receiver["total_intensity_dBm"]),
                "total_loss_dB": float(receiver["total_loss_dB"]),
                "path_info": path_info,
                "location": (
                    float(receiver["loc_x"]),
                    float(receiver["loc_y"]),
                    float(receiver["loc_z"]),
                ),
            }
    else:
        return optimized_data
    return original


def generate_adps(toas_s, doas_deg, rss_dbm, system_config, grid_config):
    fs = system_config["fs"]
    n_ant = system_config["N"]
    d_lambda = system_config["d_lambda"]

    theta_axis = np.linspace(
        grid_config["theta_min"], grid_config["theta_max"], grid_config["n_theta"]
    )
    tau_axis = np.linspace(
        grid_config["tau_min"], grid_config["tau_max"], grid_config["n_tau"]
    )

    tau_grid, theta_grid = np.meshgrid(tau_axis, theta_axis, indexing="ij")
    powers = 10 ** (rss_dbm / 10.0)

    delta_tau = tau_grid[:, :, None] - toas_s[None, None, :]
    delay_resp = np.sinc(fs * delta_tau)

    theta_rad_grid = np.deg2rad(theta_grid)
    doas_rad = np.deg2rad(doas_deg)
    delta_sin = np.sin(theta_rad_grid[:, :, None]) - np.sin(doas_rad[None, None, :])
    psi = np.pi * d_lambda * delta_sin

    numerator = np.sin(n_ant * psi)
    denominator = n_ant * np.sin(psi)

    with np.errstate(divide="ignore", invalid="ignore"):
        angle_resp = numerator / denominator
        angle_resp = np.nan_to_num(angle_resp, nan=1.0)

    adps = np.sum(powers[None, None, :] * (delay_resp ** 2) * (angle_resp ** 2), axis=2)
    return adps, tau_axis, theta_axis


def generate_sequences(adps_dbm, tau_axis, theta_axis):
    max_p_tau = np.max(adps_dbm, axis=1)
    arg_max_tau = np.argmax(adps_dbm, axis=1)
    dom_angles = theta_axis[arg_max_tau]
    seq_delay = np.stack((max_p_tau, dom_angles), axis=1)

    max_p_theta = np.max(adps_dbm, axis=0)
    arg_max_theta = np.argmax(adps_dbm, axis=0)
    dom_delays_ns = tau_axis[arg_max_theta] * 1e9
    clean_delays = np.clip(dom_delays_ns, 0.0, 1000.0)
    seq_angle = np.stack((max_p_theta, clean_delays), axis=1)

    return seq_delay, seq_angle


def parse_map_id(path_str):
    parts = os.path.abspath(path_str).split(os.sep)
    for part in parts:
        if part.startswith("map_"):
            return part.split("map_")[1]
    return "unknown"


def find_map_root(path_str):
    abs_path = os.path.abspath(path_str)
    parts = abs_path.split(os.sep)
    for idx, part in enumerate(parts):
        if part.startswith("map_"):
            return os.sep.join(parts[: idx + 1])
    return os.path.dirname(abs_path)


def load_input_data(file_path):
    if file_path.endswith(".propbin") or file_path.endswith(".propbin.gz"):
        return load_propbin(file_path)

    with gzip.open(file_path, "rb") as f:
        raw = pickle.load(f)
    if isinstance(raw, dict) and isinstance(raw.get("receivers"), np.ndarray):
        return restore_original_format(raw)
    return raw


def strip_known_suffix(filename):
    for suffix in (".pkl.gz", ".propbin.gz", ".propbin", ".pkl"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return os.path.splitext(filename)[0]


def process_file(input_path, name_suffix):
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        return

    map_id = parse_map_id(input_path)
    map_root = find_map_root(input_path)
    base_out = os.path.join(map_root, OUTPUT_ROOT_DIR_FMT.format(map_id=map_id))
    delay_root = os.path.join(base_out, "pdp")
    angle_root = os.path.join(base_out, "aps")
    for directory in (delay_root, angle_root):
        os.makedirs(directory, exist_ok=True)

    print(f"\n=== Processing {input_path} ===")
    print(f"Output root: {base_out}")

    print("1. Loading input data...")
    data = load_input_data(input_path)

    tx_loc = data.get("source_location", (0.0, 0.0, 0.0))
    all_receivers = list(data["receivers"].values())

    if SAMPLE_RECORD_SERVER_DIR:
        record_dir = os.path.join(
            SAMPLE_RECORD_SERVER_DIR,
            f"map_{map_id}",
            OUTPUT_ROOT_DIR_FMT.format(map_id=map_id),
        )
    else:
        record_dir = base_out

    print(f"   record_dir: {record_dir}")
    print(f"   JSON path:  {os.path.join(record_dir, SAMPLE_RECORD_FILENAME)} exists={os.path.exists(os.path.join(record_dir, SAMPLE_RECORD_FILENAME))}")
    sample_record = load_sample_record(record_dir, tx_loc)
    if sample_record is not None:
        samples = [
            rec for rec in all_receivers
            if (int(round(rec["location"][0])), int(round(rec["location"][1]))) in sample_record
        ]
        missing = len(sample_record) - len(samples)
        print(f"   loaded sample record from: {record_dir}")
        print(f"   matched receivers: {len(samples)}" +
              (f" (warning: {missing} recorded rx not found in this data)" if missing else ""))
    elif SAMPLE_RECORD_SERVER_DIR:
        print(f"   WARNING: SAMPLE_RECORD_SERVER_DIR is set but no JSON found at: {record_dir}")
        print(f"   Make sure GT has been processed first. Falling back to random sampling.")
        samples = random.Random(RECEIVER_SAMPLE_SEED).sample(all_receivers, min(NUM_SAMPLES, len(all_receivers)))
        save_sample_record(base_out, tx_loc, samples)
        print(f"   sampled receivers: {len(samples)} (record saved)")
    elif len(all_receivers) > NUM_SAMPLES:
        samples = random.Random(RECEIVER_SAMPLE_SEED).sample(all_receivers, NUM_SAMPLES)
        save_sample_record(base_out, tx_loc, samples)
        print(f"   sampled receivers: {len(samples)} (record saved to {SAMPLE_RECORD_FILENAME})")
    else:
        samples = all_receivers
        save_sample_record(base_out, tx_loc, samples)
        print(f"   receivers: {len(samples)} (all used, record saved to {SAMPLE_RECORD_FILENAME})")

    delay_seqs = []
    angle_seqs = []
    name_tags = []

    print("2. Building ADPS and sequences...")

    for i, rec in enumerate(samples):
        if i % 50 == 0:
            print(f"   progress: {i}/{len(samples)}...")

        paths = rec["path_info"]
        toas_s, doas_deg, rss_dbm = [], [], []

        for path in paths:
            if path["loss"] <= 0 or path["loss"] > MAX_VALID_LOSS:
                continue

            toas_s.append(path["delay"] * 1e-9)
            ang = path["arrival_angle"]
            a_rad = ang[0] if isinstance(ang, (list, tuple, np.ndarray)) else float(ang)
            doas_deg.append(np.degrees(a_rad))
            rss_dbm.append(1.0 - path["loss"])

        toas_s = np.array(toas_s)
        doas_deg = np.array(doas_deg)
        rss_dbm = np.array(rss_dbm)

        rx_loc = rec.get("location", (0.0, 0.0, 0.0))
        rx_x, rx_y = int(round(rx_loc[0])), int(round(rx_loc[1]))
        tx_x, tx_y = int(round(tx_loc[0])), int(round(tx_loc[1]))
        name_tag = f"{map_id}_{tx_x}_{tx_y}_{rx_x}_{rx_y}"
        name_tags.append(name_tag)

        if len(toas_s) == 0:
            delay_seqs.append(np.zeros((GRID_CONF["n_tau"], 2)))
            angle_seqs.append(np.zeros((GRID_CONF["n_theta"], 2)))
            continue

        s_linear, tau_axis, theta_axis = generate_adps(
            toas_s, doas_deg, rss_dbm, SYS_CONF, GRID_CONF
        )
        s_db = 10 * np.log10(s_linear + 1e-35)

        s_d, s_a = generate_sequences(s_db, tau_axis, theta_axis)
        delay_seqs.append(s_d)
        angle_seqs.append(s_a)

    for tag, s_d, s_a in zip(name_tags, delay_seqs, angle_seqs):
        np.save(
            os.path.join(delay_root, OUT_DELAY_SEQ_TPL.format(name=tag)),
            np.array(s_d, dtype=np.float16),
        )
        np.save(
            os.path.join(angle_root, OUT_ANGLE_SEQ_TPL.format(name=tag)),
            np.array(s_a, dtype=np.float16),
        )
    print("3. Done.")


def resolve_source_input(batch_dir, source_id):
    candidates = [
        f"source_{source_id}_propagation_data.pkl.gz",
        f"source_{source_id}.propbin.gz",
        f"source_{source_id}.propbin",
    ]
    for filename in candidates:
        path = os.path.join(batch_dir, filename)
        if os.path.exists(path):
            return path
    return None


def main():
    map_id_env = os.environ.get("CURRENT_MAP_ID", "50").strip()
    default_batch_dir = os.path.join(SERVER_DATA_DIR, f"map_{map_id_env}", "special_points_propbin_3.5GHz")

    start_map = os.environ.get("START_MAP_ID", "").strip()
    end_map = os.environ.get("END_MAP_ID", "").strip()
    if start_map and end_map:
        try:
            start_i = int(start_map)
            end_i = int(end_map)
        except ValueError:
            print("Error: START_MAP_ID / END_MAP_ID must be integers")
            return

        if start_i > end_i:
            start_i, end_i = end_i, start_i

        for map_id in range(start_i, end_i + 1):
            batch_dir = os.path.join(SERVER_DATA_DIR, f"map_{map_id}", "special_points_propbin_3.5GHz")
            if not os.path.isdir(batch_dir):
                print(f"Skip: {batch_dir} does not exist")
                continue

            print(f"Batch processing directory: {batch_dir}")
            for source_id in range(100):
                input_path = resolve_source_input(batch_dir, source_id)
                if input_path is None:
                    print(f"  missing: source_{source_id} input file (skip)")
                    continue
                process_file(input_path, f"source_{source_id}_propagation_data")
        return

    batch_dir = os.environ.get("SPECIAL_PKL_DIR", "").strip() or default_batch_dir

    if os.path.isdir(batch_dir):
        files = [
            filename
            for filename in os.listdir(batch_dir)
            if filename.endswith(".pkl.gz")
            or filename.endswith(".propbin.gz")
            or filename.endswith(".propbin")
        ]
        if not files:
            print(f"Directory {batch_dir} has no *.pkl.gz / *.propbin.gz / *.propbin files")
            return

        files.sort()
        print(f"Batch processing directory: {batch_dir}, total {len(files)} files")
        for filename in files:
            input_path = os.path.join(batch_dir, filename)
            name_suffix = strip_known_suffix(filename)
            process_file(input_path, name_suffix)
    else:
        print(f"Warning: {batch_dir} is not a valid directory, fallback to single-file mode")
        name_suffix = strip_known_suffix(os.path.basename(PKL_PATH))
        process_file(PKL_PATH, name_suffix)


if __name__ == "__main__":
    main()
