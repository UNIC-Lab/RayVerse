"""
Visualize GT APS and PDP from .npy files.
Randomly picks 30 samples (same map, different TX) and outputs 60 PNGs.

Usage:
    python visualize_adps_gt.py --root D:/path/to/adps_root --out D:/2D/gt_vis
    python visualize_adps_gt.py --root D:/path/to/adps_root --out D:/2D/gt_vis --map 630
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif', 'FreeSerif', 'STIXGeneral']
rcParams['mathtext.fontset'] = 'stix'

GRID_CONF = {
    "theta_min": -180, "theta_max": 180, "n_theta": 180,
    "tau_min_ns": 0,   "tau_max_ns": 1000,
}


def plot_pdp(npy_path, out_path):
    seq = np.load(npy_path).astype(np.float32)
    n_tau = seq.shape[0]
    tau_ns = np.linspace(GRID_CONF["tau_min_ns"], GRID_CONF["tau_max_ns"], n_tau)
    power  = seq[:, 0]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    ax.plot(tau_ns, power, color='#f57c6e', linewidth=1.5)
    ax.set_xlabel('Delay (ns)', fontsize=14)
    ax.set_ylabel('Power (dBm)', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#CCCCCC')
    ax.yaxis.grid(True, linestyle='--', alpha=0.45, color='#DDDDDD')
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_aps(npy_path, out_path):
    seq = np.load(npy_path).astype(np.float32)
    n_theta = seq.shape[0]
    theta   = np.linspace(GRID_CONF["theta_min"], GRID_CONF["theta_max"], n_theta)
    power   = seq[:, 0]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
    ax.plot(theta, power, color='#71b8ed', linewidth=1.5)
    ax.set_xlabel('Arrival Angle (deg)', fontsize=14)
    ax.set_ylabel('Power (dBm)', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#CCCCCC')
    ax.yaxis.grid(True, linestyle='--', alpha=0.45, color='#DDDDDD')
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def find_subdir(root, name):
    """Try gt/<name> first, then <name> directly under root."""
    for d in [os.path.join(root, "gt", name), os.path.join(root, name)]:
        if os.path.isdir(d):
            return d
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="Root dir containing pdp/ and aps/ subdirs")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <root>/gt_vis)")
    parser.add_argument("--map", default=None,
                        help="Filter by map ID prefix, e.g. 630")
    parser.add_argument("-n", type=int, default=30,
                        help="Number of samples to pick (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    pdp_dir = find_subdir(args.root, "pdp")
    aps_dir = find_subdir(args.root, "aps")
    if not pdp_dir:
        print(f"Error: pdp dir not found under {args.root}")
        return

    out_dir = args.out or os.path.join(args.root, "gt_vis")
    out_pdp = os.path.join(out_dir, "pdp")
    out_aps = os.path.join(out_dir, "aps")
    os.makedirs(out_pdp, exist_ok=True)
    os.makedirs(out_aps, exist_ok=True)

    # Scan all pdp npy files
    all_files = sorted(f for f in os.listdir(pdp_dir)
                       if f.startswith("pdp_") and f.endswith(".npy"))

    # Filter by map ID if specified
    if args.map:
        all_files = [f for f in all_files if f.startswith(f"pdp_{args.map}_")]

    if len(all_files) == 0:
        print("No matching PDP files found.")
        return

    # Group by TX (extract unique TX coords from filename)
    # Filename: pdp_<map>_<src>_<tx_x>_<tx_y>_<rx>.npy
    # Pick one sample per TX to ensure different TX
    tx_groups = {}
    for f in all_files:
        name = f[len("pdp_"):-len(".npy")]
        parts = name.split("_")
        if len(parts) >= 4:
            tx_key = "_".join(parts[:4])  # map_src_tx_x_tx_y
        else:
            tx_key = name
        tx_groups.setdefault(tx_key, []).append(f)

    # From each TX group, pick one random sample
    rng = np.random.RandomState(args.seed)
    candidates = []
    for tx_key, files in tx_groups.items():
        candidates.append(rng.choice(files))

    # Randomly pick n from candidates
    n = min(args.n, len(candidates))
    chosen = rng.choice(candidates, size=n, replace=False)

    print(f"Found {len(all_files)} total files, {len(tx_groups)} unique TX groups, picking {n}")

    ok = 0
    for f in chosen:
        name = f[len("pdp_"):-len(".npy")]

        pdp_npy = os.path.join(pdp_dir, f)
        aps_npy = os.path.join(aps_dir, f"aps_{name}.npy") if aps_dir else None

        plot_pdp(pdp_npy, os.path.join(out_pdp, f"pdp_{name}.png"))

        if aps_npy and os.path.exists(aps_npy):
            plot_aps(aps_npy, os.path.join(out_aps, f"aps_{name}.png"))
        else:
            print(f"  [skip] APS not found for {name}")

        ok += 1
        if ok % 10 == 0:
            print(f"  progress: {ok}/{n}")

    print(f"Done. {ok} samples → {out_dir}")


if __name__ == "__main__":
    main()
