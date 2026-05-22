import numpy as np
import matplotlib.pyplot as plt
import os
import glob

plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

base_dir = r"ROW_DATA\df"

def load_all_npz(path):
    files = sorted(glob.glob(os.path.join(path, "*.npz")))
    all_data = {}
    for f in files:
        data = np.load(f)
        for key in data.keys():
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(data[key])
    for key in all_data:
        arr = all_data[key][0]
        if arr.ndim == 0:
            all_data[key] = np.array(all_data[key])
        else:
            all_data[key] = np.concatenate(all_data[key], axis=0)
    return all_data

def plot_stand_sensor(stand_name, sensor_name, save_path):
    path = os.path.join(base_dir, stand_name, sensor_name, "npz_datasets")
    data = load_all_npz(path)

    n_rows = 3
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 14))
    fig.suptitle(f"{stand_name} — {sensor_name.upper()}", fontsize=16, fontweight='bold')

    magnetogram = data['magnetogram']
    velocity = data['velocity']
    accel = data['accelerometer']
    orientation = data['orientation']
    n_steps = magnetogram.shape[0]
    n_channels = magnetogram.shape[1]

    # 1. Magnetogram heatmap (first 500 steps)
    ax = axes[0, 0]
    limit = min(500, n_steps)
    im = ax.imshow(magnetogram[:limit, :], aspect='auto', cmap='viridis',
                   extent=[0, n_channels, limit, 0])
    ax.set_xlabel('Channel')
    ax.set_ylabel('Step')
    ax.set_title(f'Magnetogram Heatmap ({n_channels} ch, {n_steps} steps)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2. Magnetogram stats per step
    ax = axes[0, 1]
    mag_mean = np.mean(magnetogram, axis=1)
    mag_std = np.std(magnetogram, axis=1)
    ax.fill_between(range(n_steps), mag_mean - mag_std, mag_mean + mag_std,
                    alpha=0.3, color='steelblue')
    ax.plot(mag_mean, color='steelblue', linewidth=0.8, label='mean')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title('Magnetogram: Mean ± Std per Step')
    ax.legend()

    # 3. Velocity over time
    ax = axes[1, 0]
    ax.plot(velocity, color='coral', linewidth=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Velocity')
    ax.set_title('Velocity Over Time')

    # 4. Accelerometer 3 axes
    ax = axes[1, 1]
    ax.plot(accel[:, 0], label='X', linewidth=0.7)
    ax.plot(accel[:, 1], label='Y', linewidth=0.7)
    ax.plot(accel[:, 2], label='Z', linewidth=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Acceleration')
    ax.set_title('Accelerometer (3 axes)')
    ax.legend()

    # 5. Orientation distribution
    ax = axes[2, 0]
    unique, counts = np.unique(orientation, return_counts=True)
    ax.bar(unique.astype(str), counts, color='mediumseagreen')
    ax.set_xlabel('Orientation Value')
    ax.set_ylabel('Count')
    ax.set_title('Orientation Distribution')
    ax.tick_params(axis='x', rotation=45)

    # 6. Summary stats
    ax = axes[2, 1]
    ax.axis('off')
    stats = (
        f"=== Summary ===\n\n"
        f"Files loaded: {len(glob.glob(os.path.join(path, '*.npz')))}\n"
        f"Total steps: {n_steps}\n"
        f"Magnetogram channels: {n_channels}\n"
        f"Odom step: {data['odomstep'][0]:.6f}\n\n"
        f"Velocity:\n"
        f"  min={velocity.min()}, max={velocity.max()}, mean={velocity.mean():.1f}\n\n"
        f"Accelerometer:\n"
        f"  X: min={accel[:,0].min()}, max={accel[:,0].max()}\n"
        f"  Y: min={accel[:,1].min()}, max={accel[:,1].max()}\n"
        f"  Z: min={accel[:,2].min()}, max={accel[:,2].max()}\n\n"
        f"Orientation unique values: {len(unique)}"
    )
    ax.text(0.1, 0.5, stats, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

os.makedirs("visualizations", exist_ok=True)

combos = [
    ("stand-1_Sar-2025", "mdr", "stand1_sar_2025_mdr.png"),
    ("stand-1_Sar-2025", "umdp", "stand1_sar_2025_umdp.png"),
    ("stand-2_Bog-2026", "mdr", "stand2_bog_2026_mdr.png"),
    ("stand-2_Bog-2026", "umdp", "stand2_bog_2026_umdp.png"),
]

for stand, sensor, fname in combos:
    plot_stand_sensor(stand, sensor, os.path.join("visualizations", fname))

print("Done! Check the 'visualizations/' folder.")
