import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Style
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})

def ar1_signal(mean, std, length=120, phi=0.85, rng=None):
    if rng is None: rng = np.random.default_rng()
    noise = rng.normal(0, std * np.sqrt(1 - phi**2), size=length)
    signal = np.zeros(length)
    signal[0] = rng.normal(mean, std)
    for t in range(1, length):
        signal[t] = phi * signal[t-1] + (1 - phi) * mean + noise[t]
    return signal

def plot_gaze_patterns():
    print("Generating Gaze Pattern Visualizations...")
    rng = np.random.default_rng(42)
    t = np.arange(120)
    
    # 1. Genuine
    gen_yaw = ar1_signal(0, 2, phi=0.9, rng=rng)
    gen_pitch = ar1_signal(0, 2, phi=0.9, rng=rng)
    gen_vel = np.abs(np.diff(gen_yaw, prepend=0)) + np.abs(np.diff(gen_pitch, prepend=0))
    
    # 2. Scanning (oscillating yaw)
    scan_t = np.linspace(0, 4 * np.pi, 120)
    scan_yaw = 20 * np.sin(scan_t) + rng.normal(0, 2, 120)
    scan_pitch = ar1_signal(0, 5, phi=0.85, rng=rng)
    scan_vel = np.abs(np.diff(scan_yaw, prepend=0)) + np.abs(np.diff(scan_pitch, prepend=0))
    
    # 3. Phone (downward pitch)
    phone_yaw = ar1_signal(0, 5, phi=0.85, rng=rng)
    phone_pitch = ar1_signal(-25, 5, phi=0.9, rng=rng)
    phone_vel = np.abs(np.diff(phone_yaw, prepend=0)) + np.abs(np.diff(phone_pitch, prepend=0))

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    # Plot Yaw
    axes[0].plot(t, gen_yaw, label='Genuine (Stable)', color='green', lw=1.5)
    axes[0].plot(t, scan_yaw, label='Scanning (Cheat)', color='orange', lw=1.5)
    axes[0].plot(t, phone_yaw, label='Phone (Cheat)', color='red', lw=1.5)
    axes[0].set_ylabel('Gaze Yaw (deg)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(alpha=0.3)
    
    # Plot Pitch
    axes[1].plot(t, gen_pitch, color='green', lw=1.5)
    axes[1].plot(t, scan_pitch, color='orange', lw=1.5)
    axes[1].plot(t, phone_pitch, color='red', lw=1.5)
    axes[1].set_ylabel('Gaze Pitch (deg)')
    axes[1].grid(alpha=0.3)
    
    # Plot Velocity
    axes[2].plot(t, gen_vel, color='green', lw=1.5, alpha=0.5)
    axes[2].plot(t, scan_vel, color='orange', lw=2.0)
    axes[2].plot(t, phone_vel, color='red', lw=1.5, alpha=0.5)
    axes[2].set_ylabel('Gaze Velocity')
    axes[2].set_xlabel('Frame (t)')
    axes[2].grid(alpha=0.3)
    
    plt.suptitle('Temporal Gaze Behavioral Patterns', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = 'paper_figures/gaze_patterns.png'
    os.makedirs('paper_figures', exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"✅ Gaze patterns plot saved to {out_path}")

if __name__ == "__main__":
    plot_gaze_patterns()
