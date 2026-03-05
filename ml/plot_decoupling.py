import matplotlib.pyplot as plt
import numpy as np
import os

def plot_pose_gaze_decoupling():
    """
    Generates a scatter plot of head_pitch vs gaze_pitch to illustrate decoupling.
    """
    # Simulate data for visualization
    num_pts = 200
    
    # Genuine: Coupled
    h_p_gen = np.random.normal(0, 0.05, num_pts)
    g_p_gen = h_p_gen + np.random.normal(0, 0.02, num_pts)
    
    # Phone Usage: Decoupled
    h_p_phone = np.random.normal(0, 0.05, num_pts)
    g_p_phone = h_p_phone - 0.4 + np.random.normal(0, 0.05, num_pts)
    
    # Lateral (Side Monitor): Decoupled Yaw (represented as offset in pitch for simplicity here, 
    # but let's do Yaw/Pitch context)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(h_p_gen, g_p_gen, alpha=0.6, label='Genuine (Coupled)', color='teal')
    plt.scatter(h_p_phone, g_p_phone, alpha=0.6, label='Phone Usage (Decoupled)', color='crimson')
    
    plt.axline((0, 0), slope=1, color='black', linestyle='--', alpha=0.3, label='Perfect Alignment')
    
    plt.title('Head-Gaze Pitch Dynamics (Decoupling Detection)')
    plt.xlabel('Head Pitch (rad)')
    plt.ylabel('Gaze Pitch (rad)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_dir = '/Users/shyam/Desktop/ai-proctoring-ml/paper_figures'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'pose_gaze_scatter.png')
    plt.savefig(save_path, dpi=300)
    print(f"Decoupling plot saved to {save_path}")

if __name__ == "__main__":
    plot_pose_gaze_decoupling()
