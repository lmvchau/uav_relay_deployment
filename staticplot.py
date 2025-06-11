import math
import numpy as np
import matplotlib.pyplot as plt
from terrain_utils import crop_terrain


def plot_3d_trajectory(
    terrain_x: np.ndarray,
    terrain_y: np.ndarray,
    terrain_z: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    z_vals: np.ndarray,
    waypoints: np.ndarray,
    all_leg_idx: np.ndarray,
    use_leg_colors: bool = True,
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None,
    views: list = [(30, 225), (30, 45), (30, 135), (30, 315)],
    output_file: str = "./results/trajectory3d.png",
):

    if x_min is None: x_min = np.min(terrain_x) 
    if x_max is None: x_max = np.max(terrain_x) 
    if y_min is None: y_min = np.min(terrain_y)
    if y_max is None: y_max = np.max(terrain_y)

    terrain_x, terrain_y, terrain_z = crop_terrain(terrain_x=terrain_x, terrain_y=terrain_y, terrain_z=terrain_z,
                                                        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    # if x_min is not None and x_max is not None:
    # # Crop terrain if bounds are provided
    # if x_min is not None and x_max is not None:
    #     x_mask = (terrain_x >= x_min) & (terrain_x <= x_max)
    #     terrain_x = terrain_x[x_mask]

    # if y_min is not None and y_max is not None:
    #     y_mask = (terrain_y >= y_min) & (terrain_y <= y_max)
    #     terrain_y = terrain_y[y_mask]
    #     terrain_z = terrain_z[y_mask, :][:, x_mask]
    
    max_z = np.max(terrain_z)
    min_z = np.min(terrain_z)

    n = len(views)
    cols = 2 
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows,cols, figsize=(10*cols, 10*rows), subplot_kw={'projection': '3d'}, constrained_layout=False)
    
    axes = axes.ravel()

    # if n == 1:
    #     axes = [axes]  # Ensure axes is a list for single subplot case

    X, Y = np.meshgrid(terrain_x, terrain_y)

    if use_leg_colors:
        leg_idx = np.array(all_leg_idx)
        unique_legs = np.unique(leg_idx)
        cmap = plt.cm.get_cmap('tab10', len(unique_legs))


    for idx, (elev, azim) in enumerate(views):
        ax = axes[idx]
        ax.plot_surface(X, Y, terrain_z, color='lightgreen', alpha=0.75, edgecolor='none', zorder=0)
        ax.plot_wireframe(X, Y, terrain_z, color='gray', alpha=0.5, linewidth=0.5, zorder=1)


        x_range = terrain_x.max() - terrain_x.min()
        y_range = terrain_y.max() - terrain_y.min()
        z_range = terrain_z.max()  - terrain_z.min()

        ax.set_box_aspect((x_range, y_range, z_range))

        # --------------
        # UAV trajectory path
        #---------------

        if use_leg_colors:
            for leg_id in unique_legs:
                indices = np.where(leg_idx == leg_id)
                ax.plot(x_vals[indices], y_vals[indices], z_vals[indices]+1,
                        label=f'Leg {leg_id}', color=cmap(leg_id - 1), linewidth=4,
                        zorder=10)
        else:
            ax.plot(x_vals, y_vals, z_vals, label='UAV Trajectory', color='blue', linewidth=5)

        wp_x = waypoints[:, 0]
        wp_y = waypoints[:, 1]
        wp_z = waypoints[:, 2] # Offset waypoints slightly above terrain for visibility
    
        ax.scatter(wp_x[0], wp_y[0], wp_z[0], color='black', marker='o', s=150, label='Start Point', zorder=50)
        ax.scatter(wp_x[-2], wp_y[-2], wp_z[-2], color='red', marker='^', s=150, label='End Point', zorder=50)

        wp_x = wp_x[1:-2]  # Exclude start and end points for waypoints
        wp_y = wp_y[1:-2]
        wp_z = wp_z[1:-2]  # Exclude start and end points for waypoints

         # Plot waypoints -- add 20 to height for visibility 
        ax.scatter(wp_x, wp_y, wp_z + 20, color='blue', marker='o', s=150, label='Waypoints', zorder=50)


        # ax.set_box_aspect([terrain_x[-1] - terrain_x[0], 
        #                 (terrain_y[-1]- terrain_y[0]), 
        #                 dz*num_levels * 2])
        ax.set_zlim(bottom=min_z, top=max_z + 500)
        ax.set_xlabel("X (m)", fontsize="15")
        ax.set_ylabel("Y (m)", fontsize="15")
        ax.set_zlabel("Elevation (m)", fontsize="15")
        ax.set_title("Voxelized Terrain and UAV Trajectory")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"View: Elev={elev}°, Azim={azim}°", fontsize=20)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=20, bbox_to_anchor=(0.5, 0.92))
    
    for ax in axes[n:]:
        ax.set_visible(False)  # Hide unused subplots

    fig.suptitle('3D Trajectory Optimization with Terrain Collision Avoidance', fontsize=40, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])  # Adjust layout to fit the title
    plt.subplots_adjust(top=0.83,
                        bottom=0.01,
                        left=0.05,
                        right=0.95,
                        wspace=0.15,
                        hspace=0.05)  # Adjust top margin to fit the title

    plt.savefig(output_file)
    plt.show()
    plt.close()



# -----------------------------
# 1D Position Plots
# -----------------------------

def plot_time_series(t_vals, data, label, ylabel, filename, all_leg_idx: np.ndarray,
    use_leg_colors: bool = True):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if use_leg_colors:
        leg_idx = np.array(all_leg_idx)
        unique_legs = np.unique(leg_idx)
        cmap = plt.cm.get_cmap('tab10', len(unique_legs))


        for leg_id in unique_legs:
            indices = np.where(leg_idx == leg_id)
            plt.plot(t_vals[indices], data[indices],
                    label=f'Leg {leg_id}', color=cmap(leg_id - 1), linewidth=2)
        
    else:
        plt.plot(t_vals, data, label=label)
    # plt.plot(t_vals, data, label=label)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(label)
    plt.grid(True)
    plt.legend(loc = 'upper right')


    plt.savefig(f'./results/{filename}')
    plt.close()

def plot_velocity(t_vals, vx_vals, vy_vals, vz_vals, output_file='./results/velocity.png'):
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, vx_vals, label='vx[t]')
    plt.plot(t_vals, vy_vals, label='vy[t]')
    plt.plot(t_vals, vz_vals, label='vz[t]')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Profile Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    plt.close()

def plot_power_components(t_vals, pc_vals, pd_vals, pi_vals, pb_vals, output_file="./results/power_components.png"):
   
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, pc_vals, label='$P_{climb}$')
    plt.plot(t_vals, pd_vals, label='$P_{drag}$')
    plt.plot(t_vals, pi_vals, label='$P_{induced}$')
    plt.plot(t_vals, pb_vals, label='$P_{blade}$')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (W)')
    plt.title('Power Components Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    plt.close()