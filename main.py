

import numpy as np 
import matplotlib.pyplot as plt

from terrain_utils import load_terrain, terrain_elevation
from trajectory_optimizer import optimize_trajectory
from staticplot import plot_3d_trajectory, plot_power_components, plot_time_series, plot_velocity
from animate import animate_trajectory

from functools import partial

def main():

    tx, ty, tz = load_terrain("./terrain_data_mendocino_30.npz")
    
    terrain_elev = partial(terrain_elevation, terrain_x=tx, terrain_y=ty, terrain_z=tz)

    waypoints = np.array([
        [21823,82186,1760],
        [21216,81808,1846],
        [20561,81364,1741],
        [19278,80729,1465],
        [18000,80000,1250],
    ])

    results = optimize_trajectory(waypoints=waypoints, terrain_elev=terrain_elev, terrain_x=tx, terrain_y=ty, terrain_z=tz)

    # unpack results 
    t_vals, x_vals, y_vals, z_vals, vx_vals, vy_vals, vz_vals, pc_vals, pd_vals, pi_vals, pb_vals, p_total_vals, z_floor_vals, energy_per_leg, total_energy, leg_idx = results 

    plot_3d_trajectory(terrain_x=tx, terrain_y=ty, terrain_z=tz,
        x_vals=x_vals, y_vals=y_vals, z_vals=z_vals,
        waypoints=waypoints,
        all_leg_idx=leg_idx,
        use_leg_colors=True,
        x_min=16000, x_max=24000, y_min=78000, y_max=84000,
        views=[(30,225),(30,45),(30,135),(30,315)],
        output_file='results/trajectory3d_refactored.png'
    )

    animate_trajectory(x_vals, y_vals, z_vals, tx, ty, tz, t_vals=t_vals, p_total_vals=p_total_vals, output_file="./results/trajectory_refactored.mp4")

    # -----------------------------
    # Velocity Plot
    # -----------------------------
    plot_velocity(t_vals=t_vals, vx_vals=vx_vals, vy_vals=vy_vals, vz_vals=vz_vals)

    # -----------------------------
    # Power Components Plot
    # -----------------------------
    plot_power_components(t_vals=t_vals, pc_vals=pc_vals, pd_vals=pd_vals, pi_vals=pi_vals, pb_vals=pb_vals)

    # -----------------------------
    # Plot vertical height of drone
    # -----------------------------
    plot_time_series(t_vals, z_vals, 'z[t]', 'Z Position (m)', 'z_position_refactored.png', all_leg_idx=leg_idx,  use_leg_colors=True)

    plot_time_series(t_vals, x_vals, 'x[t]', 'X Position (m)', 'x_position_refactored.png', all_leg_idx=leg_idx)

    plot_time_series(t_vals, y_vals, 'y[t]', 'Y Position (m)', 'y_position_refactored.png', all_leg_idx=leg_idx)

main()


