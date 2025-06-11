

import numpy as np 
import matplotlib.pyplot as plt

from terrain_utils import load_terrain, terrain_elevation
from trajectory_optimizer import optimize_trajectory
from staticplot import plot_3d_trajectory, plot_power_components, plot_time_series, plot_velocity
from animate import animate_trajectory
from schedule_opt import compute_drone_schedule, display_schedule

from functools import partial

# ---------------------
# Parameters
# ---------------------
g = 9.81
m_uav = 5.0  # kg
A = 0.79     # disk area (m²)
cf = 0.13    # profile correction factor
s = 0.1      # rotor solidity
c_T = 0.3    # thrust coefficient based on disk area 
delta = 0.012  
rho = 1.225  # air density (kg/m³)
SFP = 0.01   # fuselage equivalent flat area
v_ref = 5.0  # wind velocity at reference height (m/s)
z_ref = 1000  # reference height for wind profile (m)
alpha = 0.28  # wind profile exponent
H_min = 10
v_max = 20 
dt = 1 # time step for the simulation
beta = 30 # angle of wind in degrees



def main():

    tx, ty, tz = load_terrain("./terrain_data_mendocino_30.npz")
    
    terrain_elev = partial(terrain_elevation, terrain_x=tx, terrain_y=ty, terrain_z=tz)

    # waypoints = np.array([
    #     [21823,82186,1760],
    #     [21216,81808,1846],
    #     [20561,81364,1741],
    #     [19278,80729,1465],
    #     [18000,80000,1250],
    # ])

    waypoints = np.array([
        [18000,80000,1250],
        [21823,82186,1760],
    ])


    

    # Uncomment to see the effect of switching the order of the waypoints 
    # waypoints = waypoints[::-1]

    results = optimize_trajectory(waypoints=waypoints, terrain_elev=terrain_elev, terrain_x=tx, terrain_y=ty, terrain_z=tz,
                                    H_min=H_min, v_max=v_max, 
                                    v_ref=v_ref, z_ref=z_ref, alpha=alpha, beta=beta,
                                    g=g, m_uav=m_uav, rho=rho, SFP=SFP, cf=cf, delta=delta, c_T=c_T, A=A, s=s
    )

    # unpack results 
    t_vals, x_vals, y_vals, z_vals, vx_vals, vy_vals, vz_vals, pc_vals, pd_vals, pi_vals, pb_vals, p_total_vals, z_floor_vals, energy_per_leg, total_energy, leg_idx = results 


    waypoints = np.append(waypoints, [waypoints[0]], axis=0) 

    # schedule = compute_drone_schedule(waypoints=waypoints, energy_per_leg=energy_per_leg, leg_idx=leg_idx, t_vals=t_vals)

    # display_schedule(schedule)


    plot_3d_trajectory(terrain_x=tx, terrain_y=ty, terrain_z=tz,
        x_vals=x_vals, y_vals=y_vals, z_vals=z_vals,
        waypoints=waypoints,
        all_leg_idx=leg_idx,
        use_leg_colors=True,
        x_min=16000, x_max=24000, y_min=78000, y_max=84000,
        views=[(30,225),(30,45),(30,135),(30,315)],
        output_file='results/trajectory3d_beta30_return.png'
    )

    animate_trajectory(x_vals, y_vals, z_vals, tx, ty, tz, t_vals=t_vals, p_total_vals=p_total_vals, output_file="./results/trajectory_beta30_return.mp4")

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
    plot_time_series(t_vals, z_vals, 'z[t]', 'Z Position (m)', 'z_position_beta30_return.png', all_leg_idx=leg_idx,  use_leg_colors=True)

    plot_time_series(t_vals, x_vals, 'x[t]', 'X Position (m)', 'x_position_beta30_return.png', all_leg_idx=leg_idx)

    plot_time_series(t_vals, y_vals, 'y[t]', 'Y Position (m)', 'y_position_beta30_return.png', all_leg_idx=leg_idx)

    plot_time_series(t_vals, pd_vals, 'pd[t]', 'Drag Power (W)', 'dragpower_beta30_return.png', all_leg_idx=None, use_leg_colors=False)
    

main()


