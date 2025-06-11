import numpy as np
from gekko import GEKKO 

distance_threshold = 1

def optimize_leg(
    leg_id, x1, y1, z1, x2, y2, z2,
    terrain_elev, terrain_x, terrain_y, terrain_z, H_min, v_max, 
    v_ref, z_ref, alpha, beta,
    g, m_uav, rho, SFP, cf, delta, c_T, A, s
):
    """
    Solve one leg from (x1,y1,z1)->(x2,y2,z2), returning time-series arrays.
    terrain_elev(x,y) must return ground height.
    """

    # Determine time grid based on distance 
    dist = np.linalg.norm([x2-x1, y2-y1, z2-z1])
    T = int(np.ceil(dist / (v_max*0.5)))
    N = T

    max_iters = 10
    convergence_tol = 1e-3  # Threshold for convergence
    converged = False

    print(f"Start point of leg {leg_id}: ({x1}, {y1}, {z1})")
    print(f"End point of leg {leg_id}: ({x2}, {y2}, {z2})")

    # -----------------
    # GEKKO model 
    # -----------------
    
    # initialize xy trajectory guess
    x_guess = np.linspace(x1, x2, N+1)
    y_guess = np.linspace(y1, y2, N+1)

    for iter in range(max_iters):

        # initialize clearance profile
        H_profile = np.full(N+1, H_min)

        # taper to 0 near the start and end (e.g. first/last 5 steps)
        for i in range(2):
            H_profile[i] = H_min * (i / 5)      # ramp up
            H_profile[-(i+1)] = H_min * (i / 5) # ramp down

        # get 
        z_floor_vals = np.array([terrain_elev(xi, yi) for xi, yi in zip(x_guess, y_guess)]) + H_profile

        z_floor_vals[0] = z1  # Ensure the first point is at the start height
        z_floor_vals[-1] = z2  # Ensure the last point is at the end height

        m = GEKKO(remote=False, name=f"model_{leg_id}")
        m.time = np.linspace(0, T, N+1)
        m.options.IMODE = 6  # Dynamic optimization (control)
        m.options.MAX_ITER = 100


        x = m.Var(value=x1, lb=terrain_x[0]+1, ub=terrain_x[-1]-1)  # position
        vx = m.Var(value=0, lb=-v_max, ub=v_max)  # x velocity
        m.Equation(x.dt() == vx)

        y = m.Var(value=y1, lb=terrain_y[0]+1, ub=terrain_y[-1]-1)  # position
        vy= m.Var(value=0, lb=-v_max, ub=v_max)  # y velocity
        m.Equation(y.dt() == vy)

        z = m.Var(value=z_floor_vals.tolist())  # or np.linspace(z1, z2, N+1)
        vz = m.Var(value=0, lb=-v_max, ub=v_max)  # z velocity
        m.Equation(z.dt() == vz)


        z_floor = m.Param(value=z_floor_vals)  # terrain elevation at each time step

        
        print(f'z is list: {isinstance(z, list)}')
        print(f'z value length: {len(z.value) if not isinstance(z, list) else "list of Vars, no .value"}')
        print(f'z_floor is Param: {hasattr(z_floor, "value")}')
        print(f'z_floor value length: {len(z_floor.value)}')


        m.Equation(z >= z_floor) 

        # -----------------------------
        # Wind velocity components
        # -----------------------------

        beta_rad = np.radians(beta)  # Convert beta to radians
        print(f"[DEBUG] beta (deg): {beta}, beta_rad: {beta_rad}, cos: {np.cos(beta_rad):.2f}, sin: {np.sin(beta_rad):.2f}")
        vw_mag = m.Intermediate(v_ref * (z / z_ref)**alpha)

        cos_beta = np.cos(beta_rad)
        sin_beta = np.sin(beta_rad)

        vw_x = m.Intermediate(vw_mag * cos_beta)  # Wind velocity in x direction
        vw_y = m.Intermediate(vw_mag * sin_beta)  # Wind velocity in y direction

        vx_rel = m.Intermediate(vx - vw_x)  # Relative x velocity
        vy_rel = m.Intermediate(vy - vw_y)  # Relative y velocity
        vz_rel = vz  # Relative z velocity (no wind in z direction)

        v_rel_mag = m.Intermediate(m.sqrt(vx_rel**2 + vy_rel**2 + vz_rel**2))  # Relative velocity magnitude

        v_mag = m.Intermediate(m.sqrt(vx**2 + vy**2 + vz**2))  # Total velocity magnitude

        tau_c_approx = m.Intermediate(vz / (v_mag + 1e-3))  # Approximation for climb angle

        # -----------------------------
        # Acceleration and Thrust components
        # -----------------------------


        ax = m.Var(value=0)  # x acceleration
        ay = m.Var(value=0)  # y acceleration
        az = m.Var(value=0)  # z acceleration

        m.Equation(ax == vx.dt())  # x acceleration equation
        m.Equation(ay == vy.dt())  # y acceleration equation
        m.Equation(az == vz.dt())  # z acceleration equation

        Tx = m.Intermediate(m_uav * ax + 0.5 * rho * SFP * v_rel_mag * vx_rel)  # x thrust
        Ty = m.Intermediate(m_uav * ay + 0.5 * rho * SFP * v_rel_mag * vy_rel)  # y thrust
        Tz = m.Intermediate(m_uav * az + 0.5 * rho * SFP * v_rel_mag * vz_rel - m_uav * g)  # z thrust
        # Thrust magnitude
        T_mag = m.Intermediate(m.sqrt(Tx**2 + Ty**2 + Tz**2))  # Thrust magnitude
        # -----------------------------

        # calculate Induced power 
        # Rotor parameters
        denom = (2 * rho * A) ** 2
        inside_root = m.sqrt((T_mag**2 / denom) + (v_rel_mag**4 / 4))
        bracket = m.sqrt(inside_root - (v_rel_mag**2) / 2 + 1e-3)  # ε for numerical safety

        pi = m.Intermediate((1 + cf) * T_mag * bracket)


        # Blade power 
        term1 = T_mag / (c_T * rho * A)
        term2 = 3 * v_rel_mag**2
        term3 = m.sqrt((rho * s**2 * A * T_mag) / c_T)

        pb = m.Intermediate((delta / 8) * (term1 + term2) * term3)



        m.Obj((x - x2)**2 + (y - y2)**2 + (z - z2)**2) # minimize distance to target
        pd = m.Intermediate(0.5 * rho * SFP * v_rel_mag**3)  # Drag related power
        # pc = m.Intermediate(m_uav * g * v_mag * m.sin(tau_c))  # Climb power
        pc = m.Intermediate(m_uav * g * v_mag * tau_c_approx)  # Climb power
        p_total = m.Intermediate(pc + pd + pi + pb)  # Total power

        
        m.Obj(p_total)  # Minimize climb and drag power


        try:
            m.solve(disp=True, debug=True)  # solve the optimization problem
            
            print("################################################################################################")
            # Print z_floor values (terrain floor over time)
            print("z_floor values:")
            print(z_floor.value)

            # Optionally, print side by side with z to compare clearance
            z_vals = np.array(z.value)
            z_floor_vals = np.array(z_floor.value)
            clearance = z_vals - z_floor_vals

            print("\nMin clearance: {:.2f} m".format(clearance.min()))


            print(f"[DEBUG] vw_x sample: {np.array(vw_x.value)[:10]}")
            print(f"[DEBUG] vx_rel sample: {np.array(vx_rel.value)[:10]}")
            print(f"[DEBUG] pd sample: {np.array(pd.value)[:10]}")


            print("################################################################################################")

             # Convergence check
            new_x_guess = np.array(x.value)
            new_y_guess = np.array(y.value)

            dx = np.linalg.norm(new_x_guess - x_guess)
            dy = np.linalg.norm(new_y_guess - y_guess)

            if dx < convergence_tol and dy < convergence_tol:
                print(f"Converged in {iter+1} iterations.")
                converged = True
                break

            x_guess = new_x_guess
            y_guess = new_y_guess

            if not m.options.SOLVER == 1:
                print(f"Solver did not converge in {iter+1} iterations. Retrying with updated guesses.")

            # Check if the solution is valid
        except Exception as e:
            print("An error occurred during optimization:", e)
            infeasibilities_path = os.path.join(m._path, 'infeasibilities.txt')
            if os.path.exists(infeasibilities_path):
                print(open(infeasibilities_path).read())
            break

    return x.value, y.value, z.value, vx.value, vy.value, vz.value, p_total.value, pc.value, pd.value, pi.value, pb.value, m.time, z_floor_vals

def optimize_trajectory(waypoints, terrain_elev, terrain_x, terrain_y, terrain_z,
                        H_min, v_max, 
                        v_ref, z_ref, alpha, beta,
                        g, m_uav, rho, SFP, cf, delta, c_T, A, s
):
    """
    waypoints: array of shape (M,3).  Calls optimize_leg for each segment,
    concatenates results, returns big arrays: x_vals,y_vals,z_vals,...,leg_idx
    """

    # Append first waypoint to generate return leg
    waypoints = np.append(waypoints, [waypoints[0]], axis=0) 

    # Initialize lists to store results
    all_x = []
    all_y = []
    all_z = []
    all_vx = []
    all_vy = []
    all_vz = []
    all_pc = []
    all_pd = []
    all_pi = []
    all_pb = []
    all_p_total = []
    all_t = []
    all_leg_idx = []
    all_z_floor = []
    # Initialize time offset for each leg
    current_t_offset = 0

    total_energy = 0.0
    energy_per_leg = []


    # Optimize each leg
    for i in range(len(waypoints) - 1):
        x1, y1, z1 = waypoints[i]
        x2, y2, z2 = waypoints[i + 1]

        print(f"Optimizing leg {i+1} from ({x1}, {y1}, {z1}) to ({x2}, {y2}, {z2})")
        results = optimize_leg(i+1, x1, y1, z1, x2, y2, z2, terrain_elev=terrain_elev, terrain_x=terrain_x, terrain_y=terrain_y, terrain_z=terrain_z,
                                    H_min=H_min, v_max=v_max, 
                                    v_ref=v_ref, z_ref=z_ref, alpha=alpha, beta=beta,
                                    g=g, m_uav=m_uav, rho=rho, SFP=SFP, cf=cf, delta=delta, c_T=c_T, A=A, s=s
        )
    
        if results:
            x_results, y_results, z_results, vx_results, vy_results, vz_results, p_total_results, pc_results, pd_results, pi_results, pb_results, time_results, z_floor_results = results
            
            x_vals = np.array(x_results)
            y_vals = np.array(y_results)
            z_vals = np.array(z_results)
            vx_vals = np.array(vx_results)
            vy_vals = np.array(vy_results)
            vz_vals = np.array(vz_results)
            pc_vals = np.array(pc_results)
            pd_vals = np.array(pd_results)
            pi_vals = np.array(pi_results)
            pb_vals = np.array(pb_results)
            p_total_vals = np.array(p_total_results)
            t_vals = np.array(time_results)

            # Calculate the distance from the end point along the trajectory
            dist_array = np.sqrt((x_vals - x2)**2 + (y_vals-y2)**2 + (z_vals-z2)**2)


            # --------- Extra debugging -----------------
            print(f"Leg {i+1} optimization results:")
            print(f"x_vals: {x_vals}")
            print(f"y_vals: {y_vals}")
            print(f"z_vals: {z_vals}")

            print(f"Length of dist_array:{len(dist_array)}")
            print(f"Distance array for Leg {i+1}: {dist_array}")
            # Find the index of the first point where the distance is less than the threshold
            # If no point is found, use the last point as the arrival point
            arrival_idx = np.where(dist_array < distance_threshold)[0]
            if len(arrival_idx) > 0:
                stop_idx = arrival_idx[0] + 1
                print(f"Arrival idx for Leg {i+1}: {arrival_idx[0]}")

            else:
                print(f"Warning: No arrival point found for leg {i+1}. Using last point as arrival.")
                stop_idx = len(x_vals) 

            if i == 0: 
                start_idx = 0
            else:
                start_idx = 1

            
            x_vals = x_vals[start_idx:stop_idx]  # Skip the first point to avoid duplication
            y_vals = y_vals[start_idx:stop_idx]  # Skip the first point to avoid duplication
            z_vals = z_vals[start_idx:stop_idx]
            vx_vals = vx_vals[start_idx:stop_idx]
            vy_vals = vy_vals[start_idx:stop_idx]
            vz_vals = vz_vals[start_idx:stop_idx]
            pc_vals = pc_vals[start_idx:stop_idx]
            pd_vals = pd_vals[start_idx:stop_idx]
            pi_vals = pi_vals[start_idx:stop_idx]
            pb_vals = pb_vals[start_idx:stop_idx]
            p_total_vals = p_total_vals[start_idx:stop_idx]
            t_vals = t_vals[start_idx:stop_idx] + current_t_offset
            z_floor_vals = z_floor_results[start_idx:stop_idx]
            
            
            all_x.extend(x_vals)
            all_y.extend(y_vals)
            all_z.extend(z_vals)
            all_vx.extend(vx_vals)
            all_vy.extend(vy_vals)
            all_vz.extend(vz_vals)
            all_pc.extend(pc_vals)
            all_pd.extend(pd_vals)
            all_pi.extend(pi_vals)
            all_pb.extend(pb_vals)
            all_p_total.extend(p_total_vals)
            all_t.extend(t_vals)
            all_z_floor.extend(z_floor_vals)
            current_t_offset = t_vals[-1]  # Update time offset 
            all_leg_idx.extend([i + 1] * len(x_vals))  # Store leg index for each point
            
            energy_leg = np.trapz(p_total_vals, x=t_vals)  # Calculate energy consumed for this leg
            total_energy += energy_leg
            energy_per_leg.append(energy_leg)
            print(f"Energy consumed for leg {i+1}: {energy_leg:.2f} J")
            print(f"Total energy consumed so far: {total_energy:.2f} J")
            

            print(f"Leg {i+1} optimization completed successfully.")
        else:
            print(f"Leg {i+1} optimization failed. Skipping to next leg.")


    t_vals = np.array(all_t)
    x_vals = np.array(all_x)
    y_vals = np.array(all_y)
    z_vals = np.array(all_z)
    vx_vals = np.array(all_vx)
    vy_vals = np.array(all_vy)
    vz_vals = np.array(all_vz)
    pc_vals = np.array(all_pc)
    pd_vals = np.array(all_pd)
    pi_vals = np.array(all_pi)
    pb_vals = np.array(all_pb)
    p_total_vals = np.array(all_p_total)
    z_floor_vals = np.array(all_z_floor)

    print("trajectory min and max: x_min, x_max, y_min, y_max")
    print(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max())


    return t_vals, x_vals, y_vals, z_vals, vx_vals, vy_vals, vz_vals, pc_vals, pd_vals, pi_vals, pb_vals, p_total_vals, z_floor_vals, energy_per_leg, total_energy, all_leg_idx

