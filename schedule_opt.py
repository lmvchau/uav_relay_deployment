import numpy as np 
from hover_power import compute_hover_power
import pandas as pd 

def display_schedule(schedule):
    df = pd.DataFrame(schedule)
    df = df[["drone", "launch_time", "arrival_time", "hover_time", "return_time", "energy_used", "feasible"]]
    pd.set_option('display.float_format', '{:,.1f}'.format)
    print(df)
    df.to_csv("./results/schedule.csv")


def create_scheduling_dictionaries(leg_idx, energy_per_leg, t_vals):
    E_leg_dict = {}
    T_leg_dict = {}

    unique_legs = np.unique(leg_idx)

    for leg_num in unique_legs:
        start = leg_num - 1         # adjust 1-based index to 0-based
        end = start + 1

        # Energy for this leg
        E_leg_dict[(start, end)] = energy_per_leg[leg_num - 1]

        # Time for this leg
        t_leg = t_vals[leg_idx == leg_num]
        T_leg_dict[(start, end)] = t_leg[-1] - t_leg[0] if len(t_leg) >= 2 else 0
    
    return E_leg_dict, T_leg_dict


def compute_drone_schedule(waypoints, energy_per_leg, leg_idx, t_vals, E_battery=250000,
                            hover_params=None):

    """
    params: 
    - waypoints: np.array of shape (n, 3), each row is (x, y, z) points 
    - hover_points: list of aypoint indices to hover at 
    - E_leg_dict: dict mapping (i, j) to energy in Joules, where i and j are waypoint indices
    - T_leg_dict: dict mapping (i, j) to time in seconds of each leg
    - E-battery: total battery capacity in Joules for each drone battery (assume each drone has same battery capacity)
    - hover_params: dict, additioanl params passed to compute_hover_power() function
    returns:
    - schedule: list of dicts: one per drone, with launch time, hover duration, total energy 
    """

    assert waypoints.shape[0] >= 3, "Must have at least 3 waypoints (launch, target, return)"
    assert (waypoints[0] == waypoints[-1]).all(), "First and last waypoint are not the same"

    E_leg_dict, T_leg_dict = create_scheduling_dictionaries(leg_idx=leg_idx, energy_per_leg=energy_per_leg, t_vals=t_vals)

    if hover_params is None: 
        hover_params = {}

    num_waypoints = len(waypoints)
    num_drones = num_waypoints - 1

    # create list of waypoint indices to hover at
    hover_points = list(range(1, len(waypoints)-1))

    # for every waypoint from index 1 to 2nd to last, compute hover power
    

    return_leg = (num_waypoints - 2, num_waypoints -1)
    energy_return = E_leg_dict[return_leg]
    time_return = T_leg_dict[return_leg]

    forward_legs = [(i, i+1) for i in range(num_waypoints -2)]

    energy_forward = sum(E_leg_dict[leg] for leg in forward_legs)
    time_forward = sum(T_leg_dict[leg] for leg in forward_legs)

    hover_powers = {}
    for h in hover_points: 
        pos = waypoints[h]
        hover_powers[h] = compute_hover_power(pos, **hover_params)

    schedule = []

     # First drone
    final_hover_idx = hover_points[-1]
    P_hover = hover_powers[final_hover_idx]
    E_available_for_hover = E_battery - energy_forward - energy_return
    t_hover = E_available_for_hover / P_hover

    launch_time = 0
    arrival_time = time_forward
    return_time = arrival_time + t_hover + time_return
    E_used = energy_forward + energy_return + t_hover * P_hover

    schedule.append({
        'drone': 0,
        'launch_time': launch_time,
        'arrival_time': arrival_time,
        'hover_time': t_hover,
        'return_time': return_time,
        'energy_used': E_used,
        'feasible': E_used <= E_battery
    })

    for i in range(1, num_drones):
        prev = schedule[i - 1]

        launch_time = prev['arrival_time'] + prev['hover_time']
        arrival_time = launch_time + time_forward

        # New constraint: can't return until previous drone has returned
        earliest_departure = prev['return_time']
        departure_time = max(arrival_time + 60, earliest_departure)
        t_hover = departure_time - arrival_time

        P_hover = hover_powers[final_hover_idx]
        E_used = energy_forward + energy_return + t_hover * P_hover
        return_time = departure_time + time_return

        schedule.append({
            'drone': i,
            'launch_time': launch_time,
            'arrival_time': arrival_time,
            'hover_time': t_hover,
            'return_time': return_time,
            'energy_used': E_used,
            'feasible': E_used <= E_battery
        })

    return schedule