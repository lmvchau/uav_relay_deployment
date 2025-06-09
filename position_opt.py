import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


np.random.seed(41)

# Constants
#------------------------------------

rho = 1.225           # kg/m^3, air density
A = 0.79              # m^2, rotor disk area
cf = 0.13             #
s = 0.1               #
S_FP = 0.01           # frontal area
c_T = 0.3             # thrust coefficient
delta = 0.012         # induced power factor
g = 9.81              # m/s^2, gravitational acceleration
m = 5.0               # kg, UAV mass
P_T_max = 1.0         # W, Transmission Power Mass
SNR_min_dB = 20       # dB, minimum SNR for each link
v_ref = 10            # m/s, reference wind velocity at reference height
z_ref = 100           # m, reference height
H_min = 50            # minimum height
lambda_c = 0.1304     # carrier wavelength --> 2.3 GHz
K = 1.38e-23          # J/K, Boltzmann's constant
T = 320               # K, temperature
B = 100e6             # Hz, 100MHz
G_T = 1               # Transmitter antenna gain
G_R = 1               # Receiver antenna gain
alpha = 0.28          # Hellmann Power Law, assuming forest lowlands surface cover, from reference 3
beta = np.radians(30) # Wind angle in radians (30 degrees)
sigma_dB = 8          # sigma_dB in range [4, 12]dB from reference 4
chi_sigma_dB = np.random.normal(0, scale=sigma_dB)      # dB, normally distributed shadowing loss
p = 2.1

# -------------------------------
# Initialization
# -------------------------------
# Known positions
xG, yG = 21832, 82186  # fixed UAV position (ground station)
# x1, y1 = 16288, 82582  # ground station
x1, y1 = 18000, 80000

data = np.load("./terrain_data_mendocino_30.npz")
z_meters = data["z"]
x_meters = data["x"]
y_meters = data["y"]

y_meters = np.flipud(y_meters)
z_meters = np.flipud(z_meters)

terrain_x = x_meters[0, :]  # 1D array of x coords
terrain_y = y_meters[:, 0]  # 1D array of y coords
terrain_z = z_meters  # DEM shape = (rows, cols)

x_min, x_max = 0, 40000
y_min, y_max = 0, 140000

x_bounds = (terrain_x[0], terrain_x[-1])
y_bounds = (terrain_y[0], terrain_y[-1])

# Find the closest indices in the x_meters and y_meters grids that correspond to the region
x_indices = np.where((x_meters[0, :] >= x_min) & (x_meters[0, :] <= x_max))[0]
y_indices = np.where((y_meters[:, 0] >= y_min) & (y_meters[:, 0] <= y_max))[0]

# Filter z_meters to get the terrain data within the region
filtered_z_meters = z_meters[y_indices, :][:, x_indices]

# z_max = np.max(filtered_z_meters)
z_max = np.max(z_meters)

# ==============================================
# Helper functions
# ==============================================

def norm(v):
    return np.linalg.norm(v)

def induced_power(T_vec):
    return ((1 + cf) * norm(T_vec)**2) / (2 * rho * A)

def blade_power(T_vec):
    T_norm = norm(T_vec)
    base = rho * s**2 * A * T_norm / c_T
    if base < 0:
        print(" Negative base in sqrt:", base, "| T_vec =", T_vec)
    return (delta / 8) * (T_norm / (c_T * rho * A)) * np.sqrt(max(base, 0))

def wind_velocity(z):
    z_ratio = z / z_ref
    if z_ratio <= 0:
        print(f" z/z_ref is nonpositive: z = {z}, z_ref = {z_ref}")
        z_ratio = 1e-6
    return v_ref * z_ratio**alpha

def total_hover_power(position):
    x, y, z = position
    v_wind = wind_velocity(z)
    v_w = np.array([v_wind * np.cos(beta), v_wind * np.sin(beta), 0])
    v_w_norm = norm(v_w)

    Fx = 0.5 * rho * S_FP * v_w_norm**2 * np.cos(beta)
    Fy = -0.5 * rho * S_FP * v_w_norm**2 * np.sin(beta)
    Fz = m * g
    T_vec = np.array([Fx, Fy, Fz])
    return blade_power(T_vec) + induced_power(T_vec) + (1/2) * rho * S_FP * v_w_norm**3

def compute_mmin(pos_g, pos_1):
    d0 = np.linalg.norm(np.array(pos_1) - np.array(pos_g))  # total distance
    snr_min_linear = 10 ** (SNR_min_dB / 10)
    shadowing_linear = 10 ** (chi_sigma_dB / 20)

    numerator = 4 * np.pi * d0
    denom = lambda_c

    scaling_factor = np.sqrt((K * T * B * snr_min_linear) / (P_T_max * G_T * G_R))

    M_min = math.ceil((numerator / denom) * scaling_factor * shadowing_linear)
    return M_min

def tx_power(pos_i, pos_next):
    d = norm(pos_next - pos_i)
    snr_linear = 10**(SNR_min_dB / 10)
    return (K * T * B * snr_linear * (4 * np.pi * d / lambda_c) ** 2) /(G_T * G_R)

def initialize_positions(start, end, n_uavs, z_max):
    positions = []
    for i in range(n_uavs):
        frac = i / (n_uavs - 1)
        x = start[0] + frac * (end[0] - start[0])
        y = start[1] + frac * (end[1] - start[1])
        terrain_z = terrain_elevation(x, y)
        z = terrain_z + H_min + 10
        # z = z_max + H_min
        positions.append([x, y, z])
    return np.array(positions)

def terrain_elevation(x, y):
    xi = (np.abs(terrain_x - x)).argmin()
    yi = (np.abs(terrain_y - y)).argmin()
    return terrain_z[yi, xi]

# -------------------------------
# Objective Function
# -------------------------------

def total_power(position_flat):
    positions = position_flat.reshape((-1, 3))
    positions = np.vstack((np.array([x1, y1, z1]), positions, [xG, yG, zG]))
    total = 0
    for i in range(len(positions) - 1):
        pt = tx_power(positions[i], positions[i+1])
        ph = total_hover_power(positions[i])
        total += pt + ph
    return total

# ---------------------------------
# Constraints
#----------------------------------

def make_constraints(n_uavs):
    cons = []

    n_uav_opt = n_uavs - 1  # Number of UAVs to optimize (excluding the fixed ground station)

    def height_constraint(pos_flat, i):
        pos = pos_flat.reshape((-1, 3))[i]
        terrain_z = terrain_elevation(pos[0], pos[1])
        return pos[2] - (terrain_z + H_min)

    def snr_constraint(pos_flat, i):
        pos = pos_flat.reshape((-1, 3))
        pt = tx_power(pos[i], pos[i+1])
        return P_T_max - (pt * 1.01)

    def los_constraint(pos_flat, i, n_samples=100):
      positions = pos_flat.reshape((-1, 3))
      pt1 = positions[i]
      pt2 = positions[i + 1]
      clearances = []

      for j in range(1, n_samples):
          t = j / n_samples
          x = pt1[0] * (1 - t) + pt2[0] * t
          y = pt1[1] * (1 - t) + pt2[1] * t
          z = pt1[2] * (1 - t) + pt2[2] * t

          if not (terrain_x[0] <= x <= terrain_x[-1]) or not (terrain_y[0] <= y <= terrain_y[-1]):
            print(f">>> LoS point (x={x}, y={y}) out of bounds")
            return -9999  # Return a large negative number instead of -inf

          terrain_z = terrain_elevation(x, y)
          if terrain_z is None or np.isnan(terrain_z):
            print(f">>> NaN terrain at (x={x}, y={y})")
            return -9999

          clearances.append(z - terrain_z)

      return np.mean(clearances) - 1  # Require average 2m clearance

    # Final LoS and SNR from last UAV to ground
    def final_los_constraint(pos_flat):
      n_samples=100
      pos = pos_flat.reshape((-1, 3))
      pt1 = pos[-1]
      pt2 = np.array([xG, yG, zG])
      clearances = []

      for j in range(1, n_samples):
          t = j / n_samples
          x = pt1[0] * (1 - t) + pt2[0] * t
          y = pt1[1] * (1 - t) + pt2[1] * t
          z = pt1[2] * (1 - t) + pt2[2] * t

          if not (terrain_x[0] <= x <= terrain_x[-1]) or not (terrain_y[0] <= y <= terrain_y[-1]):
              print(f" >>> Final LoS point (x={x}, y={y}) out of bounds")
              return -9999

          terrain_z = terrain_elevation(x, y)
          if terrain_z is None or np.isnan(terrain_z):
              print(f">>> Final LoS terrain NaN at (x={x}, y={y})")
              return -9999

          clearances.append(z - terrain_z)

          return np.mean(clearances) - H_min  # Require average clearance ≥ H_min

    def final_snr_constraint(pos_flat):
        pos = pos_flat.reshape((-1, 3))
        last_uav_pos = pos[-1]
        pt = tx_power(last_uav_pos, np.array([xG, yG, zG]))
        return P_T_max - (pt * 1.01)

    # Add constraints between consecutive UAVs (0 → 1 up to n_uav_opt - 2 → n_uav_opt - 1)
    for i in range(n_uav_opt - 1):
        cons.append({'type': 'ineq', 'fun': lambda pos, i=i: height_constraint(pos, i)})
        cons.append({'type': 'ineq', 'fun': lambda pos, i=i: snr_constraint(pos, i)})
        cons.append({'type': 'ineq', 'fun': lambda pos, i=i: los_constraint(pos, i)})

    # Constraint for last UAV
    cons.append({'type': 'ineq', 'fun': lambda pos: height_constraint(pos, n_uav_opt - 1)})
    cons.append({'type': 'ineq', 'fun': final_snr_constraint})
    cons.append({'type': 'ineq', 'fun': final_los_constraint})
    return cons



# -------------------------------
# Initialization
# -------------------------------
zG = terrain_elevation(xG, yG)
z1 = terrain_elevation(x1, y1)

assert terrain_x[0] <= xG <= terrain_x[-1], "xG out of bounds"
assert terrain_y[0] <= yG <= terrain_y[-1], "yG out of bounds"

n_total_uav = compute_mmin([xG, yG, zG], [x1, y1, z1]) + 1
print(n_total_uav)
n_uav_opt = n_total_uav - 1  # Since xG (ground station) is fixed
init_positions = initialize_positions(np.array([x1, y1, z1]), np.array([xG, yG, zG]), n_total_uav, z_max)
x0 = init_positions[1:].flatten()  # Exclude the fixed starting UAV


# -------------------------------
# Optimization
# -------------------------------
bounds = []
for _ in range(n_uav_opt):
    bounds.extend([
        x_bounds,     # x ∈ [min_x, max_x]
        y_bounds,     # y ∈ [min_y, max_y]
        (H_min, None) # z ≥ H_min
    ])

for c in make_constraints(n_total_uav):
    val = c['fun'](x0)
    print(f"Constraint value: {val:.2f}")

result = minimize(total_power, x0, method='slsqp',  #SLSQP
                  constraints=make_constraints(n_total_uav),
                  bounds=bounds,
                  options={'disp': True, 'maxiter': 10000})
# -------------------------------
# Results
# -------------------------------
uav_positions = result.x.reshape(-1, 3)
opt_positions = np.vstack(([x1, y1, z1], result.x.reshape((-1, 3))))
print("Optimal positions:")
print(opt_positions)