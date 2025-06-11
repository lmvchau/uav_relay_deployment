import numpy as np

DEFAULT_PARAMS = {
    'rho': 1.225,
    'A': 0.79,
    'cf': 0.13,
    's': 0.1,
    'S_FP': 0.01,
    'c_T': 0.3,
    'delta': 0.012,
    'g': 9.81,
    'm': 5.0,
    'v_ref': 10,
    'z_ref': 100,
    'alpha': 0.28,
    'beta': np.radians(30),
}

def norm(v):
    return np.linalg.norm(v)

def induced_power(T_vec, rho, A, cf):
    return ((1 + cf) * norm(T_vec)**2) / (2 * rho * A)

def blade_power(T_vec, rho, s, A, c_T, delta):
    T_norm = norm(T_vec)
    base = rho * s**2 * A * T_norm / c_T
    return (delta / 8) * (T_norm / (c_T * rho * A)) * np.sqrt(max(base, 0))

def wind_velocity(z, v_ref, z_ref, alpha):
    z_ratio = z / z_ref
    z_ratio = max(z_ratio, 1e-6)
    return v_ref * z_ratio**alpha

def compute_hover_power(position, **kwargs):
    # Use default constants and override with kwargs if provided
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    x, y, z = position
    v_wind = wind_velocity(z, params['v_ref'], params['z_ref'], params['alpha'])
    v_w = np.array([
        v_wind * np.cos(params['beta']),
        v_wind * np.sin(params['beta']),
        0
    ])
    v_w_norm = norm(v_w)

    Fx = 0.5 * params['rho'] * params['S_FP'] * v_w_norm**2 * np.cos(params['beta'])
    Fy = -0.5 * params['rho'] * params['S_FP'] * v_w_norm**2 * np.sin(params['beta'])
    Fz = params['m'] * params['g']

    T_vec = np.array([Fx, Fy, Fz])

    return (
        blade_power(T_vec, params['rho'], params['s'], params['A'], params['c_T'], params['delta']) +
        induced_power(T_vec, params['rho'], params['A'], params['cf']) +
        0.5 * params['rho'] * params['S_FP'] * v_w_norm**3
    )