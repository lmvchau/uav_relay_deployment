import numpy as np 

def load_terrain(npz_path, flipud=True):
    """Returns terrain_x (1D), terrain_y (1D), terrain_z (2D)."""
    data = np.load(npz_path)
    z = data["z"]
    if flipud:
        z = np.flipud(z)
    x = data["x"]
    y = data["y"]
    terrain_x = x[0, :]
    terrain_y = (np.flipud(y)[:,0] if flipud else y[:,0])
    return terrain_x, terrain_y, z


def crop_terrain(terrain_x, terrain_y, terrain_z, x_min, x_max, y_min, y_max):
    """Crops to the requested bounds."""
    x_mask = (terrain_x >= x_min) & (terrain_x <= x_max)
    y_mask = (terrain_y >= y_min) & (terrain_y <= y_max)
    tx = terrain_x[x_mask]
    ty = terrain_y[y_mask]
    tz = terrain_z[y_mask, :][:, x_mask]
    return tx, ty, tz

def terrain_elevation(x, y, terrain_x, terrain_y, terrain_z):
    """Look up terrain 
    Params:
    - x, y - 1D np array
    - z - 2D np array
    """
    xi = (np.abs(terrain_x - x)).argmin()
    yi = (np.abs(terrain_y - y)).argmin()
    return terrain_z[yi, xi]
