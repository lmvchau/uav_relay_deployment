import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from terrain_utils import crop_terrain


# # --- Load terrain data ---
# data = np.load("./terrain_data_mendocino_30.npz")
# z_meters   = np.flipud(data["z"])
# x_meters   = data["x"]
# y_meters   = np.flipud(data["y"])
# terrain_x  = x_meters[0, :]
# terrain_y  = y_meters[:, 0]
# terrain_z  = z_meters



def animate_trajectory(x_vals, y_vals, z_vals, terrain_x, terrain_y, terrain_z, output_file="./results/trajectory.mp4",
                       x_min=16000, x_max=24000, y_min=78000, y_max=84000,
                       fps=10, interval=100):

    """
    Animate a 3D trajectory over terrain data.
    Parameters:
    - x_vals: 1D array of x coordinates of the trajectory
    - y_vals: 1D array of y coordinates of the trajectory
    - z_vals: 1D array of z coordinates of the trajectory
    - terrain_x: 1D array of x coordinates of the terrain
    - terrain_y: 1D array of y coordinates of the terrain
    - terrain_z: 2D array of z values of the terrain
    - output_file: name of the output video file to write .mp4
    - x_min, x_max, y_min, y_max: cropping bounds for the terrain
    - fps: frames per second for the animation
    - interval: milliseconds between frames
    """

    # Crop the terrain

    # x_mask = (terrain_x >= x_min) & (terrain_x <= x_max)
    # y_mask = (terrain_y >= y_min) & (terrain_y <= y_max)

    # terrain_x_cropped = terrain_x[x_mask]
    # terrain_y_cropped = terrain_y[y_mask]
    # terrain_z_cropped = z_meters[y_mask, :][:, x_mask]

    # terrain_x = terrain_x_cropped
    # terrain_y = terrain_y_cropped
    # terrain_z = terrain_z_cropped

    if x_min is None: x_min = np.min(terrain_x) 
    if x_max is None: x_max = np.max(terrain_x) 
    if y_min is None: y_min = np.min(terrain_y)
    if y_max is None: y_max = np.max(terrain_y)

    terrain_x, terrain_y, terrain_z = crop_terrain(terrain_x=terrain_x, terrain_y=terrain_y, terrain_z=terrain_z,
                                                        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


    max_z = np.max(terrain_z)
    min_z = np.min(terrain_z)
    X, Y       = np.meshgrid(terrain_x, terrain_y)

    # Setup the figure
    fig = plt.figure(figsize=(10, 10))
    ax  = fig.add_subplot(111, projection='3d')

    # static terrain
    ax.plot_surface(
        X, Y, terrain_z,
        color='lightgreen', alpha=0.5, edgecolor='none', zorder=1
    )

    ax.plot_wireframe(X, Y, terrain_z, color='gray', alpha=0.5, linewidth=0.5, zorder=1)


    x_range = terrain_x.max() - terrain_x.min()
    y_range = terrain_y.max() - terrain_y.min()
    z_range = terrain_z.max()  - terrain_z.min()

    ax.set_box_aspect((x_range, y_range, z_range))

    # prepare the moving line and point
    line, = ax.plot([], [], [], lw=3, color='blue')
    point, = ax.plot([], [], [], 'o', color='red', markersize=6)

    # fix axes
    ax.set_xlim(terrain_x.min(), terrain_x.max())
    ax.set_ylim(terrain_y.min(), terrain_y.max())
    ax.set_zlim(terrain_z.min(), terrain_z.max() + 200)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Elevation (m)')

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point

    def update(i):
        # draw path up to step i
        line.set_data(x_vals[:i], y_vals[:i])
        line.set_3d_properties(z_vals[:i])
        point.set_data(x_vals[i], y_vals[i])
        point.set_3d_properties(z_vals[i])
        return line, point

    # build the animation
    anim = FuncAnimation(
        fig, update,
        frames=len(x_vals),
        init_func=init,
        blit=True,
        interval=100   # milliseconds per frame => 10 fps
    )

    # save to MP4 (requires ffmpeg on your PATH)
    writer = FFMpegWriter(fps=10, bitrate=1800)
    anim.save(output_file, writer=writer)
    plt.close(fig)
    print(f"Saved animation to {output_file}")