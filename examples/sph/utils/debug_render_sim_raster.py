import os
import glob
import sys
import h5py
import numpy as np
from vispy import app, scene

# -----------------------------
# User parameters
# -----------------------------
prefix = sys.argv[1]     # e.g., "build/main_example_output_density/main_example_output_density"
dt = float(sys.argv[2])  # seconds per frame

# -----------------------------
# Gather all HDF5 files
# -----------------------------
files = sorted(glob.glob(f"{prefix}.*.h5"), key=os.path.getmtime)
if not files:
    print("No files found with prefix:", prefix)
    sys.exit(1)

# -----------------------------
# Read first file to initialize volume
# -----------------------------
first_file = files[0]
with h5py.File(first_file, "r") as f:
    raster = np.array(f["raster"])
    try:
        resolution = int(f["raster"].attrs["resolution"])
        if resolution**3 != raster.size:
            resolution = int(round(raster.size ** (1/3)))
    except KeyError:
        resolution = int(round(raster.size ** (1/3)))

    raster_3d = raster.reshape((resolution, resolution, resolution))
    raster_norm = (raster_3d - np.min(raster_3d)) / (np.max(raster_3d) - np.min(raster_3d) + 1e-12)

# -----------------------------
# VisPy canvas setup
# -----------------------------
canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', size=(1000, 800), show=True)
view = canvas.central_widget.add_view()

# Create the volume visual with proper initial shape
volume = scene.visuals.Volume(
    raster_norm,
    parent=view.scene,
    threshold=0.05,      # adjust opacity cutoff
    cmap='viridis',
    method='iso'  # 'mip' is default max intensity projection
)
volume.unfreeze()
#opacity_map = np.clip(raster_norm * 2.0, 0, 1)  # scale to emphasize mid-densities
volume.shade = True
volume.threshold = 0.2  # only values above this are considered surfaces
#volume._opacity = opacity_map
volume.freeze()
# Camera & axes
view.camera = scene.cameras.TurntableCamera(fov=60, azimuth=45, elevation=30, distance=2.5 * resolution)
axis = scene.visuals.XYZAxis(parent=view.scene)

# -----------------------------
# Animation state
# -----------------------------
iteration_idx = 0

def update(event):
    global iteration_idx, volume, files

    filename = files[iteration_idx]
    with h5py.File(filename, "r") as f:
        raster = np.array(f["raster"])
        raster_3d = raster.reshape((resolution, resolution, resolution))
        raster_norm = (raster_3d - np.min(raster_3d)) / (np.max(raster_3d) - np.min(raster_3d) + 1e-12)

    # Update the volume in-place (do not recreate)
    volume.set_data(raster_norm)
    canvas.update()
    # Advance to next frame
    iteration_idx += 1
    if iteration_idx >= len(files):
        iteration_idx = 0  # loop back to start

# -----------------------------
# Start animation timer
# -----------------------------
timer = app.Timer(interval=dt, connect=update, start=True)

if __name__ == "__main__":
    app.run()
