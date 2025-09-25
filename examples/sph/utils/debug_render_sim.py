import os
import glob
import sys
import time

import h5py
import matplotlib.cm as cm
import numpy as np
from vispy import app, scene

def morton_key_to_color(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.uint32)

    # Use multiplicative hashing with large odd constants to scramble bits
    r = ((keys * 0x85ebca6b) >> 16) & 0xFF
    g = ((keys * 0xc2b2ae35) >> 8)  & 0xFF
    b = ((keys * 0x27d4eb2d))       & 0xFF

    # Normalize to [0,1]
    colors = np.stack([r, g, b, np.full_like(r, 255)], axis=-1).astype(np.float32) / 255.0
    return colors

prefix = sys.argv[1]
dt = float(sys.argv[2])

canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', show=True)
view = canvas.central_widget.add_view()

scatter = scene.visuals.Markers()
view.add(scatter)

# Set initial positions to avoid bounds error
initial_positions = np.zeros((1, 3), dtype=np.float32)  # at least 1 point
scatter.set_data(initial_positions, face_color='orange', size=5)

axis = scene.visuals.XYZAxis(parent=view.scene)
axis.transform = scene.transforms.STTransform(scale=(1, 1, 1))

view.camera = scene.cameras.TurntableCamera(
    fov=45,
    azimuth=30,
    elevation=30,
    distance=20
)

iteration_idx = 0
positions = None
n_particles = None
n_dims = None
n_iterations = None

def update(event):
    global iteration_idx, positions, scatter

    # Find all iteration files (assumes filenames like base.0.h5, base.1.h5, …)
    
    files = sorted(glob.glob(f"{prefix}.*.h5"), key=os.path.getmtime)
    if not files:
        print("No new iteration files yet... waiting", end='\r', flush=True)
        return

    iteration_idx = min(iteration_idx, len(files) - 1)
    filename = files[iteration_idx]

    print(f"Reading {filename}, iteration {iteration_idx}", end='\r', flush=True)
    with h5py.File(filename, "r") as f:

        data = f["positions"][...]   # shape (n_particles, n_dims)
        nbits = 4
        max_key = (1 << (3 * nbits)) - 1
        min_key = 0
        keys = np.array(f["morton_keys"][...], dtype=int)
        positions = np.array(data, dtype=np.float32)
        keys_norm = (keys - min_key) / (max_key - min_key + 1e-9)
        cmap = cm.get_cmap("viridis")
        colors = 0.3 * cmap(keys_norm) + 0.7 * morton_key_to_color(keys)


    scatter.set_data(positions, face_color=colors, size=5, edge_color=None)

    if iteration_idx < len(files) - 1:
        iteration_idx += 1


timer = app.Timer(interval=dt, connect=update, start=True)

if __name__ == "__main__":
    app.run()

