import os
import glob
import sys
import time

import h5py
import numpy as np
from vispy import app, scene

prefix = sys.argv[1]
dataset_name = "positions"
dt = 0.01

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
        if dataset_name not in f:
            print(f"{dataset_name} not found in {filename}")
            return

        data = f[dataset_name][...]   # shape (n_particles, n_dims)
        positions = np.array(data, dtype=np.float32)

    scatter.set_data(positions, face_color="orange", size=5, edge_color=None)

    if iteration_idx < len(files) - 1:
        iteration_idx += 1


timer = app.Timer(interval=dt, connect=update, start=True)

if __name__ == "__main__":
    app.run()

