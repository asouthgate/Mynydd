#!/usr/bin/env python3
"""
vis_rasterize_from_particles_types.py

Usage:
    python vis_rasterize_from_particles_types.py <baseprefix> <dt> [resolution]

- baseprefix: e.g. "output/particles/particles" — files named like:
    baseprefix.<iter>.h5  (must contain datasets "positions", "morton_keys", "densities", "particle_type")
- dt: seconds per frame (float)
- resolution: optional integer grid resolution (default 64)
"""

import os
import glob
import sys
import h5py
import numpy as np
import pickle
from vispy import app, scene
from scipy.ndimage import gaussian_filter

# -----------------------------
# Parse args
# -----------------------------
if len(sys.argv) < 3:
    print("Usage: python vis_rasterize_from_particles_types.py <baseprefix> <dt>")
    sys.exit(1)

prefix = sys.argv[1]
dt = float(sys.argv[2])
resolution = int(sys.argv[3]) if len(sys.argv) > 3 else 64  # used for zoom

# read in rasters (pickle); glob raster files first then read each
raster_files = sorted(glob.glob(f"{prefix}/*.pkl"), key=lambda x: int(x.split(".")[-3]))
with open(raster_files[0], "rb") as f:
    grids_by_type = pickle.load(f)


# -----------------------------
# Helper to map from (x,y,z) -> (z,y,x)
# -----------------------------
def to_vispy_array(arr_xyz):
    return np.transpose(arr_xyz, (2, 1, 0))



# Normalize and prepare VisPy volumes per type
volumes = {}
for t, grid in grids_by_type.items():
    grid_norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-12)
    volumes[t] = to_vispy_array(grid_norm)

# -----------------------------
# VisPy setup
# -----------------------------
# canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', size=(1000, 800), show=True)
canvas = scene.SceneCanvas(keys='interactive', bgcolor='#1f212b', size=(1000, 800), show=True)
# view = canvas.central_widget.add_view()
view = canvas.central_widget.add_view()

# Choose colormaps by type
type_colormaps = {
    0: 'nipy_spectral',
    1: 'plasma_r',
}

# Create one Volume visual per type
interiors = {}
for t, data in volumes.items():
    cmap = type_colormaps.get(t, 'gray')
    vis = scene.visuals.Volume(
        data,
        parent=view.scene,
        threshold=0.05,
        cmap=cmap,
        method='iso',
    )
    vis.unfreeze()
    vis.shade = True
    vis.opacity = 1.0
    vis.interpolation = 'linear'
    vis.freeze()
    interiors[t] = vis

# Camera and axes
view.camera = scene.cameras.TurntableCamera(fov=60, azimuth=65, elevation=2, distance=1.65 * resolution)
axis = scene.visuals.XYZAxis(parent=view.scene)
# -----------------------------
# Animation loop
# -----------------------------
iteration_idx = 0

def update(event):
    global iteration_idx, interiors

    fname = raster_files[iteration_idx]
    with open(fname, "rb") as f:
        grids_by_type = pickle.load(f)

    for t, vis in interiors.items():
        if t in grids_by_type:
            grid = grids_by_type[t]
            grid_norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-12)
            vis.set_data(to_vispy_array(grid_norm))
        else:
            # No particles of this type -> blank volume
            vis.set_data(np.zeros_like(list(interiors.values())[0].get_data()))

    canvas.update()
    iteration_idx = (iteration_idx + 1) % len(raster_files)

# Timer
timer = app.Timer(interval=dt, connect=update, start=True)

if __name__ == "__main__":
    app.run()
