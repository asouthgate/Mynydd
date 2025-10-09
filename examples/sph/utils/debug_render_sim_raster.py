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
from vispy import app, scene
from scipy.ndimage import gaussian_filter

# -----------------------------
# Parse args
# -----------------------------
if len(sys.argv) < 3:
    print("Usage: python vis_rasterize_from_particles_types.py <baseprefix> <dt> [resolution]")
    sys.exit(1)

prefix = sys.argv[1]
dt = float(sys.argv[2])
resolution = int(sys.argv[3]) if len(sys.argv) > 3 else 64

# -----------------------------
# Gather HDF5 files
# -----------------------------
files = sorted(glob.glob(f"{prefix}.*.h5"), key=os.path.getmtime)
if not files:
    print("No files found with prefix:", prefix)
    sys.exit(1)

print(f"Found {len(files)} files, resolution={resolution}")

# -----------------------------
# Compute global bounding box (so all frames rasterize into same voxel coordinates)
# -----------------------------
print("Scanning files to compute global bounding box (this ensures consistent rasterization)...")
mins = np.array([np.inf, np.inf, np.inf], dtype=float)
maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)

for fname in files:
    with h5py.File(fname, "r") as f:
        pos = np.asarray(f["positions"])
        if pos.size == 0:
            continue
        mins = np.minimum(mins, pos.min(axis=0))
        maxs = np.maximum(maxs, pos.max(axis=0))

# Avoid degenerate range
eps = 1e-9
ranges = maxs - mins
zero_mask = ranges <= eps
if np.any(zero_mask):
    ranges[zero_mask] = 1.0

print("Global bounds:")
print("  min:", mins)
print("  max:", maxs)

# -----------------------------
# Rasterization helper
# -----------------------------
def rasterize_positions(positions, densities, types, res, xmin, xmax):
    """
    Rasterize weighted particles into a 3D grid per type.
    Returns dict of type_id -> (nx, ny, nz) array.
    """
    if positions.size == 0:
        return {}

    pos_norm = (positions - xmin) / (xmax - xmin)
    pos_norm = np.clip(pos_norm, 0.0, 1.0)

    bins = [res, res, res]
    ranges = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

    unique_types = np.unique(types)
    type_grids = {}

    for t in unique_types:
        mask = types == t
        if not np.any(mask):
            continue

        weighted_hist, _ = np.histogramdd(pos_norm[mask], bins=bins, range=ranges, weights=densities[mask])
        counts, _ = np.histogramdd(pos_norm[mask], bins=bins, range=ranges)

        with np.errstate(divide="ignore", invalid="ignore"):
            avg_density = np.where(counts > 0, weighted_hist / counts, 0.0)

        avg_density = gaussian_filter(avg_density, sigma=0.7)
        type_grids[int(t)] = avg_density

    return type_grids

# -----------------------------
# Helper to map from (x,y,z) -> (z,y,x)
# -----------------------------
def to_vispy_array(arr_xyz):
    return np.transpose(arr_xyz, (2, 1, 0))

# -----------------------------
# Initialize from first frame
# -----------------------------
with h5py.File(files[0], "r") as f:
    positions = np.array(f["positions"])
    densities = np.array(f["densities"])
    types = np.array(f["particle_type"], dtype=np.uint32)

    grids_by_type = rasterize_positions(positions, densities, types, resolution, mins, maxs)

# Normalize and prepare VisPy volumes per type
volumes = {}
for t, grid in grids_by_type.items():
    grid_norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-12)
    volumes[t] = to_vispy_array(grid_norm)

# -----------------------------
# VisPy setup
# -----------------------------
canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', size=(1000, 800), show=True)
view = canvas.central_widget.add_view()

# Choose colormaps by type
type_colormaps = {
    0: 'viridis',
    1: 'magma',
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
view.camera = scene.cameras.TurntableCamera(fov=60, azimuth=45, elevation=30, distance=2.5 * resolution)
axis = scene.visuals.XYZAxis(parent=view.scene)

# -----------------------------
# Animation loop
# -----------------------------
iteration_idx = 0

def update(event):
    global iteration_idx, interiors

    fname = files[iteration_idx]
    with h5py.File(fname, "r") as f:
        positions = np.array(f["positions"])
        densities = np.array(f["densities"])
        types = np.array(f["particle_type"], dtype=np.uint32)
        grids_by_type = rasterize_positions(positions, densities, types, resolution, mins, maxs)

    for t, vis in interiors.items():
        if t in grids_by_type:
            grid = grids_by_type[t]
            grid_norm = (grid - grid.min()) / (grid.max() - grid.min() + 1e-12)
            vis.set_data(to_vispy_array(grid_norm))
        else:
            # No particles of this type -> blank volume
            vis.set_data(np.zeros_like(list(interiors.values())[0].get_data()))

    canvas.update()
    iteration_idx = (iteration_idx + 1) % len(files)

# Timer
timer = app.Timer(interval=dt, connect=update, start=True)

if __name__ == "__main__":
    app.run()
