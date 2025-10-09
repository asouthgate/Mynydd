#!/usr/bin/env python3
"""
vis_rasterize_from_particles.py

Usage:
    python vis_rasterize_from_particles.py <baseprefix> <dt> [resolution]

- baseprefix: e.g. "output/particles/particles" — files named like:
    baseprefix.<iter>.h5  (must contain datasets "positions" (N,3), "morton_keys", "densities")
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
    print("Usage: python vis_rasterize_from_particles.py <baseprefix> <dt> [resolution]")
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

# If degenerate range occurs, set a small padding
eps = 1e-9
ranges = maxs - mins
zero_mask = ranges <= eps
if np.any(zero_mask):
    # pad tiny ranges to avoid division by zero
    ranges[zero_mask] = 1.0

print("Global bounds:")
print("  min:", mins)
print("  max:", maxs)

# -----------------------------
# Rasterization helper
# -----------------------------
def rasterize_positions(positions, densities, res, xmin, xmax):
    """
    Rasterize weighted particles into a 3D grid.
    - positions: (N,3) array, columns = x,y,z (Z is third column and considered 'down')
    - densities: (N,) weights
    - res: int or (nx,ny,nz)
    Returns array shaped (nx, ny, nz) in coordinate order (x, y, z).
    """
    if positions.size == 0:
        return np.zeros((res, res, res), dtype=float)

    # Normalize positions into [0,1] using global bounding box
    pos_norm = (positions - xmin) / (xmax - xmin)
    pos_norm = np.clip(pos_norm, 0.0, 1.0)

    bins = [res, res, res]
    ranges = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

    # Weighted sum per voxel (by density)
    weighted_hist, edges = np.histogramdd(pos_norm, bins=bins, range=ranges, weights=densities)
    counts, _ = np.histogramdd(pos_norm, bins=bins, range=ranges)

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_density = np.where(counts > 0, weighted_hist / counts, 0.0)

    # Smooth a bit for nicer visuals
    avg_density = gaussian_filter(avg_density, sigma=0.7)

    # avg_density has shape (nx, ny, nz) — axes correspond to x,y,z (axis 2 is z as you requested)
    return avg_density


# -----------------------------
# Read first file to build initial raster + vispy setup
# -----------------------------
with h5py.File(files[0], "r") as f:
    positions = np.array(f["positions"])
    densities = np.array(f["densities"])
    raster_xyz = rasterize_positions(positions, densities, resolution, mins, maxs)  # (x,y,z)
    # normalize raster for display
    raster_norm = (raster_xyz - raster_xyz.min()) / (raster_xyz.max() - raster_xyz.min() + 1e-12)

# VisPy expects Volume data in shape (depth=z, height=y, width=x) -> (z, y, x)
def to_vispy_array(arr_xyz):
    # input arr_xyz shape (x, y, z)
    # return arr_vispy shape (z, y, x)
    return np.transpose(arr_xyz, (2, 1, 0))

raster_vispy = to_vispy_array(raster_norm)

# -----------------------------
# VisPy setup
# -----------------------------
canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', size=(1000, 800), show=True)
view = canvas.central_widget.add_view()

# Interior (unshaded, semi-transparent)
interior = scene.visuals.Volume(
    raster_vispy,
    parent=view.scene,
    threshold=0.05,
    cmap='viridis',
    method='iso',
)
interior.unfreeze()
interior.shade = True
interior.opacity = 1.0
interior.interpolation = 'nearest'
interior.freeze()

# Camera & axis
view.camera = scene.cameras.TurntableCamera(fov=60, azimuth=45, elevation=30, distance=2.5 * resolution)
axis = scene.visuals.XYZAxis(parent=view.scene)

# -----------------------------
# Animation loop
# -----------------------------
iteration_idx = 0

def update(event):
    global iteration_idx, interior, top_vispy

    fname = files[iteration_idx]
    with h5py.File(fname, "r") as f:
        positions = np.array(f["positions"])
        densities = np.array(f["densities"])
        raster_xyz = rasterize_positions(positions, densities, resolution, mins, maxs)
        raster_norm = (raster_xyz - raster_xyz.min()) / (raster_xyz.max() - raster_xyz.min() + 1e-12)

    interior.set_data(to_vispy_array(raster_norm))

    canvas.update()

    # advance
    iteration_idx = (iteration_idx + 1) % len(files)

# Timer
timer = app.Timer(interval=dt, connect=update, start=True)

if __name__ == "__main__":
    app.run()
