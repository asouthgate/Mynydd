import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import sys
import random

data = pd.read_csv(sys.argv[1])  

x = data["x"]
y = data["y"]
z = data["z"]
morton = data["morton_key"]
density = data["density"]

unique_keys = morton.unique()

n_keys = len(unique_keys)
n_colors = max(unique_keys) + 1
runlen = 20

colmap = cm.get_cmap('rainbow', n_colors)
palette = colmap(np.arange(n_colors))[:, :3]  # take only RGB

rng = np.random.default_rng(seed=12345) 
rloff = int(n_colors / runlen)
colors_map = {key: palette[rloff * (key % runlen)] * 0.3 + palette[key] * 0.7 for ki, key in enumerate(unique_keys)}

colors = np.array([colors_map[k] for k in morton])

fig = plt.figure(figsize=(10, 8), facecolor="none")  # transparent figure
ax = fig.add_subplot(111, projection='3d', facecolor="none")  # transparent axes

sc = ax.scatter(x, y, z, c=colors, marker='o', s=5, alpha=0.8)

ax.set_xlabel("X", color="white")
ax.set_ylabel("Y", color="white")
ax.set_zlabel("Z", color="white")
ax.set_title("Particle positions colored by Morton keys", color="white")
ax.tick_params(colors="white", which='both')

plt.savefig("particle_plot.png", transparent=True)
plt.show()

# Scatter plot colored by density with magma colormap
fig = plt.figure(figsize=(10, 8), facecolor="none")
ax = fig.add_subplot(111, projection="3d", facecolor="none")

# normalize density values for colormap
norm = plt.Normalize(vmin=density.min(), vmax=density.max())
cmap = cm.get_cmap("magma")

sc = ax.scatter(x, y, z, c=density, cmap=cmap, norm=norm,
                marker='o', s=5, alpha=0.8)

ax.set_xlabel("X", color="white")
ax.set_ylabel("Y", color="white")
ax.set_zlabel("Z", color="white")
ax.set_title("Particle densities", color="white")
ax.tick_params(colors="white", which="both")

# add colorbar to show density mapping
cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("Density", color="white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="white")

plt.savefig("particle_density_magma.png", transparent=True)
plt.show()

