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
ax.set_title("Particle positions colored by offset Morton key colors", color="white")
ax.tick_params(colors="white", which='both')

plt.savefig("particle_plot.png", transparent=True)
plt.show()
