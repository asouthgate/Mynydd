import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import sys

data = pd.read_csv(sys.argv[1])  

x = data["x"]
y = data["y"]
z = data["z"]
morton = data["morton_key"]
density = data["density"]
pressure = data["pressure"]
fpx = data["fpx"]
fpy = data["fpy"]
fpz = data["fpz"]

# Compute force magnitude
force_mag = np.sqrt(fpx**2 + fpy**2 + fpz**2)
# Normalize to 0-1
force_norm = (force_mag - force_mag.min()) / (force_mag.max() - force_mag.min())

fig = plt.figure(figsize=(20, 16), facecolor="none")  # bigger figure for 4 subplots

# -----------------------------
# Top-left: Morton key coloring
# -----------------------------
unique_keys = morton.unique()
n_keys = len(unique_keys)
n_colors = max(unique_keys) + 1
runlen = 20
colmap = cm.get_cmap('rainbow', n_colors)
palette = colmap(np.arange(n_colors))[:, :3]  # RGB only
rloff = int(n_colors / runlen)
colors_map = {key: palette[rloff * (key % runlen)] * 0.3 + palette[key] * 0.7 for key in unique_keys}
colors = np.array([colors_map[k] for k in morton])

ax1 = fig.add_subplot(2, 2, 1, projection='3d', facecolor="none")
ax1.scatter(x, y, z, c=colors, marker='o', s=5, alpha=0.8)
ax1.set_xlabel("X", color="white")
ax1.set_ylabel("Y", color="white")
ax1.set_zlabel("Z", color="white")
ax1.set_title("Particles colored by Morton keys", color="white")
ax1.tick_params(colors="white", which='both')

# -----------------------------
# Top-right: Density plot (magma)
# -----------------------------
ax2 = fig.add_subplot(2, 2, 2, projection='3d', facecolor="none")
norm_density = plt.Normalize(vmin=density.min(), vmax=density.max())
sc2 = ax2.scatter(x, y, z, c=density, cmap=cm.magma, norm=norm_density,
                  marker='o', s=5, alpha=0.8)
ax2.set_xlabel("X", color="white")
ax2.set_ylabel("Y", color="white")
ax2.set_zlabel("Z", color="white")
ax2.set_title("Particle densities", color="white")
ax2.tick_params(colors="white", which="both")
cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.6, pad=0.1)
cbar2.set_label("Density", color="white")
cbar2.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color="white")

# -----------------------------
# Bottom-left: Pressure plot (viridis)
# -----------------------------
ax3 = fig.add_subplot(2, 2, 3, projection='3d', facecolor="none")
norm_pressure = plt.Normalize(vmin=pressure.min(), vmax=pressure.max())
sc3 = ax3.scatter(x, y, z, c=pressure, cmap=cm.viridis, norm=norm_pressure,
                  marker='o', s=5, alpha=0.8)
ax3.set_xlabel("X", color="white")
ax3.set_ylabel("Y", color="white")
ax3.set_zlabel("Z", color="white")
ax3.set_title("Particle pressures", color="white")
ax3.tick_params(colors="white", which="both")
cbar3 = plt.colorbar(sc3, ax=ax3, shrink=0.6, pad=0.1)
cbar3.set_label("Pressure", color="white")
cbar3.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color="white")

# -----------------------------
# Bottom-right: Force magnitude (normalized)
# -----------------------------
ax4 = fig.add_subplot(2, 2, 4, projection='3d', facecolor="none")
norm_force = plt.Normalize(vmin=0, vmax=1)
sc4 = ax4.scatter(x, y, z, c=force_norm, cmap=cm.plasma, norm=norm_force,
                  marker='o', s=5, alpha=0.4)
ax4.set_xlabel("X", color="white")
ax4.set_ylabel("Y", color="white")
ax4.set_zlabel("Z", color="white")
ax4.set_title("Force magnitude (normalized)", color="white")
ax4.tick_params(colors="white", which="both")
cbar4 = plt.colorbar(sc4, ax=ax4, shrink=0.6, pad=0.1)
cbar4.set_label("Normalized Force", color="white")
cbar4.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cbar4.ax.axes, 'yticklabels'), color="white")

# -----------------------------
# Save and show
# -----------------------------
plt.tight_layout()
plt.savefig("particle_2x2_force_mag.png", transparent=True)
plt.show()
