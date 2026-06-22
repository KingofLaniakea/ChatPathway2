# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LightSource

# # =====================================================
# # Grid
# # =====================================================

# x = np.linspace(-6, 6, 300)
# y = np.linspace(-4, 4, 300)

# X, Y = np.meshgrid(x, y)

# # =====================================================
# # Multi-basin energy landscape
# # =====================================================

# Z = (
#     2.5 * np.exp(-((X+3)**2 + (Y+1.5)**2)/2.5)
#     + 1.8 * np.exp(-((X-2)**2 + (Y-1)**2)/1.8)
#     + 2.2 * np.exp(-((X-4)**2 + (Y+2)**2)/2.2)
#     - 1.5 * np.exp(-((X)**2 + (Y)**2)/6)
# )

# Z += 0.2*np.sin(X*1.2) * np.cos(Y*1.5)

# # invert to create valleys
# Z = -Z

# # =====================================================
# # Plot
# # =====================================================

# fig = plt.figure(figsize=(14, 5))
# ax = fig.add_subplot(111, projection='3d')

# ls = LightSource(azdeg=315, altdeg=45)

# rgb = ls.shade(
#     Z,
#     cmap=plt.cm.Purples,
#     vert_exag=0.5,
#     blend_mode='soft'
# )

# # surface
# ax.plot_surface(
#     X,
#     Y,
#     Z,
#     rstride=2,
#     cstride=2,
#     facecolors=rgb,
#     linewidth=0,
#     antialiased=True,
#     shade=False,
#     alpha=0.85
# )

# # wireframe
# ax.plot_wireframe(
#     X,
#     Y,
#     Z,
#     rstride=15,
#     cstride=15,
#     color='mediumpurple',
#     linewidth=0.3,
#     alpha=0.35
# )

# # =====================================================
# # Latent trajectory
# # =====================================================

# t = np.linspace(-5, 5, 120)

# traj_x = t
# traj_y = 1.2*np.sin(t/2)

# traj_z = (
#     -(
#         2.5 * np.exp(-((traj_x+3)**2 + (traj_y+1.5)**2)/2.5)
#         + 1.8 * np.exp(-((traj_x-2)**2 + (traj_y-1)**2)/1.8)
#         + 2.2 * np.exp(-((traj_x-4)**2 + (traj_y+2)**2)/2.2)
#         - 1.5 * np.exp(-((traj_x)**2 + (traj_y)**2)/6)
#     )
# )

# ax.plot(
#     traj_x,
#     traj_y,
#     traj_z,
#     color='purple',
#     linewidth=4
# )

# ax.scatter(
#     traj_x[::15],
#     traj_y[::15],
#     traj_z[::15],
#     color='darkviolet',
#     s=50
# )

# # =====================================================
# # Style
# # =====================================================

# ax.set_axis_off()

# ax.view_init(
#     elev=28,
#     azim=-65
# )

# plt.tight_layout()

# # 保存为 PNG
# plt.savefig(
#     "latent_manifold.png",
#     dpi=300,
#     bbox_inches='tight',
#     transparent=True
# )

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# =====================================================
# Grid
# =====================================================

x = np.linspace(-8, 8, 400)
y = np.linspace(-4, 4, 300)

X, Y = np.meshgrid(x, y)

# =====================================================
# Energy landscape
# =====================================================

# Initial high-energy peak
peak = 4.5 * np.exp(
    -((X + 3.5)**2 / 4 + (Y)**2 / 3)
)

# Final attractor basin
valley = -6.0 * np.exp(
    -((X - 3.5)**2 / 5 + (Y + 0.3)**2 / 4)
)

# Smooth transition structure
transition = 1.2 * np.sin(0.7 * X) * np.exp(-Y**2 / 10)

# Global energy slope (critical!)
slope = -0.12 * X

# Final landscape
Z = peak + valley + transition + slope

# =====================================================
# Plot
# =====================================================

fig = plt.figure(figsize=(14, 5))
ax = fig.add_subplot(111, projection='3d')

# Soft lighting
ls = LightSource(azdeg=315, altdeg=45)

rgb = ls.shade(
    Z,
    cmap=plt.cm.Purples,
    vert_exag=0.8,
    blend_mode='soft'
)

# Surface
ax.plot_surface(
    X,
    Y,
    Z,
    rstride=2,
    cstride=2,
    facecolors=rgb,
    linewidth=0,
    antialiased=True,
    shade=False,
    alpha=0.88
)

# Wireframe
ax.plot_wireframe(
    X,
    Y,
    Z,
    rstride=18,
    cstride=18,
    color='purple',
    linewidth=0.4,
    alpha=0.18
)

# =====================================================
# Trajectory
# =====================================================

t = np.linspace(-7, 7, 200)

traj_x = t
traj_y = 0.6*np.sin(t/1.5)

# Interpolate trajectory height
traj_z = (
    4.5 * np.exp(-((traj_x + 3.5)**2 / 4 + (traj_y)**2 / 3))
    - 6.0 * np.exp(-((traj_x - 3.5)**2 / 5 + (traj_y + 0.3)**2 / 4))
    + 1.2 * np.sin(0.7 * traj_x) * np.exp(-traj_y**2 / 10)
    - 0.12 * traj_x
)

# Flow line
ax.plot(
    traj_x,
    traj_y,
    traj_z + 0.15,
    color='#8A2BE2',
    linewidth=5
)

# Latent states
ax.scatter(
    traj_x[::20],
    traj_y[::20],
    traj_z[::20] + 0.15,
    color='darkviolet',
    s=60
)

# =====================================================
# Camera & style
# =====================================================

ax.set_axis_off()

ax.view_init(
    elev=23,
    azim=-72
)

plt.tight_layout()

# Save
plt.savefig(
    "hnn_energy_landscape.png",
    dpi=300,
    transparent=True,
    bbox_inches='tight'
)

plt.show()