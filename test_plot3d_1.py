# -*- coding: utf_8 *-*
"""
test_plot3d_1.py - plot 3d-graph

"""
import numpy as np
import matplotlib.pyplot as plt

# PARAMETER(S)

WIDTH    = 8
HEIGHT   = 5

XY_STEPS = 11	#(11)    3D x, y steps

#
# MAIN BODY
#
fig  = plt.figure(figsize=(WIDTH, HEIGHT))
axis = fig.add_subplot(111, projection="3d")

# dataset

x = np.linspace(0.0, 1.0, XY_STEPS)
y = np.linspace(0.0, 1.0, XY_STEPS)
x_mesh, y_mesh = np.meshgrid(x,y)
    
z_mesh = np.empty_like(x_mesh)
for i in range(XY_STEPS) :
  for j in range(XY_STEPS) :
    z_mesh[i][j] = ( x[i]**2 + y[j]**2 )/2.0

# setup axis

axis.cla()   # clear the current axes state
axis.set_xlabel("x")
axis.set_ylabel("y")
axis.set_zlabel("z")
axis.set_xticks([0.0, 0.5, 1.0])
axis.set_yticks([0.0, 0.5, 1.0])
axis.set_zticks([0.0, 0.5, 1.0])   

# draw 3D graph

axis.plot_surface(x_mesh, y_mesh, z_mesh, cmap = "cool")


plt.draw()
plt.show() 
