# -*- coding: utf_8 *-*
"""
test_plot2d_1.py - plot plural 2d-graph


"""
import numpy as np
import matplotlib.pyplot as plt

# PARAMETER(S)

N_ROW  = 2
N_COL  = 1
WIDTH  = 8
HEIGHT = 5

#
# MAIN BODY
#
fig, axes = plt.subplots(
    N_ROW,                   # the number of rows
    N_COL,                   # the number of columns
    figsize=(WIDTH, HEIGHT), # width, height (inches)
    tight_layout=True)

#if N_ROW * N_COL == 1: axis = axes
#else                 : axis = axes[0]

# dataset

x = np.arange(-10.0, 10.0, step=0.5)
y = x**3 - 3*x

# draw graph 1 on axes[0]

axis = axes[0]
axis.set_title("y = x**3 - 3*x")

axis.set_xlabel("x")
axis.set_xlim(-10.0, 10.0)

axis.set_ylabel("y")
axis.set_ylim(-1000.0, 1000.0)

axis.plot(x, y)

# draw graph 2 on axes[1]

axis = axes[1]
axis.set_title("zoom in")

axis.set_xlabel("x")
axis.set_xlim(-10.0, 10.0)

axis.set_ylabel("y")
axis.set_ylim(-5.0, 5.)  # changed

axis.plot(x, y,
    color     ='red',
    marker    ='o',
    linestyle ='--')     # changed    

plt.draw()
plt.show() 
