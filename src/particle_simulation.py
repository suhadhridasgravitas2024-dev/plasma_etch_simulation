# In this file write functions for all particle simulation, with a well defined input and output

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# from plasma_trial.medium_git_trial import getAcc

def get_acceleration(pos, vel,Nx, boxsize, n0, Gmtx, Lmtx):
    # Calculate Electron Number Density on the Mesh
    N = pos.shape[0]
    dx = boxsize / Nx
    j = np.floor(pos / dx).astype(int)
    jp1 = j + 1
    weight_j = (jp1 * dx - pos) / dx
    weight_jp1 = (pos - j * dx) / dx
    jp1 = np.mod(jp1, Nx)  # periodic BC
    n = np.bincount(j[:, 0], weights=weight_j[:, 0], minlength=Nx)
    n += np.bincount(jp1[:, 0], weights=weight_jp1[:, 0], minlength=Nx)
    n *= n0 * boxsize / N / dx

    # Solve Poisson's Equation: laplacian(phi) = n-n0
    phi_grid = spsolve(Lmtx, n - n0, permc_spec="MMD_AT_PLUS_A")

    # Apply Derivative to get the Electric field
    E_grid = -Gmtx @ phi_grid

    # Interpolate grid value onto particle locations
    E = weight_j * E_grid[j] + weight_jp1 * E_grid[jp1]
    a = -E

    return a

def simulate(pos, A, Nh, fig, ax, dt, vel, acc, tEnd, Nx, boxsize, n0, Gmtx, Lmtx):
    
    """
    Update the animation frame
    """
    # global pos, A, Nh, fig, ax

    
    
    # (1/2) kick
    vel += acc * dt / 2.0
    
    # drift (and apply periodic boundary conditions)
    pos += vel * dt
    pos = np.mod(pos, boxsize)
    
    # update accelerations
    acc = get_acceleration(pos, vel, Nx, boxsize, n0, Gmtx, Lmtx)
    
    # (1/2) kick
    vel += acc * dt / 2.0

    
    ax.set_xlim(0, boxsize)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("x")
    ax.set_ylabel("v")


    #it is how the code creates two oppositely moving electron beams that interact and produce the two-stream instability.
    scatter1 = ax.scatter(pos[0:Nh], vel[0:Nh], s=0.4, color="blue", alpha=0.5)
    scatter2 = ax.scatter(pos[Nh:], vel[Nh:], s=0.4, color="red", alpha=0.5)
    
    # Update scatter plot data (faster than clearing and redrawing)
    

    # scat.set_offsets(np.column_stack((x, y)))
    # ax.set_title(f"Frame {frame}", color='white', fontsize=10)

    # data=np.column_stack((scatter1, scatter2))
    # Return the modified artists for blitting
    return pos, vel, acc, scatter1, scatter2

