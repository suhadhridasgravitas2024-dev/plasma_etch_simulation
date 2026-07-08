"""
constants.py

ONLY things that are physically fixed and true regardless of which
experiment you're running: fundamental constants, and the Ar/SF6
cross-section landmark energies + collision energy losses (these come
from real atomic/molecular physics, not from your choice of run
parameters).

Anything that changes from one simulation run to the next --
N_PARTICLES, N_STEPS, dt, E0, T_GAS, sf6_fraction, pressure_mtorr, etc.
-- does NOT belong here. Decide those in your main/experiment file each
time you instantiate the simulation.

    from constants import k_B, e_charge, m_e
"""

import scipy.constants as const

# ---------------------------------------------------------------
# Fundamental physical constants
# ---------------------------------------------------------------
k_B      = const.k      # Boltzmann constant, J/K
e_charge = const.e      # elementary charge, C
m_e      = const.m_e    # electron mass, kg
M_E = m_e      # Electron mass in kg
Q_E = e_charge    

# ---------------------------------------------------------------
# Argon cross-section landmark energies (eV)
# ---------------------------------------------------------------
AR_RT_MINIMUM_E  = 0.33    # Ramsauer-Townsend minimum location
AR_EXC_THRESHOLD = 11.5    # lumped excitation threshold
AR_ION_THRESHOLD = 15.76   # ionization energy of Ar

# ---------------------------------------------------------------
# SF6 cross-section landmark energies (eV)
# ---------------------------------------------------------------
SF6_ATTACH_PEAK_E = 0.0    # dissociative attachment peaks near 0 eV
SF6_VIB_THRESHOLD = 0.1    # vibrational/inelastic onset
SF6_ION_THRESHOLD = 15.3   # ionization threshold

# ---------------------------------------------------------------
# Energy losses per collision type (eV)
# ---------------------------------------------------------------
EXC_LOSS_EV     = 11.5     # Ar excitation loss
ION_LOSS_EV_AR  = 15.76    # Ar ionization loss
ION_LOSS_EV_SF6 = 15.3     # SF6 ionization loss

VIB_LOSS_CROSSOVER = 2.0   # eV threshold splitting "low" vs "high" vib loss regime
VIB_LOSS_EV_LOW     = 0.1  # vibrational loss below crossover
VIB_LOSS_EV_HIGH    = 0.5  # vibrational loss above crossover (lumped electronic exc.)


M_E = 9.11e-31       # Electron mass (kg)
Q_E = 1.602e-19      # Elementary charge (C)
EPSILON_0 = 8.854e-12 # Vacuum permittivity (F/m)
K_B = 1.38e-23       # Boltzmann constant (J/K)