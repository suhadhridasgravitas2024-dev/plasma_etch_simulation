import numpy as np
import matplotlib
matplotlib.use('Agg')  # FIX: non-interactive backend — plt.show() fails without a display
import matplotlib.pyplot as plt
import scipy.constants as const

"""
0D-3V PIC-MCC (Monte Carlo Collision) Simulation
Simulates electron heating in an RF electric field and collisions 
with Ar and SF6 neutral gas molecules to derive the EEPF.
"""

# ==========================================
# 1. Physical Constants & Simulation Params
# ==========================================
e = const.e
m_e = const.m_e
RF_FREQ = 13.56e6
OMEGA = 2 * np.pi * RF_FREQ
E0 = 600.0

# Time parameters — reduced for speed
dt = 1.0 / (OMEGA * 30)   # 30 steps/cycle (was 50, still resolves RF well)
N_RF_CYCLES = 150          # 150 cycles to reach steady state (was 300)
N_STEPS = int((1.0 / RF_FREQ) * N_RF_CYCLES / dt)

N_PARTICLES = 1000         # Reduced from 50000


# ==========================================
# 2. Collision Frequencies
# ==========================================
NU_AR_ELASTIC = 5e7
SF6_INELASTIC_THRES = 2.5
SF6_INELASTIC_LOSS = 2.0
NU_SF6_INELASTIC = 8e7

def get_collision_probs(energies_ev, sf6_fraction):
    """Return per-particle collision probabilities (clamped to [0,1])."""
    # Argon elastic
    p_ar_elastic = (1.0 - sf6_fraction) * NU_AR_ELASTIC * dt * np.ones_like(energies_ev)

    # SF6 inelastic (threshold-gated)
    p_sf6_inelastic = np.zeros_like(energies_ev)
    mask_inelastic = energies_ev > SF6_INELASTIC_THRES
    p_sf6_inelastic[mask_inelastic] = sf6_fraction * NU_SF6_INELASTIC * dt

    # SF6 attachment (peaks at low energy)
    p_sf6_attach = sf6_fraction * (1e7 * dt) / (energies_ev + 0.1)

    # Clamp all probabilities to [0, 1]
    p_ar_elastic   = np.clip(p_ar_elastic,   0.0, 1.0)
    p_sf6_inelastic = np.clip(p_sf6_inelastic, 0.0, 1.0)
    p_sf6_attach   = np.clip(p_sf6_attach,   0.0, 1.0)

    return p_ar_elastic, p_sf6_inelastic, p_sf6_attach


def isotropic_scatter(n):
    """Return new unit-sphere velocity directions for n particles."""
    phi = 2 * np.pi * np.random.rand(n)
    cos_theta = 1.0 - 2.0 * np.random.rand(n)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    return sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta


# ==========================================
# 3. Main PIC-MCC Engine
# ==========================================
def run_mcc_simulation(sf6_fraction):
    print(f"Running simulation with {sf6_fraction*100:.0f}% SF6...")

    vx = np.random.normal(0, 1e5, N_PARTICLES)
    vy = np.random.normal(0, 1e5, N_PARTICLES)
    vz = np.random.normal(0, 1e5, N_PARTICLES)
    active = np.ones(N_PARTICLES, dtype=bool)

    time_averaged_energies = []
    start_averaging_step = int(0.8 * N_STEPS)

    for step in range(N_STEPS):
        if not np.any(active):
            break  # All electrons attached — nothing left to simulate

        t = step * dt
        E_t = E0 * np.sin(OMEGA * t)

        # --- 1. Accelerate only active particles ---
        vx[active] += (-e * E_t / m_e) * dt

        # --- 2. Compute energies only for active particles ---
        v_sq_active = vx[active]**2 + vy[active]**2 + vz[active]**2
        energies_ev_active = (0.5 * m_e * v_sq_active) / e

        p_ar_el, p_sf6_inel, p_sf6_att = get_collision_probs(energies_ev_active, sf6_fraction)

        # Draw independent random numbers for each collision channel (FIX: no shared rand subtraction)
        r_att  = np.random.rand(np.sum(active))
        r_inel = np.random.rand(np.sum(active))
        r_el   = np.random.rand(np.sum(active))

        active_indices = np.where(active)[0]

        # Attachment (removes particle)
        att_local = r_att < p_sf6_att
        active[active_indices[att_local]] = False

        # Inelastic — only on still-active, non-attached particles
        still_active_local = ~att_local
        inel_local = still_active_local & (r_inel < p_sf6_inel)
        if np.any(inel_local):
            idx = active_indices[inel_local]
            new_e = np.maximum(0.01, energies_ev_active[inel_local] - SF6_INELASTIC_LOSS)
            v_mag = np.sqrt(new_e * e * 2 / m_e)
            dx, dy, dz = isotropic_scatter(len(idx))
            vx[idx] = v_mag * dx
            vy[idx] = v_mag * dy
            vz[idx] = v_mag * dz

        # Elastic — skip particles already handled by attachment or inelastic
        el_local = still_active_local & ~inel_local & (r_el < p_ar_el)
        if np.any(el_local):
            idx = active_indices[el_local]
            v_mag = np.sqrt(v_sq_active[el_local])
            dx, dy, dz = isotropic_scatter(len(idx))
            vx[idx] = v_mag * dx
            vy[idx] = v_mag * dy
            vz[idx] = v_mag * dz

        # --- 3. Collect energies in the final 20% of steps ---
        if step > start_averaging_step and step % 5 == 0:
            surviving_vsq = vx[active]**2 + vy[active]**2 + vz[active]**2
            time_averaged_energies.extend((0.5 * m_e * surviving_vsq) / e)

    return np.array(time_averaged_energies)


# ==========================================
# 4. Run and Plot
# ==========================================
energies_pure_ar = run_mcc_simulation(sf6_fraction=0.0)
energies_mixture = run_mcc_simulation(sf6_fraction=0.4)


def get_eepf(energies, bins=70, max_energy=22):
    if len(energies) == 0:
        return np.array([]), np.array([])
    eedf, bin_edges = np.histogram(energies, bins=bins, range=(0.1, max_energy), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid_mask = eedf > 0
    eepf = eedf[valid_mask] / np.sqrt(bin_centers[valid_mask])
    return bin_centers[valid_mask], eepf


bins_ar,  eepf_ar  = get_eepf(energies_pure_ar)
bins_mix, eepf_mix = get_eepf(energies_mixture)

fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
ax.semilogy(bins_ar,  eepf_ar,  label='Pure Ar (0:10)',      color='black',  linewidth=1.5)
ax.semilogy(bins_mix, eepf_mix, label='Ar/SF6 Mixture (4:6)', color='purple', linewidth=1.5)

ax.set_xlim(0, 22)
ax.set_ylim(1e-4, 1)
ax.set_xlabel("Electron energy (eV)", fontsize=12)
ax.set_ylabel("log EEPF ($eV^{-3/2}$)", fontsize=12)
ax.set_title("Time-Averaged PIC-MCC Simulated EEPF", fontsize=13)
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/c/Users/semi/Plasma_simulation/plasma_etch_simulation/simple_simulation/eepf_plot.png", dpi=120, bbox_inches="tight")
print("Plot saved.")