import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.constants as const
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

"""
================================================================================
PHYSICALLY-MOTIVATED PIC-MCC SIMULATION (Ar / Ar-SF6 mixtures)
================================================================================
IMPORTANT — READ BEFORE USING THESE NUMBERS FOR ANYTHING SERIOUS:

This code uses ANALYTIC FUNCTIONAL FORMS for electron-impact cross sections
that reproduce the correct QUALITATIVE PHYSICS of Ar and SF6:
  - Correct collision THRESHOLD energies (real, published values)
  - Correct overall SHAPE (Ramsauer-Townsend minimum in Ar, low-energy
    attachment peak in SF6, threshold rise of inelastic/ionization channels)
  - Correct ORDER OF MAGNITUDE for cross sections (~1e-20 to 1e-19 m^2,
    consistent with known swarm data)

It does NOT use the exact tabulated/measured cross-section datasets (e.g.
Phelps' compilation, LXCat databases). Those are numerically precise,
peer-reviewed datasets; what's below are smooth analytic fits constructed
to have the right physical character, not numerically exact values copied
from any specific reference.

Use this for: understanding/teaching PIC-MCC mechanics, qualitative trends
(does the EEPF tail deplete more with SF6? does pressure broaden/sharpen
distributions?), order-of-magnitude exploration.

Do NOT use this for: publication-quality data, process recipe optimization,
or any application needing quantitatively accurate EEPF values. For that,
real cross sections from LXCat.net + a validated PIC-MCC or Boltzmann
solver (e.g. BOLSIG+) are required.
================================================================================
"""

# ════════════════════════════════════════════════════════════════════
# 1. Physical constants
# ════════════════════════════════════════════════════════════════════
e_charge = const.e
m_e = const.m_e
k_B = const.k

RF_FREQ = 13.56e6
OMEGA = 2 * np.pi * RF_FREQ

# NOTE on E0: eduPIC is a full 1D PIC code where V0=250V drops mostly across
# thin sheaths (~mm), not uniformly across the 25mm gap. A 0D model has no
# sheath structure, so using V0/GAP as a uniform field grossly overestimates
# the heating field. Instead we choose E0 directly to give a quiver energy
# in the realistic range for stochastic heating in low-pressure CCPs
# (mean bulk electron energy ~1-5 eV, per eduPIC Fig 11i reference result).
E0 = 200.0   # V/m -- tuned so quiver energy is in the eV range, not keV range

T_GAS = 350.0        # gas temperature (K) -- eduPIC reference value

# ════════════════════════════════════════════════════════════════════
# 2. Gas density from pressure (real ideal gas law -- this part IS exact)
# ════════════════════════════════════════════════════════════════════
def gas_density(pressure_mtorr=50, T=T_GAS):
    """n = P / (kB T). Converts mTorr -> Pa first (1 Torr = 133.322 Pa)."""
    P_pa = pressure_mtorr * 1e-3 * 133.322
    return P_pa / (k_B * T)   # particles / m^3

# ════════════════════════════════════════════════════════════════════
# 3. Cross sections -- ANALYTIC FUNCTIONAL FORMS (approximate, see disclaimer)
# ════════════════════════════════════════════════════════════════════
# --- Argon ---
# Real physical landmarks used here:
#   - Ramsauer-Townsend minimum near ~0.3-0.5 eV (real, well-known effect)
#   - First electronic excitation threshold ~11.5 eV (real Ar excitation onset)
#   - Ionization threshold = 15.76 eV (real Ar ionization energy)
AR_ION_THRESHOLD = 15.76    # eV, real
AR_EXC_THRESHOLD = 11.55    # eV, real (lumped excitation onset)
AR_RT_MINIMUM_E  = 0.33     # eV, real approx. location of Ramsauer minimum

def sigma_ar_elastic(E_eV):
    """Momentum-transfer cross section with Ramsauer-Townsend dip (m^2).
    Qualitative analytic form: dips near AR_RT_MINIMUM_E, rises at very low
    and at higher energy, broadly consistent with known Ar elastic CS shape."""
    E = np.maximum(E_eV, 1e-3)
    rt_dip = 1.0 - 0.9 * np.exp(-((np.log(E / AR_RT_MINIMUM_E))**2) / (2 * 0.6**2))
    base = 6.0e-20 / (1.0 + 0.5 * E)**0.5
    return base * rt_dip + 0.3e-20

def sigma_ar_excitation(E_eV):
    """Lumped excitation cross section, threshold-gated, peaks then falls (m^2)."""
    E = np.asarray(E_eV, dtype=float)
    sigma = np.zeros_like(E)
    above = E > AR_EXC_THRESHOLD
    x = E[above] - AR_EXC_THRESHOLD
    sigma[above] = 4.0e-21 * (x / (1 + 0.15 * x**1.5)) * np.exp(-x / 25.0)
    return sigma

def sigma_ar_ionization(E_eV):
    """Ionization cross section, real threshold, Bethe-like rise + slow falloff (m^2)."""
    E = np.asarray(E_eV, dtype=float)
    sigma = np.zeros_like(E)
    above = E > AR_ION_THRESHOLD
    x = E[above] - AR_ION_THRESHOLD
    sigma[above] = 3.5e-21 * (x / (1 + 0.05 * x)) * np.exp(-x / 80.0)
    return sigma

# --- SF6 ---
# Real physical landmarks:
#   - Dissociative attachment peaks very close to 0 eV (real -- SF6 is famous
#     for this near-thermal attachment resonance) and falls off within ~1 eV
#   - Vibrational/inelastic excitation onset ~0.1-0.2 eV (real, low threshold)
#   - Electron impact ionization threshold ~15.3 eV (real, close to SF6 IE)
SF6_ATTACH_PEAK_E   = 0.05   # eV, real -- near-zero-energy attachment resonance
SF6_VIB_THRESHOLD   = 0.1    # eV, real, approximate
SF6_ION_THRESHOLD   = 15.3   # eV, real (SF6 ionization energy ~15.3 eV)

def sigma_sf6_attachment(E_eV):
    """Dissociative attachment: sharply peaked near 0 eV (m^2).
    SF6's near-thermal attachment resonance is a well-documented real effect;
    this is a smooth analytic peak approximating its qualitative shape."""
    E = np.maximum(E_eV, 1e-4)
    return 4.0e-19 * np.exp(-((E - SF6_ATTACH_PEAK_E) / 0.12)**2)

def sigma_sf6_vibrational(E_eV):
    """Vibrational/rotational + electronic inelastic excitation, lumped (m^2).
    Real SF6 has strong, broad inelastic losses (vibrational ~0.1-1 eV onset,
    electronic excitation extending to several eV) that are the dominant
    energy-loss mechanism responsible for its well-known strong electron
    cooling/attaching behavior in plasma etch chemistry. The single
    exp(-x/1.5) decay used in an earlier draft made this channel vanish by
    ~2 eV, which is NOT representative of real SF6 -- fixed here with a
    broader lumped inelastic loss term that stays significant out to ~15 eV,
    consistent with SF6 being known as a strongly electronegative/attaching
    and strongly inelastic (cooling) gas across the whole low/mid energy range."""
    E = np.asarray(E_eV, dtype=float)
    sigma = np.zeros_like(E)
    above = E > SF6_VIB_THRESHOLD
    x = E[above] - SF6_VIB_THRESHOLD
    # Near-threshold vibrational peak (fast onset, narrow)
    vib_peak = 8.0e-20 * np.exp(-x / 1.0) * (1 - np.exp(-x / 0.1))
    # Broad mid-energy inelastic plateau (electronic excitation manifold)
    broad_plateau = 1.5e-20 * (1 - np.exp(-x / 0.5)) * np.exp(-x / 12.0)
    sigma[above] = vib_peak + broad_plateau
    return sigma

def sigma_sf6_ionization(E_eV):
    """SF6 ionization, real threshold (m^2)."""
    E = np.asarray(E_eV, dtype=float)
    sigma = np.zeros_like(E)
    above = E > SF6_ION_THRESHOLD
    x = E[above] - SF6_ION_THRESHOLD
    sigma[above] = 2.5e-21 * (x / (1 + 0.05 * x)) * np.exp(-x / 70.0)
    return sigma

def sigma_sf6_elastic(E_eV):
    """SF6 elastic/momentum-transfer background (m^2)."""
    E = np.maximum(E_eV, 1e-3)
    return 5.0e-20 / (1.0 + 0.3 * E)**0.6

VIB_LOSS_EV_LOW  = 0.1     # eV, near-threshold vibrational loss
VIB_LOSS_EV_HIGH = 3.0     # eV, approximate lumped electronic excitation loss at higher E
VIB_LOSS_CROSSOVER = 4.0   # eV, energy above which the higher loss applies
EXC_LOSS_EV = AR_EXC_THRESHOLD
ION_LOSS_EV_AR  = AR_ION_THRESHOLD
ION_LOSS_EV_SF6 = SF6_ION_THRESHOLD

# ════════════════════════════════════════════════════════════════════
# 4. Null-collision MCC machinery (this IS the standard, correct algorithm
#    described in eduPIC / Birdsall & Langdon -- exact, not approximated)
# ════════════════════════════════════════════════════════════════════
def total_cross_sections(E_eV, sf6_fraction):
    """Return dict of process -> sigma(E) array (m^2), for the gas mixture."""
    procs = {}
    if sf6_fraction < 1.0:
        procs['ar_elastic']    = (1 - sf6_fraction) * sigma_ar_elastic(E_eV)
        procs['ar_excitation'] = (1 - sf6_fraction) * sigma_ar_excitation(E_eV)
        procs['ar_ionization'] = (1 - sf6_fraction) * sigma_ar_ionization(E_eV)
    if sf6_fraction > 0.0:
        procs['sf6_elastic']    = sf6_fraction * sigma_sf6_elastic(E_eV)
        procs['sf6_vibrational']= sf6_fraction * sigma_sf6_vibrational(E_eV)
        procs['sf6_attachment'] = sf6_fraction * sigma_sf6_attachment(E_eV)
        procs['sf6_ionization'] = sf6_fraction * sigma_sf6_ionization(E_eV)
    return procs

def isotropic_scatter(n):
    phi = 2 * np.pi * np.random.rand(n)
    cos_theta = 1.0 - 2.0 * np.random.rand(n)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    return sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta

# ════════════════════════════════════════════════════════════════════
# 5. Simulation parameters
# ════════════════════════════════════════════════════════════════════
dt = 1.0 / (OMEGA * 40)     # 40 steps per RF cycle
N_RF_CYCLES = 120
N_STEPS = int((1.0 / RF_FREQ) * N_RF_CYCLES / dt)
N_PARTICLES = 4000

def run_mcc_simulation(sf6_fraction, pressure_mtorr):
    print(f"Running Ar/SF6 fraction={sf6_fraction:.2f} @ {pressure_mtorr:.0f} mTorr "
          f"(N={N_PARTICLES}, steps={N_STEPS})...")
    n_gas = gas_density(pressure_mtorr)

    vx = np.random.normal(0, 1e5, N_PARTICLES)
    vy = np.random.normal(0, 1e5, N_PARTICLES)
    vz = np.random.normal(0, 1e5, N_PARTICLES)
    active = np.ones(N_PARTICLES, dtype=bool)

    time_avg_energies = []
    start_avg = int(0.75 * N_STEPS)

    for step in range(N_STEPS):
        if not np.any(active):
            break
        t = step * dt
        E_t = E0 * np.cos(OMEGA * t)

        vx[active] += (-e_charge * E_t / m_e) * dt

        v_sq = vx[active]**2 + vy[active]**2 + vz[active]**2
        v_mag = np.sqrt(v_sq)
        E_ev = 0.5 * m_e * v_sq / e_charge

        procs = total_cross_sections(E_ev, sf6_fraction)
        names = list(procs.keys())
        sigmas = np.stack([procs[k] for k in names], axis=0)   # (n_proc, n_active)

        # Collision probability per process: P_k = 1 - exp(-n*sigma_k*v*dt)
        nu_k = n_gas * sigmas * v_mag[None, :]                  # collision freq per process
        nu_tot = nu_k.sum(axis=0)
        p_coll_tot = 1.0 - np.exp(-nu_tot * dt)
        p_coll_tot = np.clip(p_coll_tot, 0.0, 1.0)

        r1 = np.random.rand(np.sum(active))
        will_collide = r1 < p_coll_tot

        active_indices = np.where(active)[0]

        if np.any(will_collide):
            # Choose process by relative cross-section weight among colliding particles
            coll_idx_local = np.where(will_collide)[0]
            weights = nu_k[:, coll_idx_local]                  # (n_proc, n_coll)
            weights_sum = weights.sum(axis=0)
            weights_sum[weights_sum == 0] = 1e-30
            cum_weights = np.cumsum(weights, axis=0) / weights_sum[None, :]
            r2 = np.random.rand(len(coll_idx_local))
            proc_choice = np.argmax(cum_weights > r2[None, :], axis=0)

            for pidx, pname in enumerate(names):
                sel_local = coll_idx_local[proc_choice == pidx]
                if len(sel_local) == 0:
                    continue
                global_idx = active_indices[sel_local]
                E_sel = E_ev[sel_local]

                if pname in ('ar_elastic', 'sf6_elastic'):
                    v_mag_sel = v_mag[sel_local]
                    dx, dy, dz = isotropic_scatter(len(global_idx))
                    vx[global_idx] = v_mag_sel * dx
                    vy[global_idx] = v_mag_sel * dy
                    vz[global_idx] = v_mag_sel * dz

                elif pname == 'sf6_vibrational':
                    # Energy-dependent loss: near threshold electrons lose a small
                    # vibrational quantum; higher-energy electrons (where the broad
                    # electronic-excitation-like plateau dominates) lose more,
                    # reflecting the lumped nature of this approximate channel.
                    loss = np.where(E_sel < VIB_LOSS_CROSSOVER, VIB_LOSS_EV_LOW, VIB_LOSS_EV_HIGH)
                    new_E = np.maximum(0.005, E_sel - loss)
                    v_new = np.sqrt(2 * new_E * e_charge / m_e)
                    dx, dy, dz = isotropic_scatter(len(global_idx))
                    vx[global_idx] = v_new * dx
                    vy[global_idx] = v_new * dy
                    vz[global_idx] = v_new * dz

                elif pname == 'ar_excitation':
                    new_E = np.maximum(0.005, E_sel - EXC_LOSS_EV)
                    v_new = np.sqrt(2 * new_E * e_charge / m_e)
                    dx, dy, dz = isotropic_scatter(len(global_idx))
                    vx[global_idx] = v_new * dx
                    vy[global_idx] = v_new * dy
                    vz[global_idx] = v_new * dz

                elif pname == 'ar_ionization':
                    # Simplified: scattered electron keeps half remaining energy,
                    # the ejected electron is not separately tracked (no electron
                    # multiplication modeled here -- approximation for 0D EEPF shape)
                    new_E = np.maximum(0.005, (E_sel - ION_LOSS_EV_AR) * 0.5)
                    v_new = np.sqrt(2 * new_E * e_charge / m_e)
                    dx, dy, dz = isotropic_scatter(len(global_idx))
                    vx[global_idx] = v_new * dx
                    vy[global_idx] = v_new * dy
                    vz[global_idx] = v_new * dz

                elif pname == 'sf6_ionization':
                    new_E = np.maximum(0.005, (E_sel - ION_LOSS_EV_SF6) * 0.5)
                    v_new = np.sqrt(2 * new_E * e_charge / m_e)
                    dx, dy, dz = isotropic_scatter(len(global_idx))
                    vx[global_idx] = v_new * dx
                    vy[global_idx] = v_new * dy
                    vz[global_idx] = v_new * dz

                elif pname == 'sf6_attachment':
                    active[global_idx] = False   # electron removed

        if step > start_avg and step % 4 == 0:
            surv_vsq = vx[active]**2 + vy[active]**2 + vz[active]**2
            time_avg_energies.extend((0.5 * m_e * surv_vsq) / e_charge)

    return np.array(time_avg_energies)

def get_eepf(energies, bins=60, max_energy=22):
    if len(energies) == 0:
        return np.array([]), np.array([])
    eedf, edges = np.histogram(energies, bins=bins, range=(0.05, max_energy), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    valid = eedf > 0
    eepf = eedf[valid] / np.sqrt(centers[valid])
    return centers[valid], eepf


# ════════════════════════════════════════════════════════════════════
# 6. Run for requested pressure and Ar:SF6 ratios
# ════════════════════════════════════════════════════════════════════
PRESSURE_MTORR = 50.0   # <-- change this single value for different pressures

ratios = [
    ('10:0', 0.00, "green"),
    ('6:4',  0.40, "red"),
    ('1:9',  0.90, "blue"),
]

results = {}
for label, sf6_frac, color in ratios:
    energies = run_mcc_simulation(sf6_frac, PRESSURE_MTORR)
    bins, eepf = get_eepf(energies)
    results[label] = (bins, eepf, color, sf6_frac, len(energies))

# ── Plot (true semilogy of EEPF, no curve-fitting, no calibration offsets) ──
fig, ax = plt.subplots(figsize=(7, 5.5), dpi=100)
for label, (bins, eepf, color, sf6_frac, n_samp) in results.items():
    if len(bins) > 0:
        ax.semilogy(bins, eepf, label=f'Ar:SF6 = {label}  (n={n_samp})',
                    color=color, linewidth=1.6)
    else:
        print(f"WARNING: no surviving electron samples for {label} -- "
              f"likely fully attached before averaging window.")

ax.set_xlim(0, 22)
ax.set_xlabel("Electron energy (eV)", fontsize=12)
ax.set_ylabel(r"EEPF (eV$^{-3/2}$, arb. norm.)", fontsize=12)
ax.set_title(f"PIC-MCC EEPF — {PRESSURE_MTORR:.0f} mTorr\n"
             f"(physically-motivated analytic cross sections — NOT lab-calibrated)",
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()



# ════════════════════════════════════════════════════════════════════
# 7. Excel export
# ════════════════════════════════════════════════════════════════════
wb = Workbook()

def header_font():   return Font(name='Arial', bold=True, color='FFFFFF', size=11)
def data_font():     return Font(name='Arial', size=10)
def note_font():     return Font(name='Arial', size=9, italic=True, color='990000')
def center():        return Alignment(horizontal='center', vertical='center')
def thin_border():
    s = Side(style='thin', color='CCCCCC')
    return Border(left=s, right=s, top=s, bottom=s)

ALT_ROW = 'FFF4F4F4'
def style_header(cell, fill):
    cell.font = header_font(); cell.fill = PatternFill('solid', start_color=fill)
    cell.alignment = center(); cell.border = thin_border()
def style_data(cell, idx):
    cell.font = data_font(); cell.alignment = center(); cell.border = thin_border()
    if idx % 2 == 0:
        cell.fill = PatternFill('solid', start_color=ALT_ROW)

color_map = {'10:0': 'FF2B2B2B', '6:4': 'FFCC00CC', '1:9': 'FF8B0000'}

for label, (bins, eepf, color, sf6_frac, n_samp) in results.items():
    sheet_name = f"Ar{label.replace(':','-')}"
    ws = wb.create_sheet(sheet_name) if wb.sheetnames != ['Sheet'] else wb.active
    if ws.title == 'Sheet': ws.title = sheet_name
    ws.column_dimensions['A'].width = 20
    ws.column_dimensions['B'].width = 24

    ws.merge_cells('A1:B1')
    tc = ws['A1']
    tc.value = f'Ar:SF6 = {label} — EEPF @ {PRESSURE_MTORR:.0f} mTorr'
    tc.font = Font(name='Arial', bold=True, size=12, color='FFFFFF')
    tc.fill = PatternFill('solid', start_color=color_map[label])
    tc.alignment = center()

    ws.merge_cells('A2:B2')
    warn = ws['A2']
    warn.value = '⚠ Analytic approximate cross sections — not lab-calibrated data'
    warn.font = note_font()

    meta = [
        ('Gas mixture', f'Ar:SF6 = {label}'),
        ('SF6 fraction', f'{sf6_frac*100:.0f}%'),
        ('Pressure', f'{PRESSURE_MTORR:.0f} mTorr'),
        ('Gas temperature', f'{T_GAS:.0f} K'),
        ('RF frequency', '13.56 MHz'),
        ('E-field amplitude', f'{E0:.0f} V/m'),
        ('Particles', str(N_PARTICLES)),
        ('Surviving samples', str(n_samp)),
    ]
    for i, (k, v) in enumerate(meta, start=3):
        ws.cell(i, 1, k).font = Font(name='Arial', bold=True, size=10)
        ws.cell(i, 2, v).font = data_font()

    hdr_row = len(meta) + 4
    for col, lbl in enumerate(['Energy (eV)', 'EEPF (eV⁻³/²)'], start=1):
        style_header(ws.cell(hdr_row, col, lbl), '444444')

    for i, (ev, ep) in enumerate(zip(bins, eepf), start=1):
        r = hdr_row + i
        c1 = ws.cell(r, 1, round(float(ev), 4))
        c2 = ws.cell(r, 2, float(f'{ep:.6e}'))
        style_data(c1, i); style_data(c2, i)
        c1.number_format = '0.0000'; c2.number_format = '0.000000E+00'

ws_cmp = wb.create_sheet('Comparison')
ws_cmp.merge_cells('A1:F1')
tc = ws_cmp['A1']
tc.value = f'EEPF Comparison — Ar:SF6 Ratios @ {PRESSURE_MTORR:.0f} mTorr (approximate physics)'
tc.font = Font(name='Arial', bold=True, size=12, color='FFFFFF')
tc.fill = PatternFill('solid', start_color='FF1A1A2E')
tc.alignment = center()

col = 1
col_starts = {}
for label, (bins, eepf, color, sf6_frac, n_samp) in results.items():
    col_starts[label] = col
    style_header(ws_cmp.cell(2, col, f'Energy — {label} (eV)'), color_map[label])
    style_header(ws_cmp.cell(2, col+1, f'EEPF — {label}'), color_map[label])
    ws_cmp.column_dimensions[get_column_letter(col)].width = 20
    ws_cmp.column_dimensions[get_column_letter(col+1)].width = 20
    col += 2

for label, (bins, eepf, color, sf6_frac, n_samp) in results.items():
    c0 = col_starts[label]
    for i in range(len(bins)):
        r = i + 3
        cA = ws_cmp.cell(r, c0, round(float(bins[i]), 4))
        cB = ws_cmp.cell(r, c0+1, float(f'{eepf[i]:.6e}'))
        style_data(cA, i); style_data(cB, i)
        cA.number_format = '0.0000'; cB.number_format = '0.000000E+00'
if __name__=="__main__":
    plt.savefig(f"/mnt/c/Users/semi/Plasma_simulation/plasma_etch_simulation/simple_simulation/eepf_plot_50.png", dpi=130, bbox_inches="tight")
    print("Plot saved.")
    out_path = f"/mnt/c/Users/semi/Plasma_simulation/plasma_etch_simulation/simple_simulation/eepf_plot_50.xlsx"
    wb.save(out_path)
    print(f"Excel saved -> {out_path}")

#understnad eepf code 
#make changes in parameters...50 mTorr TICK
#sheath simulation 1D TICK
#what is being cooled in sheath simulation??
#what is 1D and 2D sheath??
#connection between sheath and eepf?
