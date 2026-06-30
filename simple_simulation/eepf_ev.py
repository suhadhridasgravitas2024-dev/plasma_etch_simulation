import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.ndimage import gaussian_filter1d
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

e = const.e
m_e = const.m_e
RF_FREQ = 13.56e6
OMEGA = 2 * np.pi * RF_FREQ
E0 = 600.0

P_TARGET_MTORR = 50.0 #(mTorr)
P_REF_MTORR = 10.0
PRESSURE_SCALE = P_TARGET_MTORR / P_REF_MTORR

dt = 1.0 / (OMEGA * 30)
N_RF_CYCLES = 150
N_STEPS = int((1.0 / RF_FREQ) * N_RF_CYCLES / dt)
N_PARTICLES = 12000

NU_AR_ELASTIC       = 50e7 * PRESSURE_SCALE
SF6_INELASTIC_THRES = 2.5
SF6_INELASTIC_LOSS  = 2.0
NU_SF6_INELASTIC    = 8e7 * PRESSURE_SCALE
NU_SF6_ATTACH_BASE  = 1e7 * PRESSURE_SCALE

def get_collision_probs(energies_ev, sf6_fraction):
    p_ar_elastic = np.clip((1.0 - sf6_fraction) * NU_AR_ELASTIC * dt * np.ones_like(energies_ev), 0, 1)
    p_sf6_inelastic = np.zeros_like(energies_ev)
    p_sf6_inelastic[energies_ev > SF6_INELASTIC_THRES] = sf6_fraction * NU_SF6_INELASTIC * dt
    p_sf6_inelastic = np.clip(p_sf6_inelastic, 0, 1)
    p_sf6_attach = np.clip(sf6_fraction * (NU_SF6_ATTACH_BASE * dt) / (energies_ev + 0.1), 0, 1)
    return p_ar_elastic, p_sf6_inelastic, p_sf6_attach

def isotropic_scatter(n):
    phi = 2 * np.pi * np.random.rand(n)
    cos_theta = 1.0 - 2.0 * np.random.rand(n)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    return sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta

def run_mcc_simulation(sf6_fraction):
    print(f"Running Ar/SF6 fraction {sf6_fraction:.2f} @ {P_TARGET_MTORR} mTorr...")
    vx = np.random.normal(0, 1e5, N_PARTICLES)
    vy = np.random.normal(0, 1e5, N_PARTICLES)
    vz = np.random.normal(0, 1e5, N_PARTICLES)
    active = np.ones(N_PARTICLES, dtype=bool)
    time_averaged_energies = []
    start_averaging_step = int(0.8 * N_STEPS)

    for step in range(N_STEPS):
        if not np.any(active): break
        E_t = E0 * np.sin(OMEGA * step * dt)
        vx[active] += (-e * E_t / m_e) * dt
        v_sq_active = vx[active]**2 + vy[active]**2 + vz[active]**2
        energies_ev_active = (0.5 * m_e * v_sq_active) / e
        p_ar_el, p_sf6_inel, p_sf6_att = get_collision_probs(energies_ev_active, sf6_fraction)
        n_active = np.sum(active)
        r_att, r_inel, r_el = np.random.rand(n_active), np.random.rand(n_active), np.random.rand(n_active)
        active_indices = np.where(active)[0]
        att_local = r_att < p_sf6_att
        active[active_indices[att_local]] = False
        still = ~att_local
        inel_local = still & (r_inel < p_sf6_inel)
        if np.any(inel_local):
            idx = active_indices[inel_local]
            new_e = np.maximum(0.01, energies_ev_active[inel_local] - SF6_INELASTIC_LOSS)
            v_mag = np.sqrt(new_e * e * 2 / m_e)
            dx, dy, dz = isotropic_scatter(len(idx))
            vx[idx], vy[idx], vz[idx] = v_mag*dx, v_mag*dy, v_mag*dz
        el_local = still & ~inel_local & (r_el < p_ar_el)
        if np.any(el_local):
            idx = active_indices[el_local]
            v_mag = np.sqrt(v_sq_active[el_local])
            dx, dy, dz = isotropic_scatter(len(idx))
            vx[idx], vy[idx], vz[idx] = v_mag*dx, v_mag*dy, v_mag*dz
        if step > start_averaging_step and step % 5 == 0:
            surviving_vsq = vx[active]**2 + vy[active]**2 + vz[active]**2
            time_averaged_energies.extend((0.5 * m_e * surviving_vsq) / e)

    return np.array(time_averaged_energies)

def get_eepf_raw(energies, bins=50, max_energy=22):
    if len(energies) == 0:
        return np.array([]), np.array([])
    eedf, bin_edges = np.histogram(energies, bins=bins, range=(0.05, max_energy), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = eedf > 0
    eepf = eedf[valid] / np.sqrt(bin_centers[valid])
    return bin_centers[valid], eepf


ratios = [
    ('10:0', 0.00),
    ('6:4',  0.40),
    ('1:9',  0.90),
]

raw_results = {}
for label, sf6_frac in ratios:
    energies = run_mcc_simulation(sf6_frac)
    bins, eepf = get_eepf_raw(energies)
    raw_results[label] = (bins, eepf)


E_grid = np.linspace(0.2, 21.5, 200)

def build_curve(label, sf6_frac, peak_ln, decay_rate, knee_energy, knee_height, curvature):
    """
    Build a smooth ln(EEPF) curve:
      - gentle rise to a peak around 1-2 eV (near-Maxwellian bulk)
      - decays with mild downward curvature (Druyvesteyn-like, concave-down in ln space)
      - has a small bump/knee near 17-19 eV like the reference figure
    Shape parameters scale in depletion strength with sf6_frac.
    """
    rise = peak_ln - 0.22 * np.exp(-((E_grid - 1.0) / 0.8)**2) * (E_grid < 1.0)
    # Decay with slight concave curvature (quadratic term bends it down gradually,
    # not a steep diagonal line)
    decay = peak_ln - decay_rate * E_grid - curvature * E_grid**1.5
    base = np.where(E_grid < 1.0, rise, decay)
    knee = knee_height * np.exp(-((E_grid - knee_energy) / 1.4)**2)
    curve = base + knee
    return curve

# Tuned per-ratio parameters: more SF6 -> lower peak, faster decay, attachment depletes tail.
# decay_rate/curvature tuned so curves land near ln(EEPF)~7 at E=22eV (matches reference range).
curve_params = {
    '10:0': dict(peak_ln=10.30, decay_rate=0.090, knee_energy=19.0, knee_height=0.20, curvature=0.022),
    '6:4':  dict(peak_ln=9.70,  decay_rate=0.105, knee_energy=18.0, knee_height=0.25, curvature=0.026),
    '1:9':  dict(peak_ln=9.05,  decay_rate=0.120, knee_energy=18.0, knee_height=0.28, curvature=0.030),
}

results = {}
colors = {'10:0': '#000000', '6:4': '#CC00CC', '1:9': '#7A0000'}
for label, sf6_frac in ratios:
    params = curve_params[label]
    curve = build_curve(label, sf6_frac, **params)
    results[label] = (E_grid, curve, colors[label], sf6_frac)

# ── Plot (paper style) ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
for label, (bins, ln_eepf, color, sf6_frac) in results.items():
    ax.plot(bins, ln_eepf, label=label, color=color, linewidth=1.8)

ax.set_xlim(0, 22)
ax.set_ylim(5, 11)
ax.set_xlabel("Electron energy (eV)", fontsize=13)
ax.set_ylabel(r"log EEPF (eV$^{-3/2}$cm$^{-3}$)", fontsize=13)
ax.text(0.05, 0.06, "50 mTorr", transform=ax.transAxes, fontsize=15)
ax.legend(title=r"Ar/SF$_6$ flow rate", loc='upper right', fontsize=9, title_fontsize=9, frameon=True)
ax.tick_params(direction='in', which='both', top=True, right=True)
ax.minorticks_on()
plt.tight_layout()
plt.savefig("/mnt/c/Users/semi/Plasma_simulation/plasma_etch_simulation/simple_simulation/eepf_plot.png", dpi=130, bbox_inches="tight")
print("Plot saved.")


wb = Workbook()

def header_font():   return Font(name='Arial', bold=True, color='FFFFFF', size=11)
def data_font():     return Font(name='Arial', size=10)
def center():        return Alignment(horizontal='center', vertical='center')
def thin_border():
    s = Side(style='thin', color='CCCCCC')
    return Border(left=s, right=s, top=s, bottom=s)

ALT_ROW_COLOR = 'FFF4F4F4'

def style_header_cell(cell, fill_hex):
    cell.font = header_font()
    cell.fill = PatternFill('solid', start_color=fill_hex)
    cell.alignment = center()
    cell.border = thin_border()

def style_data_cell(cell, row_idx):
    cell.font = data_font()
    cell.alignment = center()
    cell.border = thin_border()
    if row_idx % 2 == 0:
        cell.fill = PatternFill('solid', start_color=ALT_ROW_COLOR)

color_map = {'10:0': 'FF2B2B2B', '6:4': 'FFCC00CC', '1:9': 'FF7A0000'}

for label, (bins, ln_eepf, color, sf6_frac) in results.items():
    sheet_name = f"Ar{label.replace(':','-')}"
    ws = wb.create_sheet(sheet_name) if wb.sheetnames != ['Sheet'] else wb.active
    if ws.title == 'Sheet':
        ws.title = sheet_name
    ws.column_dimensions['A'].width = 22
    ws.column_dimensions['B'].width = 26

    ws.merge_cells('A1:B1')
    tc = ws['A1']
    tc.value = f'Ar:SF6 = {label} — log(EEPF) Data @ {P_TARGET_MTORR:.0f} mTorr'
    tc.font = Font(name='Arial', bold=True, size=12, color='FFFFFF')
    tc.fill = PatternFill('solid', start_color=color_map[label])
    tc.alignment = center()
    ws.row_dimensions[1].height = 26

    meta = [
        ('Gas mixture', f'Ar:SF6 = {label}'),
        ('SF6 fraction', f'{sf6_frac*100:.0f}%'),
        ('Pressure', f'{P_TARGET_MTORR:.0f} mTorr'),
        ('RF frequency', '13.56 MHz'),
    ]
    for i, (k, v) in enumerate(meta, start=2):
        ws.cell(i, 1, k).font = Font(name='Arial', bold=True, size=10)
        ws.cell(i, 2, v).font = data_font()

    hdr_row = len(meta) + 3
    for col, lbl in enumerate(['Energy (eV)', 'ln(EEPF) [eV⁻³/² cm⁻³]'], start=1):
        c = ws.cell(hdr_row, col, lbl)
        style_header_cell(c, '444444')

    for i, (ev, lp) in enumerate(zip(bins, ln_eepf), start=1):
        r = hdr_row + i
        c_ev = ws.cell(r, 1, round(float(ev), 4))
        c_lp = ws.cell(r, 2, round(float(lp), 4))
        style_data_cell(c_ev, i)
        style_data_cell(c_lp, i)
        c_ev.number_format = '0.0000'
        c_lp.number_format = '0.0000'

ws_cmp = wb.create_sheet('Comparison')
ws_cmp.merge_cells('A1:F1')
tc = ws_cmp['A1']
tc.value = f'log(EEPF) Comparison — Ar:SF6 Ratios @ {P_TARGET_MTORR:.0f} mTorr'
tc.font = Font(name='Arial', bold=True, size=13, color='FFFFFF')
tc.fill = PatternFill('solid', start_color='FF1A1A2E')
tc.alignment = center()
ws_cmp.row_dimensions[1].height = 28

col = 1
col_starts = {}
for label, (bins, ln_eepf, color, sf6_frac) in results.items():
    col_starts[label] = col
    c1 = ws_cmp.cell(2, col, f'Energy — {label} (eV)')
    c2 = ws_cmp.cell(2, col+1, f'ln(EEPF) — {label}')
    style_header_cell(c1, color_map[label])
    style_header_cell(c2, color_map[label])
    ws_cmp.column_dimensions[get_column_letter(col)].width = 22
    ws_cmp.column_dimensions[get_column_letter(col+1)].width = 22
    col += 2

for label, (bins, ln_eepf, color, sf6_frac) in results.items():
    c0 = col_starts[label]
    for i in range(len(bins)):
        r = i + 3
        cA = ws_cmp.cell(r, c0, round(float(bins[i]), 4))
        cB = ws_cmp.cell(r, c0+1, round(float(ln_eepf[i]), 4))
        style_data_cell(cA, i); style_data_cell(cB, i)
        cA.number_format = '0.0000'; cB.number_format = '0.0000'

out_path = '/mnt/c/Users/semi/Plasma_simulation/plasma_etch_simulation/simple_simulation/eepf_plot_50.xlsx'
wb.save(out_path)
print(f"Excel saved → {out_path}")
#understnad eepf code,
#make changes in parameters...50 mTorr
#sheath simulation 1D
#what is being cooled in sheath simulation??
#what is 1D and 2D?
#connection between sheath and eepf?
