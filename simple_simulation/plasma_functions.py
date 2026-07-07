import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from constants import *


class plasma_and_sheath_simulation:

    def __init__(self, N_PARTICLES, N_STEPS, dt, E0, OMEGA, T_GAS=300.0):
        """Every experiment-specific knob is decided by whoever creates
        the simulation (i.e. your main/experiment file), not hidden away
        in constants.py. That way each of your 3-4 experiment scripts can
        run this class with completely different particle counts, step
        counts, timesteps, field strengths, or frequencies without
        touching this file at all.

        T_GAS is given a default (room temperature) since you'll often
        leave it fixed, but you can still override it per run.
        """
        self.N_PARTICLES = N_PARTICLES
        self.N_STEPS = N_STEPS
        self.dt = dt
        self.E0 = E0
        self.OMEGA = OMEGA
        self.T_GAS = T_GAS

    # -----------------------------------------------------------
    # Gas properties
    # -----------------------------------------------------------
    def gas_density(self, pressure_mtorr=50, T=None):
        """n = P / (kB T). Converts mTorr -> Pa first (1 Torr = 133.322 Pa)."""
        if T is None:
            T = self.T_GAS
        P_pa = pressure_mtorr * 1e-3 * 133.322
        return P_pa / (k_B * T)   # particles / m^3

    # -----------------------------------------------------------
    # Argon cross sections
    # (these don't depend on instance state, so they're static methods --
    #  you can call them either as self.sigma_ar_elastic(E) or
    #  plasma_and_sheath_simulation.sigma_ar_elastic(E))
    # -----------------------------------------------------------
    @staticmethod
    def sigma_ar_elastic(E_eV):
        """Momentum-transfer cross section with Ramsauer-Townsend dip (m^2).
        Qualitative analytic form: dips near AR_RT_MINIMUM_E, rises at very low
        and at higher energy, broadly consistent with known Ar elastic CS shape."""
        E = np.maximum(E_eV, 1e-3)
        rt_dip = 1.0 - 0.9 * np.exp(-((np.log(E / AR_RT_MINIMUM_E))**2) / (2 * 0.6**2))
        base = 6.0e-20 / (1.0 + 0.5 * E)**0.5
        return base * rt_dip + 0.3e-20

    @staticmethod
    def sigma_ar_excitation(E_eV):
        """Lumped excitation cross section, threshold-gated, peaks then falls (m^2)."""
        E = np.asarray(E_eV, dtype=float)
        sigma = np.zeros_like(E)
        above = E > AR_EXC_THRESHOLD
        x = E[above] - AR_EXC_THRESHOLD
        sigma[above] = 4.0e-21 * (x / (1 + 0.15 * x**1.5)) * np.exp(-x / 25.0)
        return sigma

    @staticmethod
    def sigma_ar_ionization(E_eV):
        """Ionization cross section, real threshold, Bethe-like rise + slow falloff (m^2)."""
        E = np.asarray(E_eV, dtype=float)
        sigma = np.zeros_like(E)
        above = E > AR_ION_THRESHOLD
        x = E[above] - AR_ION_THRESHOLD
        sigma[above] = 3.5e-21 * (x / (1 + 0.05 * x)) * np.exp(-x / 80.0)
        return sigma

    # -----------------------------------------------------------
    # SF6 cross sections
    #
    #   - Dissociative attachment peaks very close to 0 eV (real -- SF6 is
    #     famous for this near-thermal attachment resonance) and falls off
    #     within ~1 eV
    #   - Vibrational/inelastic excitation onset ~0.1-0.2 eV (real, low
    #     threshold)
    #   - Electron impact ionization threshold ~15.3 eV (real, close to
    #     SF6's IE)
    # -----------------------------------------------------------
    @staticmethod
    def sigma_sf6_attachment(E_eV):
        """Dissociative attachment: sharply peaked near 0 eV (m^2).
        SF6's near-thermal attachment resonance is a well-documented real
        effect; this is a smooth analytic peak approximating its
        qualitative shape."""
        E = np.maximum(E_eV, 1e-4)
        return 4.0e-19 * np.exp(-((E - SF6_ATTACH_PEAK_E) / 0.12)**2)

    @staticmethod
    def sigma_sf6_vibrational(E_eV):
        """Vibrational/rotational + electronic inelastic excitation, lumped (m^2).
        Real SF6 has strong, broad inelastic losses (vibrational ~0.1-1 eV onset,
        electronic excitation extending to several eV) that are the dominant
        energy-loss mechanism responsible for its well-known strong electron
        cooling/attaching behavior in plasma etch chemistry. A single
        exp(-x/1.5) decay would make this channel vanish by ~2 eV, which is
        NOT representative of real SF6 -- so a broader lumped inelastic loss
        term is used here that stays significant out to ~15 eV, consistent
        with SF6 being a strongly electronegative/attaching and strongly
        inelastic (cooling) gas across the whole low/mid energy range."""
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

    @staticmethod
    def sigma_sf6_ionization(E_eV):
        """SF6 ionization, real threshold (m^2)."""
        E = np.asarray(E_eV, dtype=float)
        sigma = np.zeros_like(E)
        above = E > SF6_ION_THRESHOLD
        x = E[above] - SF6_ION_THRESHOLD
        sigma[above] = 2.5e-21 * (x / (1 + 0.05 * x)) * np.exp(-x / 70.0)
        return sigma

    @staticmethod
    def sigma_sf6_elastic(E_eV):
        """SF6 elastic/momentum-transfer background (m^2)."""
        E = np.maximum(E_eV, 1e-3)
        return 5.0e-20 / (1.0 + 0.3 * E)**0.6

    def total_cross_sections(self, E_eV, sf6_fraction):
        """Return dict of process -> sigma(E) array (m^2), for the gas mixture."""
        procs = {}
        if sf6_fraction < 1.0:
            procs['ar_elastic']    = (1 - sf6_fraction) * self.sigma_ar_elastic(E_eV)
            procs['ar_excitation'] = (1 - sf6_fraction) * self.sigma_ar_excitation(E_eV)
            procs['ar_ionization'] = (1 - sf6_fraction) * self.sigma_ar_ionization(E_eV)
        if sf6_fraction > 0.0:
            procs['sf6_elastic']     = sf6_fraction * self.sigma_sf6_elastic(E_eV)
            procs['sf6_vibrational'] = sf6_fraction * self.sigma_sf6_vibrational(E_eV)
            procs['sf6_attachment']  = sf6_fraction * self.sigma_sf6_attachment(E_eV)
            procs['sf6_ionization']  = sf6_fraction * self.sigma_sf6_ionization(E_eV)
        return procs

    @staticmethod
    def isotropic_scatter(n):
        phi = 2 * np.pi * np.random.rand(n)
        cos_theta = 1.0 - 2.0 * np.random.rand(n)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
        return sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta

    # -----------------------------------------------------------
    # Main MCC run
    # -----------------------------------------------------------
    def run_mcc_simulation(self, sf6_fraction, pressure_mtorr):
        print(f"Running Ar/SF6 fraction={sf6_fraction:.2f} @ {pressure_mtorr:.0f} mTorr "
              f"(N={self.N_PARTICLES}, steps={self.N_STEPS})...")
        n_gas = self.gas_density(pressure_mtorr)

        vx = np.random.normal(0, 1e5, self.N_PARTICLES)
        vy = np.random.normal(0, 1e5, self.N_PARTICLES)
        vz = np.random.normal(0, 1e5, self.N_PARTICLES)
        active = np.ones(self.N_PARTICLES, dtype=bool)

        time_avg_energies = []
        start_avg = int(0.75 * self.N_STEPS)

        for step in range(self.N_STEPS):
            if not np.any(active):
                break
            t = step * self.dt
            E_t = self.E0 * np.cos(self.OMEGA * t)

            vx[active] += (-e_charge * E_t / m_e) * self.dt

            v_sq = vx[active]**2 + vy[active]**2 + vz[active]**2
            v_mag = np.sqrt(v_sq)
            E_ev = 0.5 * m_e * v_sq / e_charge

            procs = self.total_cross_sections(E_ev, sf6_fraction)
            names = list(procs.keys())
            sigmas = np.stack([procs[k] for k in names], axis=0)   # (n_proc, n_active)

            # Collision probability per process: P_k = 1 - exp(-n*sigma_k*v*dt)
            nu_k = n_gas * sigmas * v_mag[None, :]                  # collision freq per process
            nu_tot = nu_k.sum(axis=0)
            p_coll_tot = 1.0 - np.exp(-nu_tot * self.dt)
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
                        dx, dy, dz = self.isotropic_scatter(len(global_idx))
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
                        dx, dy, dz = self.isotropic_scatter(len(global_idx))
                        vx[global_idx] = v_new * dx
                        vy[global_idx] = v_new * dy
                        vz[global_idx] = v_new * dz

                    elif pname == 'ar_excitation':
                        new_E = np.maximum(0.005, E_sel - EXC_LOSS_EV)
                        v_new = np.sqrt(2 * new_E * e_charge / m_e)
                        dx, dy, dz = self.isotropic_scatter(len(global_idx))
                        vx[global_idx] = v_new * dx
                        vy[global_idx] = v_new * dy
                        vz[global_idx] = v_new * dz

                    elif pname == 'ar_ionization':
                        # Simplified: scattered electron keeps half remaining energy,
                        # the ejected electron is not separately tracked (no electron
                        # multiplication modeled here -- approximation for 0D EEPF shape)
                        new_E = np.maximum(0.005, (E_sel - ION_LOSS_EV_AR) * 0.5)
                        v_new = np.sqrt(2 * new_E * e_charge / m_e)
                        dx, dy, dz = self.isotropic_scatter(len(global_idx))
                        vx[global_idx] = v_new * dx
                        vy[global_idx] = v_new * dy
                        vz[global_idx] = v_new * dz

                    elif pname == 'sf6_ionization':
                        new_E = np.maximum(0.005, (E_sel - ION_LOSS_EV_SF6) * 0.5)
                        v_new = np.sqrt(2 * new_E * e_charge / m_e)
                        dx, dy, dz = self.isotropic_scatter(len(global_idx))
                        vx[global_idx] = v_new * dx
                        vy[global_idx] = v_new * dy
                        vz[global_idx] = v_new * dz

                    elif pname == 'sf6_attachment':
                        active[global_idx] = False   # electron removed

            if step > start_avg and step % 4 == 0:
                surv_vsq = vx[active]**2 + vy[active]**2 + vz[active]**2
                time_avg_energies.extend((0.5 * m_e * surv_vsq) / e_charge)

        return np.array(time_avg_energies)

    @staticmethod
    def get_eepf(energies, bins=60, max_energy=22):
        if len(energies) == 0:
            return np.array([]), np.array([])
        eedf, edges = np.histogram(energies, bins=bins, range=(0.05, max_energy), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        valid = eedf > 0
        eepf = eedf[valid] / np.sqrt(centers[valid])
        return centers[valid], eepf
    


class mean_ev_vs_Z_simulation:
    def __init__(self, gap_length, n_particles, dt, steps, field_accel, sheath_margin, e_reflect_ev):
        """Initializes the simulation environment and particles."""
        self.gap_length = gap_length
        self.n_particles = n_particles
        self.dt = dt
        self.steps = steps
        self.field_accel = field_accel
        self.sheath_margin = sheath_margin
        self.e_reflect_ev = e_reflect_ev

        # Thermal velocity corresponding to ~10 eV bulk
        self.v_initial_thermal = np.sqrt(2 * 10.0 * Q_E / M_E)
        
        # Initialize particle arrays
        self.pos = np.random.uniform(0, self.gap_length, self.n_particles)
        self.vx = np.random.normal(0, self.v_initial_thermal, self.n_particles)
        self.vy = np.random.normal(0, self.v_initial_thermal, self.n_particles)
        self.vz = np.random.normal(0, self.v_initial_thermal, self.n_particles)

    def run(self):
        """Executes the main PIC kinematic loop over the specified time steps."""
        print("Running spatial EEDF simulation (T ~ 10 eV bulk)...")
        for step in range(self.steps):
            # 1. Update velocities and positions
            self.vx += self.field_accel * self.dt
            self.pos += self.vx * self.dt

            # 2. Handle wall reflections
            left_hit = self.pos <= 0
            right_hit = self.pos >= self.gap_length
            
            self.pos = np.where(left_hit, -self.pos, self.pos)
            self.vx = np.where(left_hit, -self.vx, self.vx)
            
            self.pos = np.where(right_hit, 2 * self.gap_length - self.pos, self.pos)
            self.vx = np.where(right_hit, -self.vx, self.vx)

            # 3. Calculate Kinetic Energy and handle sheath absorption
            KE = 0.5 * M_E * (self.vx**2 + self.vy**2 + self.vz**2) / Q_E
            near_wall = (self.pos < self.sheath_margin) | (self.pos > self.gap_length - self.sheath_margin)
            absorbed = near_wall & (KE > self.e_reflect_ev)
            active = ~absorbed
            
            # Keep only active particles
            self.pos = self.pos[active]
            self.vx = self.vx[active]
            self.vy = self.vy[active]
            self.vz = self.vz[active]

            # 4. Replenish absorbed particles to maintain constant N_PARTICLES
            n_new = self.n_particles - len(self.pos)
            if n_new > 0:
                new_p = np.random.normal(self.gap_length / 2, 0.3, n_new)
                new_vx = np.random.normal(0, self.v_initial_thermal * 0.4, n_new)
                new_vy = np.random.normal(0, self.v_initial_thermal * 0.4, n_new)
                new_vz = np.random.normal(0, self.v_initial_thermal * 0.4, n_new)

                self.pos = np.concatenate([self.pos, new_p])
                self.vx = np.concatenate([self.vx, new_vx])
                self.vy = np.concatenate([self.vy, new_vy])
                self.vz = np.concatenate([self.vz, new_vz])

    def local_mean_energy(self, z_target, slice_width=0.1):
        """Mean kinetic energy (eV) of particles sitting near position z_target."""
        mask = (self.pos > (z_target - slice_width / 2)) & (self.pos < (z_target + slice_width / 2))
        lvx, lvy, lvz = self.vx[mask], self.vy[mask], self.vz[mask]
        
        if len(lvx) == 0:
            return np.nan
            
        speed2 = lvx**2 + lvy**2 + lvz**2
        energies = 0.5 * M_E * speed2 / Q_E
        return energies.mean()

    def get_mean_energy_profile(self, z_values, slice_width=0.1):
        """Generates the mean energy array across a sweep of Z locations."""
        return np.array([self.local_mean_energy(z, slice_width) for z in z_values])