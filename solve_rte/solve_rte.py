import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.integrate import solve_ivp
from numba import jit
from pdr_functions import *
import os

##### functions
def solve_exact_rte(eng, Aul, Bul, Blu, I0, Rcloud_pc, distance_cm, density_l, density_u, lineprofile=None):

    eng_per_4pi = eng/4/np.pi/u.sr

    Rcloud_cm = (Rcloud_pc.to(u.cm)).value
    dmax = distance_cm[-1]
    barr_cm = np.linspace(0.999*Rcloud_cm, Rcloud_cm - dmax, 50)
    # barr_cm = np.linspace(0.99*Rcloud_cm, Rcloud_cm - dmax, 50)
    print("darr", (Rcloud_cm - barr_cm)*u.cm.to(u.pc))
    print("I0", I0)
    
    # if lineprofile is None:
    #     lineprofile = lambda nu: 1

    intensities = []
    tvals = []
    for b_cm in barr_cm:

        print(f"Solving the LOS with d = {(Rcloud_cm - b_cm)*u.cm.to(u.pc)} pc")

        smid = np.sqrt(Rcloud_cm**2 - b_cm**2)
        
        # Solve ODE
        rte_sol = solve_ivp(
            fun=func_dInuds,
            t_span=[0, smid*2],
            y0=[I0],
            t_eval=np.linspace(0, smid*2, 1000),  
            # t_eval=np.linspace(0, smid*2, 10),  
            method="BDF",
            args=[smid, b_cm, Rcloud_cm, Aul.value, Bul.value, Blu.value, eng_per_4pi.value, distance_cm, density_u, density_l],
            rtol=1e-6,
            atol=1e-30
        )
        if rte_sol.success:
            intensities.append(rte_sol.y)
            tvals.append(rte_sol.t)
            print("Integration succeeded:", rte_sol.message)
        else:
            print("Integration failed:", rte_sol.message)

    return intensities, tvals, barr_cm

@jit(nopython=True)
def interpolate_density(distance_cm, density, x):
    """
    Perform linear interpolation manually. Assumes distance_cm is sorted.
    """
    for i in range(len(distance_cm) - 1):
        if distance_cm[i] <= x <= distance_cm[i + 1]:
            t = (x - distance_cm[i]) / (distance_cm[i + 1] - distance_cm[i])
            return density[i] * (1 - t) + density[i + 1] * t
    raise ValueError("x out of bounds")

@jit(nopython=True)
def los_density_u(s, smid, b_cm, Rcloud_cm, distance_cm, density_u):
    """
    Computes line-of-sight density for the upper state.
    """
    d = Rcloud_cm - np.sqrt((s - smid) ** 2 + b_cm ** 2)
    return interpolate_density(distance_cm, density_u, d)
    # return 2e-7

@jit(nopython=True)
def los_density_l(s, smid, b_cm, Rcloud_cm, distance_cm, density_l):
    """
    Computes line-of-sight density for the lower state.
    """
    d = Rcloud_cm - np.sqrt((s - smid) ** 2 + b_cm ** 2)
    return interpolate_density(distance_cm, density_l, d)
    # return 4e-7

@jit(nopython=True)
def func_dInuds(s, Inu, smid, b_cm, Rcloud_cm, Aul, Bul, Blu, eng_per_4pi, distance_cm, density_u, density_l):

    density_u = los_density_u(s, smid, b_cm, Rcloud_cm, distance_cm, density_u)
    density_l = los_density_l(s, smid, b_cm, Rcloud_cm, distance_cm, density_l)
    
    term1 = Aul * density_u
    term2 = Bul * density_u * Inu
    term3 = -Blu * density_l * Inu

    dInuds = (term1 + term2 + term3) * eng_per_4pi
    
    return dInuds

def get_line_data(linedat, leveldat, bkg_intensity, nl, nu):
    
    gl = next(g for n, g in zip(leveldat["n"], leveldat["g"]) if n == nl)
    gu = next(g for n, g in zip(leveldat["n"], leveldat["g"]) if n == nu)

    eng = energy_K2erg(linedat["E"][np.logical_and(linedat["nu"] == nu, linedat["nl"] == nl)])
    nuul = (eng/const.h).to(u.Hz)
    lambul = (const.c/nuul).to(u.AA)
    Aul = linedat["Aul"][np.logical_and(linedat["nu"] == nu, linedat["nl"] == nl)][0]/u.s
    Bul = (Aul*const.c**2/(2*const.h*nuul**3)).cgs
    Blu = gu*Bul/gl
    
    I0 = ((bkg_intensity[2][np.argmin(np.abs(bkg_intensity[0] - lambul.value))]*u.erg/(u.cm**2*u.s*u.sr*u.AA)*lambul**2/const.c).cgs).value[0]

    return eng, Aul, Bul, Blu, I0

##### loading models

quantities = {
    "AV": 0,
    "Distance": 1,
    "nH": 2,
    "Temperature": 3,
    "n(H)": 4,
    "n(H2)": 5,
    "n(C+)": 6,
    "n(C)": 7,
    "n(CO)": 8,
    "n(13CO)": 9,
    "n(C_18O)": 10,
    "n(O)": 11,
    "n(H2O)": 12,
    "n(H2O J=1,ka=1,kc=0)": 13,
    "n(C+ El=2P,J=3/2)": 14,
    "n(C El=3P,J=1)": 15,
    "n(CO v=0,J=2)": 16,
    "n(CO v=0,J=4)": 17,
    "n(13CO J=2)": 18,
    "n(C_18O J=2)": 19
}

lines = {
    "n(C+ El=2P,J=3/2)" : 0,
    "n(C+ El=2P,J=1/2)" : 1,
    "n(C El=3P,J=0)" : 2,
    "n(C El=3P,J=1)" : 3,
    "n(CO v=0,J=1)" : 4,
    "n(CO v=0,J=2)" : 5,
    "n(CO v=0,J=3)" : 6,
    "n(CO v=0,J=4)" : 7,
    "n(H2O J=1,ka=0,kc=1)" : 8,
    "n(H2O J=1,ka=1,kc=0)" : 9
}

line_dat_path = "path_to_MeudonPDR7/data/Lines/"
level_dat_path = "path_to_MeudonPDR7/data/Levels/"

co_linedat = np.genfromtxt(os.path.join(line_dat_path, "line_co.dat"), skip_header=4, usecols=(0, 1, 2, 3, 4, 11), dtype=[('n', 'i4'), ('nu', 'i4'), ('nl', 'i4'), ('E', 'f8'), ('Aul', 'f8'), ('freq', 'f8')])
co_leveldat = np.genfromtxt(os.path.join(level_dat_path, "level_co.dat"), skip_header=8, usecols=(0, 1, 2, 4, 5), dtype=[('n', 'i4'), ('g', 'i4'),  ('E', 'f8'), ('v', 'i4'),  ('J', 'i4')])

cp_linedat = np.genfromtxt(os.path.join(line_dat_path, "line_cp.dat"), skip_header=4, usecols=(0, 1, 2, 3, 4), dtype=[('n', 'i4'), ('nu', 'i4'), ('nl', 'i4'), ('E', 'f8'), ('Aul', 'f8')])
cp_leveldat = np.genfromtxt(os.path.join(level_dat_path, "level_cp.dat"), skip_header=8, usecols=(0, 1, 2, 4, 5), dtype=[('n', 'i4'), ('g', 'i4'),  ('E', 'f8'), ('v', 'i4'),  ('J', 'i4')])

c_linedat = np.genfromtxt(os.path.join(line_dat_path, "line_c.dat"), skip_header=4, usecols=(0, 1, 2, 3, 4), dtype=[('n', 'i4'), ('nu', 'i4'), ('nl', 'i4'), ('E', 'f8'), ('Aul', 'f8')])
c_leveldat = np.genfromtxt(os.path.join(level_dat_path, "level_c.dat"), skip_header=8, skip_footer=1, usecols=(0, 1, 2), dtype=[('n', 'i4'), ('g', 'i4'),  ('E', 'f8')])

h2o_linedat = np.genfromtxt(os.path.join(line_dat_path, "line_h2o.dat"), skip_header=4, usecols=(0, 1, 2, 3, 4), dtype=[('n', 'i4'), ('nu', 'i4'), ('nl', 'i4'), ('E', 'f8'), ('Aul', 'f8')])
h2o_leveldat = np.genfromtxt(os.path.join(level_dat_path, "level_h2o.dat"), skip_header=8, skip_footer=1, usecols=(0, 1, 2), dtype=[('n', 'i4'), ('g', 'i4'),  ('E', 'f8')])

bkg_intensity = np.genfromtxt("horsehead_thick_p5e6_h2_surfb_IncRadField.dat", skip_header=10, unpack=True)

Rcloud_pc = 5*u.pc

chem_models = {
    "thick_p5e6_h2_surfb": np.genfromtxt("horsehead_thick_p5e6_h2_surfb.dat", skip_header=1, unpack=True)
}

chem_model_lines = np.genfromtxt("horsehead_thick_p5e6_h2_surfb_lines", skip_header=6, unpack=True)

Icp, tcp, bcp = solve_exact_rte(*get_line_data(cp_linedat, cp_leveldat, bkg_intensity, 1, 2), Rcloud_pc, chem_models["thick_p5e6_h2_surfb"][quantities["Distance"]], chem_model_lines[lines["n(C+ El=2P,J=1/2)"]], chem_model_lines[lines["n(C+ El=2P,J=3/2)"]])

Ic, tc, bc = solve_exact_rte(*get_line_data(c_linedat, c_leveldat, bkg_intensity, 1, 2), Rcloud_pc, chem_models["thick_p5e6_h2_surfb"][quantities["Distance"]], chem_model_lines[lines["n(C El=3P,J=0)"]], chem_model_lines[lines["n(C El=3P,J=1)"]])

I12co_2_1, t12co_2_1, b12co_2_1 = solve_exact_rte(*get_line_data(co_linedat, co_leveldat, bkg_intensity, 2, 3), Rcloud_pc, chem_models["thick_p5e6_h2_surfb"][quantities["Distance"]], chem_model_lines[lines["n(CO v=0,J=1)"]], chem_model_lines[lines["n(CO v=0,J=2)"]]) 

I12co_4_3, t12co_4_3, b12co_4_3 = solve_exact_rte(*get_line_data(co_linedat, co_leveldat, bkg_intensity, 4, 5), Rcloud_pc, chem_models["thick_p5e6_h2_surfb"][quantities["Distance"]], chem_model_lines[lines["n(CO v=0,J=3)"]], chem_model_lines[lines["n(CO v=0,J=4)"]]) 

Ih2o, th2o, bh2o = solve_exact_rte(*get_line_data(h2o_linedat, h2o_leveldat, bkg_intensity, 2, 4), Rcloud_pc, chem_models["thick_p5e6_h2_surfb"][quantities["Distance"]], chem_model_lines[lines["n(H2O J=1,ka=0,kc=1)"]], chem_model_lines[lines["n(H2O J=1,ka=1,kc=0)"]])

line_labels = ["N(C+ El=2P,J=3/2)", "N(C El=3P,J=1)", "N(CO v=0,J=2)", "N(CO v=0,J=4)", "N(H2O J=1,ka=1,kc=0)"]
line_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
linenames = ["n(C+ El=2P,J=3/2)", "n(C El=3P,J=1)", "n(CO v=0,J=2)", "n(CO v=0,J=4)", "n(H2O J=1,ka=1,kc=0)"]

Rcloud_cm = (Rcloud_pc.to(u.cm)).value
dmax = chem_models["thick_p5e6_h2_surfb"][quantities["Distance"]][-1]
barr_cm = np.linspace(0.999*Rcloud_cm, Rcloud_cm - dmax, 50)
darr_arcsec = dist_2_arcsec((Rcloud_cm - barr_cm)*u.cm).value

# barr_pc = np.linspace(4.85, 5, 200)*u.pc
# colden_xarr_arcsec = dist_2_arcsec((Rcloud_pc - barr_pc)).value

fig_rte, axs_rte = plt.subplots(1, 3, figsize=(11, 3.5))
ax1_twin = axs_rte[0].twinx()
ax1_twin.set_yticks([])

ax1_twin.plot(np.nan, np.nan, "k-", alpha=0.6, label="RTE")
ax1_twin.plot(np.nan, np.nan, "k--", label="column density")
# ax1_twin.plot(np.nan, np.nan, "k-.", label="convolved")

for iline, (line, sline) in enumerate([(Icp, tcp), (Ic, tc), (I12co_2_1, t12co_2_1), (I12co_4_3, d12co_4_3), (Ih2o, th2o)]):
    intensity = np.array([line[i][0][-1] for i in range(len(line))])
    intensity = intensity/intensity.max()
    if iline < 2:
        axs_rte[0].plot(darr_arcsec, intensity, c=line_colors[iline], lw=3, alpha=0.7, label=line_labels[iline])
        # axs_rte[0].plot(colden_xarr_arcsec, column_densities[iline]/column_densities[iline].max(), "--", lw=2, c=line_colors[iline])
    elif iline < 4:
        axs_rte[1].plot(darr_arcsec, intensity, c=line_colors[iline], lw=3, alpha=0.7, label=line_labels[iline])
        # axs_rte[1].plot(colden_xarr_arcsec, column_densities[iline]/column_densities[iline].max(), "--", lw=2, c=line_colors[iline])
    else:
        axs_rte[2].plot(darr_arcsec[:-1], intensity, c=line_colors[iline], lw=3, alpha=0.7, label=line_labels[iline])
        # axs_rte[2].plot(colden_xarr_arcsec, column_densities[iline]/column_densities[iline].max(), "--", lw=2, c=line_colors[iline])

for ax in axs_rte:
    ax.legend(loc="center left")
    ax.set_xlabel(f'Distance ["]')
    # ax.set_xlim([-2, 80])
    ax.invert_xaxis()
ax1_twin.legend(loc="lower left")
axs_rte[0].set_ylabel("Normalized Intensity")
fig_rte.tight_layout()
# fig_rte.savefig(figure_path+"rte_comparison.pdf", bbox_inches="tight", pad_inches=0)

#### plotting