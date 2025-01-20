#!/usr/bin/env python3
# encoding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d
from scipy.integrate import quad, solve_ivp
from numba import jit

def create_dict(filename):
    """create the dictionary for refering quantities from the file used to extract data by IDAT

    Args:
        filename (string): the name of the file used to extract data 

    Returns:
        qty_dict: a dictionary containing the quantities and their corresponding index
    """

    with open(filename, "r") as file:
        qty_dict = {line.strip(): i for i, line in enumerate(file)}
    return qty_dict

# quantities = {
#     "AV": 0,
#     "Distance": 1,
#     "nH": 2,
#     "Temperature": 3,
#     "n(H)": 4,
#     "n(H2)": 5,
#     "n(C+)": 6,
#     "n(C)": 7,
#     "n(CO)": 8,
#     "n(13CO)": 9,
#     "n(C_18O)": 10,
#     "n(O)": 11,
#     "n(H2O)": 12,
#     "n(H2O J=1,ka=1,kc=0)": 13,
#     "n(C+ El=2P,J=3/2)": 14,
#     "n(C El=3P,J=1)": 15,
#     "n(CO v=0,J=2)": 16,
#     "n(CO v=0,J=4)": 17,
#     "n(13CO J=2)": 18,
#     "n(C_18O J=2)": 19
# }

# ##### observation data #####
# datafolder = "Darek_data/"

# d12co_2_1, l12co_2_1 = np.loadtxt(datafolder+"horsehead-12co2-1-smo.area", unpack=True, dtype=float)
# indx_12co_2_1 = np.argwhere(l12co_2_1 > 0)
# # norm_12co_2_1 = 41.9
# norm_12co_2_1 = np.max(l12co_2_1[np.logical_and(l12co_2_1 > 0, d12co_2_1 < 250)])

# d12co_4_3, l12co_4_3 = np.loadtxt(datafolder+"horsehead-12co4-3-smo.area", unpack=True, dtype=float)
# indx_12co_4_3 = np.argwhere(l12co_4_3 > 0)
# norm_12co_4_3 = 31.64
# norm_12co_4_3 = np.max(l12co_4_3[np.logical_and(l12co_4_3 > 0, d12co_4_3 < 250)])

# d13co_2_1, l13co_2_1 = np.loadtxt(datafolder+"horsehead-13co2-1-smo.area", unpack=True, dtype=float)
# indx_13co_2_1 = np.argwhere(l13co_2_1 > 0)
# norm_13co_2_1 = 17.2
# norm_13co_2_1 = np.max(l13co_2_1[np.logical_and(l13co_2_1 > 0, d13co_2_1 < 250)])

# dc18o_2_1, lc18o_2_1 = np.loadtxt(datafolder+"horsehead-c18o2-1-smo.area", unpack=True, dtype=float)
# indx_c18o_2_1 = np.argwhere(lc18o_2_1 > 0)
# norm_c18o_2_1 = 6.54
# norm_c18o_2_1 = np.max(lc18o_2_1[np.logical_and(lc18o_2_1 > 0, dc18o_2_1 < 250)])

# dc2, lc2 = np.loadtxt(datafolder+"horsehead-cii-smo.area", unpack=True, dtype=float)
# # norm_c2 = 28.1
# norm_c2 = np.max(lc2[np.logical_and(dc2 > 0, dc2 < 50)])
# indx_c2 = np.logical_and(lc2 > 0, dc2 < 250)

# dc1, lc1 = np.loadtxt(datafolder+"horsehead-ci1-0-smo.area", unpack=True, dtype=float)
# norm_c1 = 9
# norm_c1 = np.max(lc1[np.logical_and(dc1 > 0, dc1 < 50)])
# indx_c1 = np.logical_and(lc1 > 0, dc1 < 250)

# dh2o, lh2o = np.loadtxt(datafolder+"horsehead-h2o.area", unpack=True, dtype=float)
# indx_h2o = np.argwhere(lh2o > 0)
# norm_h2o = 0.26
# norm_h2o = np.max(lh2o[np.logical_and(dh2o > 0, dh2o < 50)])

# # find the distances between peaks
# peak_12co_2_1 = d12co_2_1[indx_12co_2_1][np.argmin(np.abs(l12co_2_1[indx_12co_2_1]/norm_12co_2_1-1.0))]
# peak_13co_2_1 = d13co_2_1[indx_13co_2_1][np.argmin(np.abs(l13co_2_1[indx_13co_2_1]/norm_13co_2_1-1.0))]
# peak_c18o_2_1 = dc18o_2_1[indx_c18o_2_1][np.argmin(np.abs(lc18o_2_1[indx_c18o_2_1]/norm_c18o_2_1-1.0))]
# peak_12co_4_3 = d12co_4_3[indx_12co_4_3][np.argmin(np.abs(l12co_4_3[indx_12co_4_3]/norm_12co_4_3-1.0))]
# peak_c2 = dc2[indx_c2][np.argmin(np.abs(lc2[indx_c2]/norm_c2-1.0))]
# peak_c1 = dc1[indx_c1][np.argmin(np.abs(lc1[indx_c1]/norm_c1-1.0))]
# peak_h2o = dh2o[indx_h2o][np.argmin(np.abs(lh2o[indx_h2o]/norm_h2o-1.0))]

def dist_2_arcsec(dists, dist2obj=400*u.pc):
    """convert given distance array to arcsec

    Args:
        dists (float ndarray with unit): distance array to be converted
        dist2obj (float with unit, optional): distance to the object. Defaults to 400*u.pc (the distance to the horsehead nebula).

    Returns:
        _type_: distance array as in arcsec
    """
    return (dists.to(u.pc)/dist2obj*u.rad).to(u.arcsec)

def arcsec_2_dist(arcsecs, distunit="cm", dist2obj=400*u.pc):
    """convert given distance array in arcsec to physical distance

    Args:
        arcsecs (float ndarray): distance array in arcsec to be converted
        distunit (str, optional): physical distance unit to be converted into. Defaults to "cm".
        dist2obj (float with unit, optional): distance to the object. Defaults to 400*u.pc (the distance to the horsehead nebula).

    Returns:
        _type_: _description_
    """
    if distunit == "cm":
        return (arcsecs.to(u.rad)).value*dist2obj.to(u.cm)
    elif distunit == "pc":
        return (arcsecs.to(u.rad)).value*dist2obj.to(u.pc)

def plot_structure(model, quantities, xvar="AV", densities=None, xlim=None, ylim=None, title=None, titleloc=None, legendloc="best", normalize=False, colors=None, savefolder="./", savename="abundances",save=False):
    '''
    plot the number densities of the given quantities
    '''
    if not densities:
        densities = ["n(H)", "n(H2)", "n(C+)", "n(C)", "n(CO)", "n(O)", "n(H2O)"]
    # if colors is None: colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    if xvar == "AV": 
        xx = model[quantities["AV"]]
        for ivar, yvar in enumerate(densities):  
            index = quantities[yvar]
            yy = model[index]/model[quantities["nH"]] if normalize else model[index]
            ax.loglog(xx, yy,  label=yvar)
        if xlim is None: xlim = [1e-4, 45]     
        ax.set_xlim(xlim)
        ax.set_xlabel(r"$A_V$")
    elif xvar == "arcsec":
        xx = (model[quantities["Distance"]]*u.cm).to(u.pc)/(400*u.pc)*u.rad.to(u.arcsec)
        for ivar, yvar in enumerate(densities):  
            index = quantities[yvar]
            yy = model[index]/model[quantities["nH"]] if normalize else model[index]
            ax.semilogy(xx, yy,  label=yvar)
        ax.set_xlabel(r'Distance ["]')
    elif xvar == "pc":
        xx = (model[quantities["Distance"]]*u.cm).to(u.pc)
        for ivar, yvar in enumerate(densities):  
            index = quantities[yvar]
            yy = model[index]/model[quantities["nH"]] if normalize else model[index]
            ax.semilogy(xx, yy,  label=yvar)
        ax.set_xlabel(r'Distance [pc]')
    if normalize: 
        ax.set_ylabel("n(X)/nH")
    else:
        ax.set_ylabel(r"n(X) [$\mathrm{cm}^{-3}$]")
    ax.set(ylim=ylim)
    
    ax.invert_xaxis()
    ax.legend(loc=legendloc)
    # if title: ax.text(0.05, 0.02, title, transform=ax.transAxes)
    if not titleloc: titleloc = (0.05, 0.02)
    if title: ax.text(titleloc[0], titleloc[1], title, transform=ax.transAxes)
    if save: fig.savefig(savefolder+savename+".pdf", bbox_inches="tight", pad_inches=0)

    return fig, ax

### use different markers for different models
# def plot_densities(models, model_params, quantities, yvars=["n(H)"], xvar="AV", xlim=None, legendnames=None, legendloc="best", linestyles=None, colors=None, marks=None, normalize=False, savefolder="./", savename="densities",save=False):
#     '''
#     plot the number densities of the given quantities
#     '''

#     if linestyles is None: linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]
#     if colors is None: colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
#     if marks is None: marks = [".", "+", "x", "1", "2", "3", "4", "v", "<", ">"]

#     fig, ax = plt.subplots(1, 1)
#     if xvar == "AV": 
#         for ivar, yvar in enumerate(yvars): 
#             index = quantities[yvar]
#             for iparam, param in enumerate(model_params):
#                 xx = models[param][quantities["AV"]]
#                 yy = models[param][index]/models[param][quantities["nH"]] if normalize else models[param][index]
#                 # ax.loglog(xx, yy, c=colors[ivar], marker=marks[iparam], markevery=0.1)
#                 ax.loglog(xx, yy, c=colors[ivar], linestyle=linestyles[iparam])
#             ax.loglog(np.nan, np.nan, label=yvar, c=colors[ivar], linestyle=linestyles[iparam])
#         if xlim is None: xlim = [1e-4, 42]
#         ax.set_xlim(xlim)
#         ax.set_xlabel(r"$A_V$")
#     else:
#         for ivar, yvar in enumerate(yvars):  
#             index = quantities[yvar]
#             for iparam, param in enumerate(model_params):
#                 xx = (models[param][quantities["Distance"]]*u.cm).to(u.pc)/(400*u.pc)*u.rad.to(u.arcsec)
#                 yy = models[param][index]/models[param][quantities["nH"]] if normalize else models[param][index]
#                 # ax.semilogy(xx, yy, c=colors[ivar], marker=marks[iparam], markevery=0.1)
#                 ax.semilogy(xx, yy, c=colors[ivar], linestyle=linestyles[iparam])
#             ax.semilogy(np.nan, np.nan, label=yvar, c=colors[ivar], linestyle=linestyles[iparam])
#         ax.set_xlabel(r'Distance ["]')
#     if normalize: 
#         ax.set_ylabel("n(X)/nH")
#     else:
#         ax.set_ylabel("n(X)")

#     for iparam, param in enumerate(model_params):
#         label = legendnames[iparam] if legendnames else param
#         ax.plot(np.nan, np.nan, c="k", label=label, linestyle=linestyles[iparam])
    
#     ax.invert_xaxis()
#     ax.legend(loc=legendloc)
#     if save: fig.savefig(savefolder+savename+".pdf", bbox_inches="tight", pad_inches=0)

#     return fig, ax

### use different colors for different models
def plot_densities(
    models, model_params, quantities, 
    yvars=["n(H)"], xvar="AV", xlim=None, normalize=False,
    modelnames=None, colors=None, ls=None, color_by="model", colorlgdloc="best", lslgdloc="lower left",
    varls=None, varcolors=None, 
    savefolder="./", savename="densities", save=False
):

    if ls is None: ls = ["-", "--", "-.", (0, (3, 1, 1, 1, 1, 1)), ":"]
    
    if color_by == "model":
        if colors is None: 
            colors = plt.get_cmap("tab10").colors
        mdlcolors = {param: color for param, color in zip(model_params, colors)}
        varls = {yvar: ls for yvar, ls in zip(yvars, ls)}
    elif color_by == "yvar":
        if colors is None: 
            colors = plt.get_cmap("Set2").colors
        varcolors = {yvar: color for yvar, color in zip(yvars, colors)}
        mdlls = {param: ls for param, ls in zip(model_params, ls)}
    else:
        raise ValueError("Please choose from model or yvars for color_by")

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    axls = ax.twinx()

    if xvar == "AV": 
        for ivar, yvar in enumerate(yvars): 
            index = quantities[yvar]
            for iparam, param in enumerate(model_params):
                xx = models[param][quantities["AV"]]
                yy = models[param][index] / models[param][quantities["nH"]] if normalize else models[param][index]
                if color_by == "model":
                    color = mdlcolors[param]
                    linestyle = varls[yvar]
                else:
                    color = varcolors[yvar]
                    linestyle = mdlls[param]
                ax.loglog(
                    xx, yy, 
                    c=color, 
                    linestyle=linestyle
                )
            if color_by == "model":
                axls.semilogy(np.nan, np.nan, "k", linestyle=linestyle, label=yvar)
            else:
                ax.semilogy(np.nan, np.nan, c=varcolors[yvar], label=yvar)
        if xlim is None: 
            xlim = [1e-4, 42]
        ax.set_xlim(xlim)
        ax.set_xlabel(r"$A_V$")
    else:
        for ivar, yvar in enumerate(yvars):  
            index = quantities[yvar]
            for iparam, param in enumerate(model_params):
                if xvar == "arcsec":
                    xx = (models[param][quantities["Distance"]] * u.cm).to(u.pc) / (400 * u.pc) * u.rad.to(u.arcsec)
                elif xvar == "cm":
                    xx = models[param][quantities["Distance"]] * u.cm
                elif xvar == "pc":
                    xx = (models[param][quantities["Distance"]] * u.cm).to(u.pc)
                else:
                    raise ValueError("Please choose from arcsec, cm, pc")
                yy = models[param][index] / models[param][quantities["nH"]] if normalize else models[param][index]
                if color_by == "model":
                    color = mdlcolors[param]
                    linestyle = varls[yvar]
                else:
                    color = varcolors[yvar]
                    linestyle = mdlls[param]
                ax.semilogy(
                    xx, yy, 
                    c=color, 
                    linestyle=linestyle
                )
            if color_by == "model":
                axls.semilogy(np.nan, np.nan, "k", linestyle=linestyle, label=yvar)
            else:
                ax.semilogy(np.nan, np.nan, c=varcolors[yvar], label=yvar)
                
        if xvar == "arcsec":
            ax.set_xlabel(r'Distance ["]')
        else:
            ax.set_xlabel(f"Distance [{xvar}]") 

    if normalize: 
        ax.set_ylabel("n(X)/nH")
    else:
        ax.set_ylabel(r"n(X) [$\mathrm{cm}^{-3}$]")

    labels = modelnames if modelnames else model_params
    if color_by == "model":
        for iparam, param in enumerate(model_params):
            ax.plot(np.nan, np.nan, c=mdlcolors[param], label=labels[iparam])
    elif color_by == "yvar":
        for iparam, param in enumerate(model_params):
            axls.plot(np.nan, np.nan, "k", linestyle=mdlls[param],  label=labels[iparam])
    
    ax.invert_xaxis()
    ax.legend(loc=colorlgdloc)
    axls.legend(loc=lslgdloc)
    axls.set_yticks([])
    
    if save: 
        fig.savefig(f"{savefolder}/{savename}.pdf", bbox_inches="tight", pad_inches=0)

    return fig, [ax, axls]


# def plot_lines(models, model_params, quantities, yvars=["n(H)"], xvar="cm", xlim=None, maxindex=500, peak=True, legendnames=None, legendloc="best", linestyles=None, colors=None, marks=None, normalize=False, savefolder="./", savename="emissivity",save=False):
#     '''
#     plot the local emissivity of the given quantities
#     '''

#     if linestyles is None: linestyles = ["-", "--", "-.",  (0, (3, 1, 1, 1, 1, 1)), ":"]
#     if colors is None: colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
#     if marks is None: marks = [".", "1", "+", "x", "v", "<", ">"]

#     fig, ax = plt.subplots(1, 1)

#     if xvar == "AV": 
#         for ivar, yvar in enumerate(yvars): 
#             index = quantities[yvar]
#             if peak: print("")
#             for iparam, param in enumerate(model_params):
#                 xx = models[param][quantities["AV"]]
#                 yy = models[param][index]
#                 if peak: 
#                     peak = np.argmax(yy[:maxindex])
#                     print(f"model {param}: {yvar} peaks at {xx[peak]} Av")
#                 if normalize:
#                     norm = yy[:maxindex].max()
#                     yy = yy/norm
#                 ax.semilogx(xx[:maxindex], yy[:maxindex], c=colors[ivar], marker=marks[iparam], markevery=0.1)
#             ax.plot(np.nan, np.nan, label=yvar, c=colors[ivar])
#         if xlim is None: xlim = [1e-4, 30]
#         ax.set_xlim(xlim)
#         ax.set_xlabel(r"$A_V$")
#     else:
#         for ivar, yvar in enumerate(yvars):  
#             index = quantities[yvar]
#             if peak: print("")
#             for iparam, param in enumerate(model_params):
#                 if xvar == "arcsec":
#                     xx = (models[param][quantities["Distance"]]*u.cm).to(u.pc)/(400*u.pc)*u.rad.to(u.arcsec)
#                 elif xvar == "cm":
#                     xx = models[param][quantities["Distance"]]*u.cm
#                 elif xvar == "pc":
#                     xx = (models[param][quantities["Distance"]]*u.cm).to(u.pc)
#                 else:
#                     raise ValueError("Please choose from arcsec, cm, pc")
#                 yy = models[param][index]
#                 if peak: 
#                     peak = np.argmax(yy[:maxindex])
#                     print(f"model {param}: {yvar} peaks at {xx[peak]:.4f} " + xvar)
#                 if normalize:
#                     norm = yy[:maxindex].max()
#                     yy = yy/norm
#                 ax.plot(xx[:maxindex], yy[:maxindex], c=colors[ivar], marker=marks[iparam], markevery=0.1)
#             ax.plot(np.nan, np.nan, label=yvar, c=colors[ivar])
#         if xvar == "arcsec":
#             ax.set_xlabel(r'Distance ["]')
#         else:
#             ax.set_xlabel(f"Distance [{xvar}]") 
#     if normalize: 
#         ax.set_ylabel(f"n(X)/{norm:.2f}")
#     else:
#         ax.set_ylabel("n(X)")

#     for iparam, param in enumerate(model_params):
#         ax.plot(np.nan, np.nan, c="k", label=param, marker=marks[iparam])
    
#     ax.invert_xaxis()
#     ax.legend(loc=legendloc)
#     if save: fig.savefig(savefolder+savename+".pdf", bbox_inches="tight", pad_inches=0)

#     return fig, ax

def plot_lines(models, model_params, quantities, yvars=["n(H)"], xvar="cm", xlim=None, maxindex=500, peak=True, legendnames=None, legendloc="best", linestyles=None, colors=None, marks=None, normalize=False, savefolder="./", savename="emissivity",save=False):
    '''
    plot the local emissivity of the given quantities
    '''

    if linestyles is None: linestyles = ["-", "--", "-.",  (0, (3, 1, 1, 1, 1, 1)), ":"]
    if colors is None: colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    if marks is None: marks = [".", "1", "+", "x", "v", "<", ">"]

    fig, ax = plt.subplots(1, 1)

    if xvar == "AV": 
        for ivar, yvar in enumerate(yvars): 
            index = quantities[yvar]
            if peak: print("")
            for iparam, param in enumerate(model_params):
                xx = models[param][quantities["AV"]]
                yy = models[param][index]
                if peak: 
                    peak = np.argmax(yy[:maxindex])
                    print(f"model {param}: {yvar} peaks at {xx[peak]} Av")
                if normalize:
                    norm = yy[:maxindex].max()
                    yy = yy/norm
                ax.semilogx(xx[:maxindex], yy[:maxindex], c=colors[iparam], linestyle=linestyles[ivar])
            ax.plot(np.nan, np.nan, "k", linestyle=linestyles[ivar], label=yvar)
        if xlim is None: xlim = [1e-4, 30]
        ax.set_xlim(xlim)
        ax.set_xlabel(r"$A_V$")
    else:
        for ivar, yvar in enumerate(yvars):  
            index = quantities[yvar]
            if peak: print("")
            for iparam, param in enumerate(model_params):
                if xvar == "arcsec":
                    xx = (models[param][quantities["Distance"]]*u.cm).to(u.pc)/(400*u.pc)*u.rad.to(u.arcsec)
                elif xvar == "cm":
                    xx = models[param][quantities["Distance"]]*u.cm
                elif xvar == "pc":
                    xx = (models[param][quantities["Distance"]]*u.cm).to(u.pc)
                else:
                    raise ValueError("Please choose from arcsec, cm, pc")
                yy = models[param][index]
                if peak: 
                    peak = np.argmax(yy[:maxindex])
                    print(f"model {param}: {yvar} peaks at {xx[peak]:.4f} " + xvar)
                if normalize:
                    norm = yy[:maxindex].max()
                    yy = yy/norm
                ax.plot(xx[:maxindex], yy[:maxindex], c=colors[iparam], linestyle=linestyles[ivar])
            ax.plot(np.nan, np.nan, "k", linestyle=linestyles[ivar], label=yvar)
        if xvar == "arcsec":
            ax.set_xlabel(r'Distance ["]')
        else:
            ax.set_xlabel(f"Distance [{xvar}]") 
    if normalize: 
        ax.set_ylabel(f"n(X)/{norm:.2f}")
    else:
        ax.set_ylabel("n(X)")

    for iparam, param in enumerate(model_params):
        label = legendnames[iparam] if legendnames else param
        ax.plot(np.nan, np.nan, c=colors[iparam], label=label)
    
    ax.invert_xaxis()
    ax.legend(loc=legendloc)
    if save: fig.savefig(savefolder+savename+".pdf", bbox_inches="tight", pad_inches=0)

    return fig, ax

# def convolve_uniform(xarr, data, resolution, gridsize=int(1e4), mapback=False, expandx=True):

#     if np.all(np.diff(xarr) >= 0):
#         increase = True
#     elif np.all(np.diff(xarr) <= 0):
#         increase = False
#         xarr = xarr[::-1]
#         data = data[::-1]
#     else:
#         raise ValueError("The xarr must be sorted")

#     x_uniform = np.linspace(xarr[0], xarr[-1], gridsize)
#     dx_uniform = x_uniform[1] - x_uniform[0]
#     x_arr_unique, x_arr_unique_indices = np.unique(xarr, return_index=True)
#     finterp = interp1d(x_arr_unique, data[x_arr_unique_indices])
#     y_uniform = finterp(x_uniform)
 
#     sigma = resolution/(2*np.sqrt(2*np.log(2)))
#     filterfunc = lambda x: 1/np.sqrt(2*np.pi)/sigma*np.exp(-x**2/2/sigma**2)
#     norm = np.sum(filterfunc(np.arange(-3*sigma, 3*sigma, dx_uniform)))

#     if expandx: 
#         # truncate after 3 sigma
#         x_expanded = np.hstack((np.arange(xarr[0] - 3*sigma, xarr[0], dx_uniform), x_uniform, np.arange(xarr[-1] + dx_uniform, xarr[-1] + 3*sigma, dx_uniform)))
#         xarr_indx = [np.where(x_expanded == xarr[0])[0], np.where(x_expanded == xarr[-1])[0]]
#         y_expanded = np.hstack((np.zeros(xarr_indx[0], dtype=float), y_uniform, np.zeros(len(x_expanded) - xarr_indx[1] - 1, dtype=float)))

#         convv = np.zeros_like(y_expanded)
#         for ix, x in enumerate(x_expanded):
#             xfilter = filterfunc(x - x_expanded)/norm
#             convv[ix] = np.dot(y_expanded, xfilter) 
#         if increase: 
#             return convv, x_expanded
#         else: 
#             return convv[::-1], x_expanded[::-1]
#     else:
#         convv = np.zeros_like(y_uniform)
#         for ix, x in enumerate(x_uniform):
#             xfilter = filterfunc(x - x_uniform)/norm
#             convv[ix] = np.dot(y_uniform, xfilter) 

#         if mapback: 
#             finterpback = interp1d(x_uniform, convv)
#             return finterpback(xarr), xarr
#         if increase: 
#             return convv, x_uniform
#         else: 
#             return convv[::-1], x_uniform[::-1]

def convolve_uniform(xarr, data, resolution, gridsize=int(1e4), truncate_sigma=3, mapback=False, expandx=True):

    if np.all(np.diff(xarr) >= 0):
        increase = True
    elif np.all(np.diff(xarr) <= 0):
        increase = False
        xarr = xarr[::-1]
        data = [y[::-1] for y in data]
    else:
        raise ValueError("The xarr must be sorted")

    xarr_unique, xarr_unique_indices = np.unique(xarr, return_index=True)
    data = [y[xarr_unique_indices] for y in data]
    
    x_uniform = np.linspace(xarr_unique[0], xarr_unique[-1], gridsize, endpoint=True)
    dx_uniform = x_uniform[1] - x_uniform[0]
 
    sigma = resolution/(2*np.sqrt(2*np.log(2)))
    x_kernel = np.arange(-truncate_sigma*sigma, truncate_sigma*sigma + dx_uniform, dx_uniform)
    kernel = np.exp(-x_kernel**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    kernel /= np.sum(kernel)

    if expandx:
        x_grid = np.hstack((
            np.arange(xarr_unique[0] - truncate_sigma*sigma, xarr_unique[0], dx_uniform),
            x_uniform,
            np.arange(xarr_unique[-1] + dx_uniform, xarr_unique[-1]+ truncate_sigma*sigma+dx_uniform, dx_uniform)
        ))
        fy_grid = lambda y: np.hstack((
            np.zeros(len(x_grid[x_grid < xarr_unique[0]])),
            y,
            np.zeros(len(x_grid[x_grid > xarr_unique[-1]]))
        ))
    else:
        x_grid = x_uniform
        fy_grid = lambda y: y

    convv_ydata = []
    for ydata in data:
        finterp = interp1d(xarr_unique, ydata)
        y_uniform = finterp(x_uniform)
        y_grid = fy_grid(y_uniform)
        yconvv = np.convolve(y_grid, kernel, mode="same")
        if mapback:
            finterpback = interp1d(x_grid, yconvv, bounds_error=False, fill_value=0)
            yconvv = finterpback(xarr)

        if not increase:
            yconvv = yconvv[::-1]
        convv_ydata.append(yconvv)

    if expandx:
        return convv_ydata, x_grid if increase else x_grid[::-1]
    elif mapback:
        return convv_ydata, xarr if increase else xarr[::-1]
    else:
        return convv_ydata, x_uniform if increase else x_uniform[::-1]
    
def plot_convolved_lines(models, model_params, quantities, resolution=1, gridsize=1e4, xvar="arcsec", yvars=["n(H)"], maxindex=500, peak=True, xlim=None, legendloc="best", colors=None, marks=None, normalize=False, savefolder="./", savename="emissivity",save=False):
    '''
    plot the local emissivity of the given quantities, convolved with instruments resolution
    '''

    if colors is None: colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    if marks is None: marks = [".", "1", "+", "x", "v", "<", ">"]

    # fig, ax = plt.subplots(1, 1)
    fig, ax = plot_lines(models, model_params, quantities, yvars, xvar, xlim=xlim, normalize=normalize)
    peaks = {}

    print("")
    for ivar, yvar in enumerate(yvars):  
        index = quantities[yvar]
        for iparam, param in enumerate(model_params):
            if xvar == "arcsec":
                xx = (models[param][quantities["Distance"]]*u.cm).to(u.pc)/(400*u.pc)*u.rad.to(u.arcsec)
            elif xvar == "cm":
                xx = models[param][quantities["Distance"]]*u.cm
            elif xvar == "pc":
                xx = (models[param][quantities["Distance"]]*u.cm).to(u.pc)
            else:
                raise ValueError("Invalid xvar, xvar must be one of arcsec, cm, pc")
            
            yconvv, xconvv = convolve_uniform(xx[:maxindex], models[param][index, :maxindex], resolution, gridsize, expandx=True)
            peak = xconvv[np.argmax(yconvv[:maxindex])]
            peaks[yvar] = peak
            print(f"model {param}: {yvar} peaks at {peak:.4f} arcsec")
            if normalize:
                norm = yconvv[:maxindex].max()
                # print(yvar, norm)
                yconvv = yconvv/norm
            ax.plot(xconvv[:maxindex], yconvv[:maxindex], "--", c=colors[ivar], marker=marks[iparam], markevery=0.1)
        # ax.plot(np.nan, np.nan, "--", label="convolved "+yvar, c=colors[ivar])
    ax.plot(np.nan, np.nan, "k--", label="convolved")

    # for iparam, param in enumerate(model_params):
    #     ax.plot(np.nan, np.nan, c="k", label=param, marker=marks[iparam])

    if xlim: ax.set_xlim(xlim)

    if xvar == "arcsec":
        ax.set_xlabel(r'Distance ["]')
    else:
        ax.set_xlabel(f"Distance [{xvar}]") 

    if normalize: 
        ax.set_ylabel("normalized n(X)")
    else:
        ax.set_ylabel("n(X)")
    
    ax.legend(loc=legendloc)
    if save: fig.savefig(savefolder+savename+".pdf", bbox_inches="tight", pad_inches=0)

    return fig, ax, peaks

def column_density(R, b, xarr, data, func_loc_density=None):
    '''
    R: size of the cloud
    b: impact parameter, defined as the distance from the sphere center to the line of sight
    '''

    if not func_loc_density:
        x_arr_unique, x_arr_unique_indices = np.unique(xarr, return_index=True)
        func_loc_density = interp1d(x_arr_unique, data[x_arr_unique_indices])

    dmax = xarr[-1].value # can be earlier?
    smax = np.sqrt(R**2 - b**2)
    smin = 0 if b > R - dmax else np.sqrt((R - dmax)**2 - b**2)
    func_los_loc_density = lambda s: func_loc_density(R - np.sqrt(s**2 + b**2))

    col_density, _ = quad(func_los_loc_density, smin, smax, limit=500)

    return 2*col_density, smin, smax

def column_density_profile(R, b, xarr, data):
    '''
    R: size of the cloud
    b: impact parameter, defined as the distance from the sphere center to the line of sight
    '''

    func_loc_density = interp1d(xarr, data, bounds_error=False, fill_value=0)
    smax = np.sqrt(R**2 - b**2)
    los = np.linspace(-smax, smax, 500)
    func_los_loc_density = lambda s: func_loc_density(R - np.sqrt(s**2 + b**2))
    los_loc_density = func_los_loc_density(los)
    
    col_density = np.zeros_like(los, dtype=float)
    for i, sval in enumerate(los):
        col_density[i], _ = quad(func_los_loc_density, -smax, sval, limit=100)

    return col_density, los_loc_density, los

# def plot_column_densities(model, quantities, Rcloud_pc, barr_pc, resolution, lines=["n(C+ El=2P,J=3/2)", "n(C El=3P,J=1)", "n(CO v=0,J=2)", "n(CO v=0,J=4)", "n(H2O J=1,ka=1,kc=0)"], xvars="arcsec", xlim=None, plot_model=False, plot_dpdr=False, line_labels=None, legendloc="best", line_colors=None, savefolder="./", savename="convv_colden",save=False):
    
#     model_x_cm = model[quantities["Distance"]]*u.cm
#     model_x_cm_unique, model_x_cm_unique_indices = np.unique(model_x_cm, return_index=True)
#     resolution_xunit = arcsec_2_dist(resolution*u.arcsec, "cm")
    
#     column_densities = []
#     convv_column_densities = []
#     convv_xs = []

#     Rcloud_cm = Rcloud_pc.to(u.cm).value
#     barr_cm = barr_pc.to(u.cm).value
    
#     for line in lines:
#         yval = model[quantities[line]]
#         column_density_y = np.zeros_like(barr_cm, dtype=float)

#         func_loc_density = interp1d(model_x_cm_unique, yval[model_x_cm_unique_indices])

#         for i, b in enumerate(barr_cm):
#             column_density_y[i], _, _ = column_density(Rcloud_cm, b, model_x_cm, yval, func_loc_density=func_loc_density)
    
#         convv_column_density, convv_x = convolve_uniform(Rcloud_cm - barr_cm, column_density_y, resolution=resolution_xunit.value, gridsize=500, expandx=True)
        
#         column_densities.append(column_density_y)
#         convv_column_densities.append(convv_column_density)
#         convv_xs.append(convv_x)
    
#     if line_labels is None:
#         line_labels = ["N(C+ El=2P,J=3/2)", "N(C El=3P,J=1)", "N(CO v=0,J=2)", "N(CO v=0,J=4)", "N(H2O J=1,ka=1,kc=0)"]
#     if line_colors is None:
#         line_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

#     fig, axs = plt.subplots(1, 3, figsize=(11, 4))
#     ax_twin = axs[2].twinx()
#     ax_twin.set_yticks([])

#     if plot_model: 
#         ax_twin.plot(np.nan, np.nan, "k:", label="N(X) original")
#     ax_twin.plot(np.nan, np.nan, "k-", label="N(X) convolved")
#     ax_twin.plot(np.nan, np.nan, "k--", label="observation")

#     axs[0].plot(dc2[indx_c2], lc2[indx_c2]/norm_c2, "--", c=line_colors[0], alpha=0.7)
#     axs[0].plot(dc1[indx_c1], lc1[indx_c1]/norm_c1, "--", c=line_colors[1], alpha=0.7)

#     axs[1].plot(d12co_2_1[indx_12co_2_1], l12co_2_1[indx_12co_2_1]/norm_12co_2_1, "--", c=line_colors[2], alpha=0.7)
#     axs[1].plot(d12co_4_3[indx_12co_4_3], l12co_4_3[indx_12co_4_3]/norm_12co_4_3, "--", c=line_colors[3], alpha=0.7) 
#     axs[2].plot(dh2o[indx_h2o], lh2o[indx_h2o]/norm_h2o, "s--", c=line_colors[4], alpha=0.7)

#     darr_arcsec = dist_2_arcsec(Rcloud_pc-barr_pc)

#     for iline, line in enumerate(lines):
#         if plot_model: max_column_density = column_densities[iline].max()
#         max_convv_column_density = convv_column_densities[iline].max()
        
#         if iline < 2:
#             if plot_model:
#                 axs[0].plot(darr_arcsec, column_densities[iline]/max_column_density, ":", c=line_colors[iline])
#             axs[0].plot(dist_2_arcsec(convv_xs[iline]*u.cm), convv_column_densities[iline]/max_convv_column_density, "-", c=line_colors[iline], label=line_labels[iline])
#         elif iline < 4:
#             if plot_model:
#                 axs[1].plot(darr_arcsec, column_densities[iline]/max_column_density, ":", c=line_colors[iline])
#             axs[1].plot(dist_2_arcsec(convv_xs[iline]*u.cm), convv_column_densities[iline]/max_convv_column_density, "-", c=line_colors[iline], label=line_labels[iline])
#         else:
#             if plot_model:
#                 axs[2].plot(darr_arcsec, column_densities[iline]/max_column_density, ":", c=line_colors[iline])
#             axs[2].plot(dist_2_arcsec(convv_xs[iline]*u.cm), convv_column_densities[iline]/max_convv_column_density, "-", c=line_colors[iline], label=line_labels[iline])
#         plt.plot()

#     for ax in axs:
#         ax.set(xlim=xlim, ylim=[-0.1, 1.1], xlabel=f"d [{u.arcsec}]")

#     ax_twin.legend(loc="upper left")
#     axs[0].set_ylabel("Normalized N(X)")
#     axs[0].text(0.01, 0.95, f"R = {Rcloud_pc:.1f}", transform=axs[0].transAxes)
#     axs[0].legend(loc="lower left")
#     axs[1].legend(loc="lower left")
#     axs[2].legend(loc="lower left")

#     axs[1].set_yticks([])
#     axs[2].set_yticks([])

#     fig.tight_layout(w_pad=0, h_pad=0)

#     if save: fig.savefig(savefolder+savename+".pdf", bbox_inches="tight", pad_inches=0)

#     return column_densities, convv_column_densities, convv_xs

##### solving radiative transfer equation #####
def energy_K2erg(energy_K):
    return (const.k_B*energy_K*u.K).to(u.erg)

def get_line_data(linedat, leveldat, bkg_intensity, nl, nu):
    """Get line information needed for solving the radiative transfer equation:
    Einstein coefficients, energy, and background intensity at the line frequency.


    Args:
        linedat (ndarray): line data, read from MeudonPDR
        leveldat (ndarray): level data, read from MeudonPDR
        bkg_intensity (ndarray): background intensity, read from MeudonPDR
        nl (int): level index of lower level, as defined in MeudonPDR
        nu (int): level index of upper level, as defined in MeudonPDR

    Returns:
        tuple: Energy, Einstein coefficient, Einstein coefficients Aul, Bul, and Blu, background intensity at the line frequency
    """
    gl = next(g for n, g in zip(leveldat["n"], leveldat["g"]) if n == nl)
    gu = next(g for n, g in zip(leveldat["n"], leveldat["g"]) if n == nu)

    eng = energy_K2erg(linedat["E"][np.logical_and(linedat["nu"] == nu, linedat["nl"] == nl)])
    nuul = (eng/const.h).to(u.Hz)
    lambul = (const.c/nuul).to(u.AA)
    Aul = linedat["Aul"][np.logical_and(linedat["nu"] == nu, linedat["nl"] == nl)][0]/u.s
    Bul = (Aul*const.c**2/(2*const.h*nuul**3)).cgs
    Blu = gu*Bul/gl
    
    # the unit of bkg_intensity is erg/(cm^2 s sr AA), need to convert to erg/(cm^2 s sr Hz)
    I0 = ((bkg_intensity[2][np.argmin(np.abs(bkg_intensity[0] - lambul.value))]*u.erg/(u.cm**2*u.s*u.sr*u.AA)*lambul**2/const.c).cgs).value[0]

    return eng, Aul, Bul, Blu, I0

def solve_exact_rte(eng, Aul, Bul, Blu, I0, Rcloud_pc, distance_cm, density_l, density_u, lineprofile=None):

    eng_per_4pi = eng/4/np.pi/u.sr

    Rcloud_cm = (Rcloud_pc.to(u.cm)).value
    dmax = distance_cm[-1]
    barr_cm = np.linspace(0.999*Rcloud_cm, Rcloud_cm - dmax, 50)
    print("darr", (Rcloud_cm - barr_cm)*u.cm.to(u.pc))
    print("I0", I0)
    
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

@jit(nopython=True)
def los_density_l(s, smid, b_cm, Rcloud_cm, distance_cm, density_l):
    """
    Computes line-of-sight density for the lower state.
    """
    d = Rcloud_cm - np.sqrt((s - smid) ** 2 + b_cm ** 2)
    return interpolate_density(distance_cm, density_l, d)

@jit(nopython=True)
def func_dInuds(s, Inu, smid, b_cm, Rcloud_cm, Aul, Bul, Blu, eng_per_4pi, distance_cm, density_u, density_l):

    density_u = los_density_u(s, smid, b_cm, Rcloud_cm, distance_cm, density_u)
    density_l = los_density_l(s, smid, b_cm, Rcloud_cm, distance_cm, density_l)
    
    term1 = Aul * density_u
    term2 = Bul * density_u * Inu
    term3 = -Blu * density_l * Inu

    dInuds = (term1 + term2 + term3) * eng_per_4pi
    
    return dInuds