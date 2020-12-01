import os
import os.path
import collections
import functools
import numpy as np
import nuflux
import nuSQUIDSpy as nsq
import nuSQUIDSTools
import oscillator
import prop_store
import likelihood
import matplotlib
import matplotlib.pyplot as plt

s22th24 = 0.1
dm2 = 4.5

dm2_grid = np.logspace(-1, 1, 10*2+1)
th14 = 0.0
s22th24_grid = np.logspace(-2, 0, 10*2+1)
th24_grid = np.arcsin(np.sqrt(s22th24_grid)) / 2.0
s22th34_grid = np.logspace(-2, 0, 10*2+1)
th34_grid = np.arcsin(np.sqrt(s22th34_grid)) / 2.0


th24_index = np.argmin(np.abs(s22th24_grid - s22th24))
dm2_index = np.argmin(np.abs(dm2_grid - dm2))

dm2 = dm2_grid[dm2_index]
th24 = th24_grid[th24_index]

units = nsq.Const()
ebins = np.logspace(1, 6, 100 + 1) * units.GeV
czbins = np.linspace(-1, 1, 100 + 1)

flux = nuflux.makeFlux("H3a_SIBYLL23C")
osc = oscillator.oscillator(
    "H3a_SIBYLL23C", flux, ebins, czbins, "sterile", "./fluxes/", cache_size=10
)

baseline = nsq.nuSQUIDSAtm('./fluxes/standard/H3a_SIBYLL23C.h5')
null = osc[(3, 0,0,0,0,0)]

energy_bins = np.logspace(1, 5, 120+1)
zenith_bins = np.arccos(np.linspace(-1,0,50+1))[::-1]

diff_expect = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
standard_expect_0 = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
standard_expect_1 = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
for j in range(len(zenith_bins)-1):
    for k in range(len(energy_bins)-1):
        z = (zenith_bins[j] + zenith_bins[j+1])/2.0
        e = (energy_bins[k] + energy_bins[k+1])/2.0
        standard_expect_0[k,j] = baseline.EvalFlavor(1, np.cos(z), e*units.GeV, 0)
        standard_expect_1[k,j] = baseline.EvalFlavor(1, np.cos(z), e*units.GeV, 1)


for th34 in th34_grid:
    params = (4, dm2, th14, th24, th34, 0)
    alt = osc[params]

    cm = plt.get_cmap('plasma')
    cm.set_under('black')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sterile_expect_0 = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
    for j in range(len(zenith_bins)-1):
        for k in range(len(energy_bins)-1):
            z = (zenith_bins[j] + zenith_bins[j+1])/2.0
            e = (energy_bins[k] + energy_bins[k+1])/2.0
            sterile_expect_0[k,j] = alt.EvalFlavor(1, np.cos(z), e*units.GeV, 0)

    ex = sterile_expect_0 / standard_expect_0

    fig, ax = plt.subplots(figsize=(7,5))
    X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
    Y = np.array([energy_bins]*(len(zenith_bins))).T
    mesh = ax.pcolormesh(X,Y,ex, cmap=cm, norm=norm)
    ax.set_yscale('log')
    ax.set_ylim((1e1, 1e5))
    #ax.set_xlim((-1,1))
    ax.set_ylabel('Neutrino Energy [GeV]')
    ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
    cb = fig.colorbar(mesh, ax=ax)
    cb.ax.set_ylabel('Alt / null')
    cb.ax.minorticks_on()
    plt.tight_layout()
    fig.savefig('./plots/th34/numu_' + ('%.03f' % th34) + '.png', dpi=200)
    #fig.savefig(outdir + 'survival_numubar.pdf')
    #fig.savefig(outdir + 'survival_numubar.png', dpi=200)
    fig.clf()
    plt.close(fig)

    cm = plt.get_cmap('plasma')
    cm.set_under('black')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sterile_expect_1 = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
    for j in range(len(zenith_bins)-1):
        for k in range(len(energy_bins)-1):
            z = (zenith_bins[j] + zenith_bins[j+1])/2.0
            e = (energy_bins[k] + energy_bins[k+1])/2.0
            sterile_expect_1[k,j] = alt.EvalFlavor(1, np.cos(z), e*units.GeV, 1)

    ex = sterile_expect_1 / standard_expect_1

    fig, ax = plt.subplots(figsize=(7,5))
    X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
    Y = np.array([energy_bins]*(len(zenith_bins))).T
    mesh = ax.pcolormesh(X,Y,ex, cmap=cm, norm=norm)
    ax.set_yscale('log')
    ax.set_ylim((1e1, 1e5))
    #ax.set_xlim((-1,1))
    ax.set_ylabel('Neutrino Energy [GeV]')
    ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
    cb = fig.colorbar(mesh, ax=ax)
    cb.ax.set_ylabel('Transition prob')
    cb.ax.minorticks_on()
    plt.tight_layout()
    fig.savefig('./plots/th34/numubar_' + ('%.03f' % th34) + '.png', dpi=200)
    #fig.savefig(outdir + 'survival_numubar.pdf')
    #fig.savefig(outdir + 'survival_numubar.png', dpi=200)
    fig.clf()
    plt.close(fig)
    #nuSQUIDSTools.PlotFlavorRatio(null, alt, 1, 1, colorscale='log', clim=(-2,2.3))
    #plt.savefig('./plots/th34/' + ('%.03f' % th34) + '.png', dpi=200)
