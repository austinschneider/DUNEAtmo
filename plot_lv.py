"""
### Plot 2d histograms for data and simulation ###
"""

import sys
import os
import os.path
import matplotlib
import matplotlib.style
matplotlib.use('Agg')
matplotlib.style.use('./paper.mplstyle')
import common
import numpy as np
import scipy.stats
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import argparse
import glob
import json

import numpy as np
import h5py as h5
import LWpy
import LeptonInjector
import nuSQUIDSpy as nsq
import json
import sim_tools
import functools
from tqdm import tqdm

outdir = './plots/lv/'

units = nsq.Const()
conv = nsq.nuSQUIDSAtm("./fluxes/conv.h5")

class LV_scenario:
    @staticmethod
    def construct_interp(x, y, z):
        xx = np.sort(np.unique(x))
        xr = np.arange(len(xx))
        yy = np.sort(np.unique(y))
        yr = np.arange(len(yy))
        zz = np.full((len(xx), len(yy)), np.nan)
        print("Setting grid")
        for zi,(xv, yv, zv) in tqdm(enumerate(zip(x,y,z)), total=len(z)):
            xi = xr[xx == xv].item()
            yi = yr[yy == yv].item()
            zz[xi, yi] = zv

        #for xi,xv in tqdm(enumerate(xx), total=len(xx)):
        #    for yi,yv in tqdm(enumerate(yy), total=len(yy)):
        #        zz[xi,yi] = z[np.logical_and(x == xv, y == yv)].item()
        print("Done with grid")
        print("Constructing interpolator")
        print(xx.shape, yy.shape, zz.shape)
        res = scipy.interpolate.interp2d(xx, yy, zz.T, fill_value=np.nan)
        print("Done constructing interpolator")
        return res

    @staticmethod
    def interps_from_data(data, n_pre_columns=1):
        data = np.asarray(data)
        assert(len(data.shape) == 2)
        n = data.shape[1]
        assert(n > (n_pre_columns+2))
        assert((n-(n_pre_columns+2)) % 2 == 0)
        x = data[:,(0+n_pre_columns)]
        y = data[:,(1+n_pre_columns)]
        interps = [LV_scenario.construct_interp(x, y, data[:,i]) for i in range((n_pre_columns+2),n)]
        return interps

    def __init__(self, nu_fname, nubar_fname, n_pre_columns=1):
        self.nu_data = np.loadtxt(fname=nu_fname)
        self.nubar_data = np.loadtxt(fname=nubar_fname)
        self.nu_interps = LV_scenario.interps_from_data(self.nu_data, n_pre_columns)
        self.nubar_interps = LV_scenario.interps_from_data(self.nubar_data, n_pre_columns)
        self.n_neutrinos = len(self.nu_interps)/2
        self.n_antineutrinos = len(self.nubar_interps)/2

    def get_interps(self, flavortype, mattertype):
        assert(abs(mattertype) < 2)
        n = [self.n_neutrinos, self.n_antineutrinos][mattertype]
        interp_set = [self.nu_interps, self.nubar_interps][mattertype]
        assert(flavortype < n)
        null_i = 2 * int(flavortype) + 1
        bsm_i = 2 * int(flavortype)
        null_interp = interp_set[null_i]
        bsm_interp = interp_set[bsm_i]

        return null_interp, bsm_interp

    def correction(self, x, y, flavortype, mattertype):
        null_interp, bsm_interp = self.get_interps(flavortype, mattertype)
        null = null_interp(x, y)
        bsm = bsm_interp(x, y)
        res = bsm/null
        res[np.isnan(res)] = 0.0
        return res

def color_bounds(arrays, lower_alpha=0.95, logsnap=False, pos=False):
    amin = np.inf
    amax = -np.inf
    for ar in arrays:
        ar = ar.flat
        if pos:
            ar = ar[ar>0]
        sar = np.sort(ar)[::-1]
        cdf = np.cumsum(sar)/np.sum(sar)
        i = np.arange(len(sar))[cdf >= lower_alpha][0]
        armin = sar[i]
        armax = np.amax(ar)

        amin = min(armin, amin)
        amax = max(armax, amax)
    if logsnap:
        amin = 10**np.floor(np.log10(amin))
        amax = 10**np.ceil(np.log10(amax))
    return amin, amax

lv_s = LV_scenario('./fluxes/LV/dimension_3/fluxes_22_neutrino.txt', './fluxes/LV/dimension_3/fluxes_22_antineutrino.txt')

energy_bins = np.logspace(1, 5, 40+1)
zenith_bins = np.arccos(np.linspace(-1,0,20+1))[::-1]

expect_shape = (len(energy_bins)-1, len(zenith_bins)-1)
diff_expect = np.empty(expect_shape)
standard_expect = np.empty(expect_shape)
lv_expect = np.empty(expect_shape)
correction_expect = np.empty(expect_shape)
pairs = []

print("nusquids")
for j in range(len(zenith_bins)-1):
    for k in range(len(energy_bins)-1):
        z = (zenith_bins[j] + zenith_bins[j+1])/2.0
        e = (energy_bins[k] + energy_bins[k+1])/2.0
        conv_flux = conv.EvalFlavor(1, np.cos(z), e*units.GeV, 1)
        correction_expect[k,j] = lv_s.correction(np.cos(z), e, 1, 1)
        conv_lv_flux = conv_flux * correction_expect[k,j]

        diff_expect[k,j] = 1 - conv_lv_flux / conv_flux
        standard_expect[k,j] = conv_flux
        lv_expect[k,j] = conv_lv_flux

cm = plt.get_cmap('plasma')
cm.set_under('black')
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,diff_expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Fractional deficit')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'survival_numubar.pdf')
fig.savefig(outdir + 'survival_numubar.png', dpi=200)
fig.clf()
plt.close(fig)

norm = matplotlib.colors.LogNorm()

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,correction_expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Fractional deficit')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'correction_numubar.pdf')
fig.savefig(outdir + 'correction_numubar.png', dpi=200)
fig.clf()
plt.close(fig)

norm = matplotlib.colors.LogNorm()
fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,standard_expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Flux')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'standard_numubar.pdf')
fig.savefig(outdir + 'standard_numubar.png', dpi=200)
fig.clf()
plt.close(fig)

norm = matplotlib.colors.LogNorm()

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,lv_expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Flux')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'lv_numubar.pdf')
fig.savefig(outdir + 'lv_numubar.png', dpi=200)
fig.clf()
plt.close(fig)

diff_expect = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
standard_expect = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
lv_expect = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
pairs = []
for j in range(len(zenith_bins)-1):
    for k in range(len(energy_bins)-1):
        z = (zenith_bins[j] + zenith_bins[j+1])/2.0
        e = (energy_bins[k] + energy_bins[k+1])/2.0
        conv_flux = conv.EvalFlavor(1, np.cos(z), e*units.GeV, 0)
        correction_expect[k,j] = lv_s.correction(np.cos(z), e, 1, 0)
        conv_lv_flux = conv_flux * correction_expect[k,j]
        diff_expect[k,j] = 1 - conv_lv_flux / conv_flux
        standard_expect[k,j] = conv_flux
        lv_expect[k,j] = conv_lv_flux

norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,diff_expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Fractional deficit')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'survival_numu.pdf')
fig.savefig(outdir + 'survival_numu.png', dpi=200)
fig.clf()
plt.close(fig)

norm = matplotlib.colors.LogNorm()

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,correction_expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Fractional deficit')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'correction_numu.pdf')
fig.savefig(outdir + 'correction_numu.png', dpi=200)
fig.clf()
plt.close(fig)

norm = matplotlib.colors.LogNorm()

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,standard_expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Flux')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'standard_numu.pdf')
fig.savefig(outdir + 'standard_numu.png', dpi=200)
fig.clf()
plt.close(fig)

norm = matplotlib.colors.LogNorm()

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,lv_expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Flux')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'lv_numu.pdf')
fig.savefig(outdir + 'lv_numu.png', dpi=200)
fig.clf()
plt.close(fig)

data = json.load(open('./weighted/weighted.json', 'r'))
energy = np.array(data["energy"])
zenith = np.pi - np.array(data["zenith"])
azimuth = 2.*np.pi - np.array(data["azimuth"])
bjorken_x = np.array(data["bjorken_x"])
bjorken_y = np.array(data["bjorken_y"])
final_type_0 = np.array(data["final_type_0"]).astype(int)
final_type_1 = np.array(data["final_type_1"]).astype(int)
particle = np.array(data["particle"]).astype(int)
x = np.array(data["x"])
y = np.array(data["y"])
z = np.array(data["z"])
total_column_depth = np.array(data["total_column_depth"])
gen_prob = np.array(data["gen_prob"])
mu_energy = np.array(data["mu_energy"])
mu_x = np.array(data["mu_x"])
mu_y = np.array(data["mu_y"])
mu_z = np.array(data["mu_z"])
mu_zenith = np.pi - np.array(data["mu_zenith"])
mu_azimuth = 2.*np.pi - np.array(data["mu_azimuth"])
entry_energy = np.array(data["entry_energy"])
entry_x = np.array(data["entry_x"])
entry_y = np.array(data["entry_y"])
entry_z = np.array(data["entry_z"])
entry_zenith = np.pi - np.array(data["entry_zenith"])
entry_azimuth = 2.*np.pi - np.array(data["entry_azimuth"])
exit_energy = np.array(data["exit_energy"])
exit_x = np.array(data["exit_x"])
exit_y = np.array(data["exit_y"])
exit_z = np.array(data["exit_z"])
exit_zenith = np.pi - np.array(data["exit_zenith"])
exit_azimuth = 2.*np.pi - np.array(data["exit_azimuth"])
track_length = np.array(data["track_length"])
morphology = np.array([sim_tools.EventMorphology(m) for m in data["morphology"]])
deposited_energy = np.array(data["deposited_energy"])
entry_distance = np.sqrt((entry_x - x)**2 + (entry_y - y)**2 + (entry_z - z)**2)
injector_count = np.array(data["injector_count"]).astype(int)[::-1]
muon_start_energy = np.array(entry_energy)
mask = np.isnan(muon_start_energy)
muon_start_energy[mask] = mu_energy[mask]
muon_start_zenith = np.array(entry_zenith)
mask = np.isnan(muon_start_zenith)
muon_start_zenith[mask] = mu_zenith[mask]
muon_nx = exit_x - entry_x
muon_ny = exit_y - entry_y
muon_nz = exit_z - entry_z
muon_d = np.sqrt(muon_nx**2 + muon_ny**2 + muon_nz**2)
muon_nx /= muon_d
muon_ny /= muon_d
muon_nz /= muon_d
muon_zenith = np.pi - np.arccos(muon_nz)
muon_azimuth = 2.*np.pi - np.arctan2(muon_ny, muon_nx)

data = np.empty(len(energy), dtype=[
      ("energy", energy.dtype),
      ("zenith", zenith.dtype),
      ("azimuth", azimuth.dtype),
      ("bjorken_x", bjorken_x.dtype),
      ("bjorken_y", bjorken_y.dtype),
      ("final_type_0", final_type_0.dtype),
      ("final_type_1", final_type_1.dtype),
      ("particle", particle.dtype),
      ("x", x.dtype),
      ("y", y.dtype),
      ("z", z.dtype),
      ("total_column_depth", total_column_depth.dtype),
      ("gen_prob", gen_prob.dtype),
      ("entry_energy", entry_energy.dtype),
      ("entry_x", entry_x.dtype),
      ("entry_y", entry_y.dtype),
      ("entry_z", entry_z.dtype),
      ("entry_zenith", entry_zenith.dtype),
      ("entry_azimuth", entry_azimuth.dtype),
      ("exit_energy", exit_energy.dtype),
      ("exit_x", exit_x.dtype),
      ("exit_y", exit_y.dtype),
      ("exit_z", exit_z.dtype),
      ("exit_zenith", exit_zenith.dtype),
      ("exit_azimuth", exit_azimuth.dtype),
      ("track_length", track_length.dtype),
      ("morphology", morphology.dtype),
      ("deposited_energy", deposited_energy.dtype),
      ("entry_distance", entry_distance.dtype),
      ])

data["energy"] = energy
data["zenith"] = zenith
data["azimuth"] = azimuth
data["bjorken_x"] = bjorken_x
data["bjorken_y"] = bjorken_y
data["final_type_0"] = final_type_0
data["final_type_1"] = final_type_1
data["particle"] = particle
data["x"] = x
data["y"] = y
data["z"] = z
data["total_column_depth"] = total_column_depth
data["gen_prob"] = gen_prob
data["entry_energy"] = entry_energy
data["entry_x"] = entry_x
data["entry_y"] = entry_y
data["entry_z"] = entry_z
data["entry_zenith"] = entry_zenith
data["entry_azimuth"] = entry_azimuth
data["exit_energy"] = exit_energy
data["exit_x"] = exit_x
data["exit_y"] = exit_y
data["exit_z"] = exit_z
data["exit_zenith"] = exit_zenith
data["exit_azimuth"] = exit_azimuth
data["track_length"] = track_length
data["morphology"] = morphology
data["deposited_energy"] = deposited_energy
data["entry_distance"] = entry_distance

props = data

flux = np.empty(len(props))

flavors = (np.abs(particle) / 2 - 6).astype(int)

flux_standard = np.zeros(len(props))
flux_lv = np.zeros(len(props))

for i, (ff, zz, ee, particle_type) in enumerate(zip(flavors, props["zenith"], props["energy"], props["particle"])):
    ff = int(ff)
    zz = float(zz)
    ee = float(ee)
    particle_type = int(particle_type)
    particle_type = 0 if particle_type > 0 else 1

    cth = np.cos(zz)
    cth = min(1.0, cth)
    cth = max(-1.0, cth)

    try:
        conv_flux = conv.EvalFlavor(ff, cth, ee*units.GeV, particle_type)
        flux_standard[i] = conv_flux
        correction = lv_s.correction(np.cos(zz), ee, ff, particle_type)
        conv_lv_flux = correction * conv_flux
        flux_lv[i] = conv_lv_flux
    except:
        print(ff, cth, ee*units.GeV, particle_type)
        print(list(zip(props.dtype.names, props[i])))
        raise

livetime = 365.25 * 24 * 3600

weights = flux_standard * livetime / gen_prob
standard_weights = flux_standard * livetime / gen_prob
lv_weights = flux_lv * livetime / gen_prob

energy_bins = np.logspace(-1, 5, 60+1)
zenith_bins = np.arccos(np.linspace(-1,0,50+1))[::-1]

masks = common.get_bin_masks(energy, zenith, energy_bins, zenith_bins)

standard_expect = np.array([np.sum(standard_weights[m]) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T
lv_expect = np.array([np.sum(lv_weights[m]) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T

amin, amax =color_bounds([standard_expect, lv_expect], lower_alpha=0.95, logsnap=True, pos=True)

norm = matplotlib.colors.LogNorm(vmin=amin, vmax=amax, clip=False)
fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,standard_expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Events per year per module')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'neutrino_standard_dist.pdf')
fig.savefig(outdir + 'neutrino_standard_dist.png', dpi=200)
fig.clf()
plt.close(fig)

norm = matplotlib.colors.LogNorm(vmin=amin, vmax=amax, clip=False)
fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,lv_expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Events per year per module')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'neutrino_lv_dist.pdf')
fig.savefig(outdir + 'neutrino_lv_dist.png', dpi=200)
fig.clf()
plt.close(fig)
