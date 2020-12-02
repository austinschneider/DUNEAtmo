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

outdir = './plots/conv_sin22th_0p02_dm2_p15/'

units = nsq.Const()
conv = nsq.nuSQUIDSAtm("./fluxes/conv.h5")
#conv_sterile = nsq.nuSQUIDSAtm("./fluxes/conv_sin22th_0p1.h5")
conv_sterile = nsq.nuSQUIDSAtm("./fluxes/conv_sin22th_0p02_dm2_p15.h5")

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

energy_bins = np.logspace(1, 5, 12+1)
zenith_bins = np.arccos(np.linspace(-1,0,5+1))[::-1]

diff_expect = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
standard_expect = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
sterile_expect = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
pairs = []
for j in range(len(zenith_bins)-1):
    for k in range(len(energy_bins)-1):
        z = (zenith_bins[j] + zenith_bins[j+1])/2.0
        e = (energy_bins[k] + energy_bins[k+1])/2.0
        conv_flux = conv.EvalFlavor(1, np.cos(z), e*units.GeV, 1)
        conv_sterile_flux = conv_sterile.EvalFlavor(1, np.cos(z), e*units.GeV, 1)
        diff_expect[k,j] = 1 - conv_sterile_flux / conv_flux
        standard_expect[k,j] = conv_flux
        sterile_expect[k,j] = conv_sterile_flux

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

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,sterile_expect, cmap=cm, norm=norm)
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
fig.savefig(outdir + 'sterile_numubar.pdf')
fig.savefig(outdir + 'sterile_numubar.png', dpi=200)
fig.clf()
plt.close(fig)

diff_expect = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
standard_expect = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
sterile_expect = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
pairs = []
for j in range(len(zenith_bins)-1):
    for k in range(len(energy_bins)-1):
        z = (zenith_bins[j] + zenith_bins[j+1])/2.0
        e = (energy_bins[k] + energy_bins[k+1])/2.0
        conv_flux = conv.EvalFlavor(1, np.cos(z), e*units.GeV, 0)
        conv_sterile_flux = conv_sterile.EvalFlavor(1, np.cos(z), e*units.GeV, 0)
        diff_expect[k,j] = 1 - conv_sterile_flux / conv_flux
        standard_expect[k,j] = conv_flux
        sterile_expect[k,j] = conv_sterile_flux

norm = matplotlib.colors.LogNorm()

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

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,sterile_expect, cmap=cm, norm=norm)
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
fig.savefig(outdir + 'sterile_numu.pdf')
fig.savefig(outdir + 'sterile_numu.png', dpi=200)
fig.clf()
plt.close(fig)

flux = np.empty(len(props))

flavors = (np.abs(particle) / 2 - 6).astype(int)

flux_standard = np.zeros(len(props))
flux_sterile = np.zeros(len(props))

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

        conv_sterile_flux = conv_sterile.EvalFlavor(ff, cth, ee*units.GeV, particle_type)
        flux_sterile[i] = conv_sterile_flux
    except:
        print(ff, cth, ee*units.GeV, particle_type)
        print(list(zip(props.dtype.names, props[i])))
        raise

livetime = 365.25 * 24 * 3600

weights = flux_standard * livetime / gen_prob
standard_weights = flux_standard * livetime / gen_prob
sterile_weights = flux_sterile * livetime / gen_prob
"""
def print_stats(name, v, z):
    m0 = np.cos(z) < 0
    m1 = np.cos(z) > 0
    v0 = v[m0]
    v1 = v[m1]
    v_min0 = np.amin(v0)
    v_max0 = np.amax(v0)
    v_mean0 = np.sum(v0)/len(v0)
    v_stddev0 = np.sqrt(np.sum((v0-v_mean0)**2)/len(v0))

    v_min1 = np.amin(v1)
    v_max1 = np.amax(v1)
    v_mean1 = np.sum(v1)/len(v1)
    v_stddev1 = np.sqrt(np.sum((v1-v_mean1)**2)/len(v1))
    print(name + ":")
    print("\t"+"min:", v_min0, v_min1)
    print("\t"+"max:", v_max0, v_max1)
    print("\t"+"mean:", v_mean0, v_mean1)
    print("\t"+"stddev:", v_stddev0, v_stddev1)
    print()

print_stats("gen_prob", gen_prob, zenith)
print_stats("1/gen_prob", 1./gen_prob, zenith)
print_stats("weights", weights, zenith)
print_stats("1/weights", 1./weights, zenith)
print_stats("standard_weights", standard_weights, zenith)
print_stats("1/standard_weights", 1./standard_weights, zenith)
print_stats("sterile_weights", sterile_weights, zenith)
print_stats("1/sterile_weights", 1./sterile_weights, zenith)
print_stats("flux_standard", flux_standard, zenith)
print_stats("1/flux_standard", 1./flux_standard, zenith)
print_stats("flux_sterile", flux_sterile, zenith)
print_stats("1/flux_sterile", 1./flux_sterile, zenith)

def print_stats(name):
    m0 = np.cos(zenith) < 0
    m1 = np.cos(zenith) > 0
    w0 = standard_weights[m0]
    w1 = standard_weights[m1]
    v0 = props[name][m0]
    v1 = props[name][m1]
    #good = ~np.logical_or(np.isnan(v), np.isinf(v))
    #w = w[good]
    #v = v[good]
    v_min0 = np.amin(v0)
    v_max0 = np.amax(v0)
    v_mean0 = np.sum(v0*w0)/np.sum(w0)
    v_stddev0 = np.sqrt(np.sum((v0-v_mean0)**2*w0)/np.sum(w0))

    v_min1 = np.amin(v1)
    v_max1 = np.amax(v1)
    v_mean1 = np.sum(v1*w1)/np.sum(w1)
    v_stddev1 = np.sqrt(np.sum((v1-v_mean1)**2*w1)/np.sum(w1))
    print(name + ":")
    print("\t"+"min:", v_min0, v_min1)
    print("\t"+"max:", v_max0, v_max1)
    print("\t"+"mean:", v_mean0, v_mean1)
    print("\t"+"stddev:", v_stddev0, v_stddev1)
    print()
keys = sorted(props.dtype.names)
for k in keys:
    print_stats(k)
"""
cut_mask = np.ones(len(energy)).astype(bool)
#cut_mask = track_length > 2
#cut_mask = np.logical_and(cut_mask, entry_energy > 100)
#cut_mask = energy > 100
#cut_mask = np.logical_and(cut_mask, entry_zenith > np.pi/2.)
cut_mask = np.logical_and(cut_mask, zenith > np.pi/2.)
cut_mask = np.logical_and(cut_mask, track_length > 2.)
up_mask = zenith > np.pi/2.
numu_mask = particle == 14
numubar_mask = particle == -14
track_length_mask = track_length > 2.
deposited_energy_mask = deposited_energy > 1
through_going_mask = morphology == 3
tev_other_mask = np.logical_or(energy <= 1e3, energy >= 2e3)
tev_mask = np.logical_and(energy > 1e3, energy < 2e3)
three_gev_other_mask = np.logical_or(energy <= 3e2, energy >= 1e3)
three_gev_mask = np.logical_and(energy > 3e2, energy < 1e3)
one_gev_other_mask = np.logical_or(energy <= 1e2, energy >= 3e2)
one_gev_mask = np.logical_and(energy > 1e2, energy < 3e2)
vup_mask = np.cos(zenith) < -0.8
starting_contained_mask = np.logical_or(np.logical_or(morphology == 1, morphology == 4), morphology == 5)
red = lambda x: functools.reduce(np.logical_and, x)

energy_bins = np.logspace(-1, 5, 18+1)
zenith_bins = np.arccos(np.linspace(-1,0,5+1))[::-1]

masks = common.get_bin_masks(energy, zenith, energy_bins, zenith_bins)

standard_expect = np.array([np.sum(standard_weights[m]) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T
print('Standard expectation:', np.sum(standard_expect))
sterile_expect = np.array([np.sum(sterile_weights[m]) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T
print('Sterile expectation:', np.sum(sterile_expect))

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
ex = standard_expect-sterile_expect
amin, amax = common.color_bounds([ex], pos=True, logsnap=True, lower_alpha=0.95)
norm = matplotlib.colors.LogNorm(vmin=amin, vmax=amax, clip=True)
norm = matplotlib.colors.LogNorm()
mesh = ax.pcolormesh(X,Y,ex, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Deficit per year per module')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'neutrino_deficit_dist.pdf')
fig.savefig(outdir + 'neutrino_deficit_dist.png', dpi=200)
fig.clf()
plt.close(fig)

norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=1, clip=False)

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
ex = (standard_expect - sterile_expect) / standard_expect
mesh = ax.pcolormesh(X,Y,ex, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Fractional deficit per year per module')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'neutrino_frac_deficit_dist.pdf')
fig.savefig(outdir + 'neutrino_frac_deficit_dist.png', dpi=200)
fig.clf()
plt.close(fig)

amin, amax = common.color_bounds([standard_expect, sterile_expect], pos=True, logsnap=True, lower_alpha=0.95)

norm = matplotlib.colors.LogNorm(vmin=amin, vmax=amax, clip=True)
norm = matplotlib.colors.LogNorm()
fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
print('Standard expectation:', np.sum(standard_expect))
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

norm = matplotlib.colors.LogNorm(vmin=amin, vmax=amax, clip=True)
fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,sterile_expect, cmap=cm, norm=norm)
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
fig.savefig(outdir + 'neutrino_sterile_dist.pdf')
fig.savefig(outdir + 'neutrino_sterile_dist.png', dpi=200)
fig.clf()
plt.close(fig)


mask = ~np.isnan(total_column_depth)
total_column_depth_expect = np.array([np.sum(standard_weights[np.logical_and(m,mask)]*total_column_depth[np.logical_and(m,mask)])/np.sum(standard_weights[np.logical_and(m,mask)]) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T

norm = matplotlib.colors.LogNorm()
fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,total_column_depth_expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Average tcd')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'neutrino_total_column_depth_dist.pdf')
fig.savefig(outdir + 'neutrino_total_column_depth_dist.png', dpi=200)
fig.clf()
plt.close(fig)

masks = common.get_bin_masks(muon_start_energy, muon_start_zenith, energy_bins, zenith_bins)

standard_expect = np.array([np.sum(standard_weights[m]) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T
print('Standard expectation:', np.sum(standard_expect))
sterile_expect = np.array([np.sum(sterile_weights[m]) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T
print('Sterile expectation:', np.sum(sterile_expect))

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
ex = standard_expect-sterile_expect
amin, amax = common.color_bounds([ex], pos=True, logsnap=True, lower_alpha=0.95)
norm = matplotlib.colors.LogNorm(vmin=amin, vmax=amax, clip=True)
print('Muon deficit over threshold:', np.sum([x for x in ex.flat if x > amin]))
norm = matplotlib.colors.LogNorm()
mesh = ax.pcolormesh(X,Y,ex, cmap=cm, norm=norm)
ax.set_yscale('log')
#ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Muon energy at detector [GeV]')
ax.set_xlabel(r'Muon $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Deficit per year per module')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'muon_deficit_dist.pdf')
fig.savefig(outdir + 'muon_deficit_dist.png', dpi=200)
fig.clf()
plt.close(fig)

norm = matplotlib.colors.LogNorm(vmin=1e-4, vmax=1, clip=False)
#norm = matplotlib.colors.LogNorm()

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T

ex = (standard_expect-sterile_expect) / standard_expect
k = sterile_expect
l = np.array(standard_expect)
k *= 8
l *= 8
pois = l - k*np.log(l) + scipy.special.gammaln(k)
pois_data = k - k*np.log(k) + scipy.special.gammaln(k)
print("DeltaChi2", np.sum(pois-pois_data))
mesh = ax.pcolormesh(X,Y,pois-pois_data, cmap=cm, norm=norm)
ax.set_yscale('log')
#ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Muon energy at detector [GeV]')
ax.set_xlabel(r'Muon $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Reduced negative LLH')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'muon_pois_dist.pdf')
fig.savefig(outdir + 'muon_pois_dist.png', dpi=200)
fig.clf()
plt.close(fig)

amin, amax = common.color_bounds([standard_expect, sterile_expect], pos=True, logsnap=True, lower_alpha=0.95)

norm = matplotlib.colors.LogNorm(vmin=amin, vmax=amax, clip=True)
norm = matplotlib.colors.LogNorm()
fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,standard_expect, cmap=cm, norm=norm)
print('Standard expectation:', np.sum(standard_expect))
ax.set_yscale('log')
#ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Muon energy at detector [GeV]')
ax.set_xlabel(r'Muon $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Events per year per module')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'muon_standard_dist.pdf')
fig.savefig(outdir + 'muon_standard_dist.png', dpi=200)
fig.clf()
plt.close(fig)

norm = matplotlib.colors.LogNorm(vmin=amin, vmax=amax, clip=True)
fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,sterile_expect, cmap=cm, norm=norm)
ax.set_yscale('log')
#ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Muon energy at detector [GeV]')
ax.set_xlabel(r'Muon $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Events per year per module')
cb.ax.minorticks_on()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'muon_sterile_dist.pdf')
fig.savefig(outdir + 'muon_sterile_dist.png', dpi=200)
fig.clf()
plt.close(fig)
