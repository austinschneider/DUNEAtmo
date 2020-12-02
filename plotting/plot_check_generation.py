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

outdir = './plots/'

units = nsq.Const()
#kaon = nsq.nuSQUIDSAtm("./kaon_atmospheric_final.hdf5")
#pion = nsq.nuSQUIDSAtm("./pion_atmospheric_final.hdf5")
kaon = nsq.nuSQUIDSAtm("./kaon_atmospheric.hdf5")
pion = nsq.nuSQUIDSAtm("./pion_atmospheric.hdf5")
prompt = nsq.nuSQUIDSAtm("./prompt_atmospheric_final.hdf5")
conv = nsq.nuSQUIDSAtm("./conv.h5")
conv_sterile = nsq.nuSQUIDSAtm("./conv_sin22th_0p1.h5")

s = LWpy.read_stream('./config_DUNE.lic')
blocks = s.read()
earth_model_params = [
    "DUNE",
    "../LWpy/LWpy/resources/earthparams/",
    ["PREM_dune"],
    ["Standard"],
    "NoIce",
    20.0*LeptonInjector.Constants.degrees,
    1480.0*LeptonInjector.Constants.m]

generators = []
for block in blocks:
    block_name, block_version, _ = block
    if block_name == 'EnumDef':
        continue
    elif block_name == 'VolumeInjectionConfiguration':
        gen = LWpy.volume_generator(block)
    elif block_name == 'RangedInjectionConfiguration':
        gen = LWpy.ranged_generator(block, earth_model_params)
    else:
        raise ValueError("Unrecognized block! " + block_name)
    generators.append(gen)

data = json.load(open('propagated.json', 'r'))
energy = np.array(data["energy"])
zenith = np.array(data["zenith"])
azimuth = np.array(data["azimuth"])
bjorken_x = np.array(data["bjorken_x"])
bjorken_y = np.array(data["bjorken_y"])
final_type_0 = np.array(data["final_type_0"]).astype(int)
final_type_1 = np.array(data["final_type_1"]).astype(int)
particle = np.array(data["particle"]).astype(int)
x = np.array(data["x"])
y = np.array(data["y"])
z = np.array(data["z"])
total_column_depth = np.array(data["total_column_depth"])
mu_energy = np.array(data["mu_energy"])
mu_x = np.array(data["mu_x"])
mu_y = np.array(data["mu_y"])
mu_z = np.array(data["mu_z"])
mu_zenith = np.array(data["mu_zenith"])
mu_azimuth = np.array(data["mu_azimuth"])
entry_energy = np.array(data["entry_energy"])
entry_x = np.array(data["entry_x"])
entry_y = np.array(data["entry_y"])
entry_z = np.array(data["entry_z"])
entry_zenith = np.array(data["entry_zenith"])
entry_azimuth = np.array(data["entry_azimuth"])
exit_energy = np.array(data["exit_energy"])
exit_x = np.array(data["exit_x"])
exit_y = np.array(data["exit_y"])
exit_z = np.array(data["exit_z"])
exit_zenith = np.array(data["exit_zenith"])
exit_azimuth = np.array(data["exit_azimuth"])
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
print(entry_x)
print(exit_x)
muon_d = np.sqrt(muon_nx**2 + muon_ny**2 + muon_nz**2)
muon_nx /= muon_d
muon_ny /= muon_d
muon_nz /= muon_d
muon_zenith = np.arccos(muon_nz)
print(muon_zenith)
muon_azimuth = np.arctan2(muon_ny, muon_nx)

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

nu_interactions_list = LWpy.get_standard_interactions()
int_model = LWpy.interaction_model(nu_interactions_list, earth_model_params)
gen_prob = np.zeros(len(props))
for i, gen in enumerate(generators):
    #prob = gen.prob(props)
    p = gen.prob_final_state(props)
    p *= gen.prob_stat(props)
    #p *= injector_count[i]
    nonzero = p != 0
    if np.any(nonzero):
        p[nonzero] *= gen.prob_dir(props[nonzero])
        nonzero = p != 0
    if np.any(nonzero):
        p[nonzero] *= gen.prob_e(props[nonzero])
        nonzero = p != 0
    if np.any(nonzero):
        p[nonzero] *= gen.prob_area(props[nonzero])
        nonzero = p != 0
    if np.any(nonzero):
        p[nonzero] *= gen.prob_pos(props[nonzero])
        nonzero = p != 0
    if np.any(nonzero):
        p[nonzero] *= gen.prob_kinematics(props[nonzero])
        nonzero = p != 0
    if np.any(nonzero):
        first_pos, last_pos = gen.get_considered_range(props[nonzero])
        pos_prob = int_model.prob_pos(props[nonzero], first_pos, last_pos)
        int_prob = int_model.prob_interaction(props[nonzero], first_pos, last_pos)
        p[nonzero] /= pos_prob * int_prob
    gen_prob += p
k_prob = int_model.prob_kinematics(props)
fs_prob = int_model.prob_final_state(props)
gen_prob /= k_prob * fs_prob

energy_bins = np.logspace(1, 5, 80+1)
zenith_bins = np.arccos(np.linspace(-1,0,50+1))[::-1]

expect = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
pairs = []
for j in range(len(zenith_bins)-1):
    for k in range(len(energy_bins)-1):
        z = (zenith_bins[j] + zenith_bins[j+1])/2.0
        e = (energy_bins[k] + energy_bins[k+1])/2.0
        conv_flux = conv.EvalFlavor(1, np.cos(z), e*units.GeV, 1)
        conv_sterile_flux = conv_sterile.EvalFlavor(1, np.cos(z), e*units.GeV, 1)
        expect[k,j] = 1 - conv_sterile_flux / conv_flux

cm = plt.get_cmap('plasma')
cm.set_under('white')
norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Fractional deficit')
cb.ax.minorticks_off()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'survival_numubar.pdf')
fig.savefig(outdir + 'survival_numubar.png', dpi=200)
fig.clf()
plt.close(fig)

expect = np.empty((len(energy_bins)-1, len(zenith_bins)-1))
pairs = []
for j in range(len(zenith_bins)-1):
    for k in range(len(energy_bins)-1):
        z = (zenith_bins[j] + zenith_bins[j+1])/2.0
        e = (energy_bins[k] + energy_bins[k+1])/2.0
        conv_flux = conv.EvalFlavor(1, np.cos(z), e*units.GeV, 0)
        conv_sterile_flux = conv_sterile.EvalFlavor(1, np.cos(z), e*units.GeV, 0)
        expect[k,j] = 1 - conv_sterile_flux / conv_flux

cm = plt.get_cmap('plasma')
cm.set_under('white')
norm = matplotlib.colors.LogNorm()

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Fractional deficit')
cb.ax.minorticks_off()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'survival_numu.pdf')
fig.savefig(outdir + 'survival_numu.png', dpi=200)
fig.clf()
plt.close(fig)

flux = np.empty(len(props))

flavors = (np.abs(particle) / 2 - 6).astype(int)

flux_standard = np.empty(len(props))
flux_sterile = np.empty(len(props))

for i, (ff, zz, ee, particle_type) in enumerate(zip(flavors, props["zenith"], props["energy"], props["particle"])):
    ff = int(ff)
    zz = float(zz)
    ee = float(ee)
    particle_type = int(particle_type)
    particle_type = 0 if particle_type > 0 else 1
    if np.cos(zz) > 0:
        flux[i] = 0
        continue
    #prompt_flux = prompt.EvalFlavor(ff, np.cos(zz), ee*units.GeV, particle_type)
    pion_flux = pion.EvalFlavor(ff, np.cos(zz), ee*units.GeV, particle_type)
    kaon_flux = kaon.EvalFlavor(ff, np.cos(zz), ee*units.GeV, particle_type)
    conv_flux = conv.EvalFlavor(ff, np.cos(zz), ee*units.GeV, particle_type)
    conv_sterile_flux = conv_sterile.EvalFlavor(ff, np.cos(zz), ee*units.GeV, particle_type)
    flux_sterile[i] = conv_sterile_flux
    flux_standard[i] = conv_flux

livetime = 365.25 * 24 * 3600

weights = flux_standard * livetime / gen_prob
standard_weights = flux_standard * livetime / gen_prob
sterile_weights = flux_sterile * livetime / gen_prob
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

energy_bins = np.logspace(-1, 5, 120+1)
zenith_bins = np.arccos(np.linspace(-1,0,50+1))[::-1]

masks = common.get_bin_masks(energy, zenith, energy_bins, zenith_bins)

standard_expect = np.array([np.sum(standard_weights[m]) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T
sterile_expect = np.array([np.sum(sterile_weights[m]) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T

cm = plt.get_cmap('plasma_r')
cm.set_under('white')
#norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=10, clip=False)
norm = matplotlib.colors.LogNorm()

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
mesh = ax.pcolormesh(X,Y,standard_expect-sterile_expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Deficit per year per module')
cb.ax.minorticks_off()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'neutrino_deficit_dist.pdf')
fig.savefig(outdir + 'neutrino_deficit_dist.png', dpi=200)
fig.clf()
plt.close(fig)

norm = matplotlib.colors.LogNorm(vmin=1e-5, vmax=1, clip=False)
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
cb.ax.minorticks_off()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'neutrino_standard_dist.pdf')
fig.savefig(outdir + 'neutrino_standard_dist.png', dpi=200)
fig.clf()
plt.close(fig)

norm = matplotlib.colors.LogNorm(vmin=1e-5, vmax=1, clip=False)
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
cb.ax.minorticks_off()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'neutrino_sterile_dist.pdf')
fig.savefig(outdir + 'neutrino_sterile_dist.png', dpi=200)
fig.clf()
plt.close(fig)

