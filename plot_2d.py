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

outdir = './plots/standard/'

units = nsq.Const()

conv = nsq.nuSQUIDSAtm("./fluxes/conv.h5")
conv_sterile = nsq.nuSQUIDSAtm("./fluxes/conv_sin22th_0p1.h5")

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

############################
############################

flavors = (np.abs(particle) / 2 - 6).astype(int)

flux_standard = np.empty(len(props))
flux_sterile = np.empty(len(props))

for i, (ff, zz, ee, particle_type) in enumerate(zip(flavors, props["zenith"], props["energy"], props["particle"])):
    ff = int(ff)
    zz = float(zz)
    ee = float(ee)
    particle_type = int(particle_type)
    particle_type = 0 if particle_type > 0 else 1

    conv_flux = conv.EvalFlavor(ff, np.cos(zz), ee*units.GeV, particle_type)
    conv_sterile_flux = conv_sterile.EvalFlavor(ff, np.cos(zz), ee*units.GeV, particle_type)
    flux_sterile[i] = conv_sterile_flux
    flux_standard[i] = conv_flux

livetime = 365.25 * 24 * 3600

weights = flux_standard * livetime / gen_prob
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
print("All", np.sum(weights))
print("Upgoing", np.sum(weights[up_mask]))
print("Upgoing numu", np.sum(weights[red([up_mask, numu_mask])]))
print("Upgoing numubar", np.sum(weights[red([up_mask, numubar_mask])]))
print("Upgoing numubar between 100GeV-300GeV", np.sum(weights[red([up_mask, numubar_mask, one_gev_mask])]))
print("Upgoing numubar not between 100GeV-300GeV", np.sum(weights[red([up_mask, numubar_mask, one_gev_other_mask])]))
print("Upgoing numubar between 300GeV-1TeV", np.sum(weights[red([up_mask, numubar_mask, three_gev_mask])]))
print("Upgoing numubar not between 300GeV-1TeV", np.sum(weights[red([up_mask, numubar_mask, three_gev_other_mask])]))
print("Upgoing numubar between 1-2TeV", np.sum(weights[red([up_mask, numubar_mask, tev_mask])]))
print("Upgoing numubar not between 1-2TeV", np.sum(weights[red([up_mask, numubar_mask, tev_other_mask])]))
print("Upgoing numubar w/ track length > 2m", np.sum(weights[red([up_mask, numubar_mask, track_length_mask])]))
print("Upgoing numubar w/ deposited energy > 1GeV", np.sum(weights[red([up_mask, numubar_mask, deposited_energy_mask])]))
print("Upgoing numubar w/ track length > 2m and deposited energy > 1GeV", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, deposited_energy_mask])]))
print("Upgoing numubar w/ track length > 2m and deposited energy > 1GeV and between 100GeV-300GeV", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, deposited_energy_mask, one_gev_mask])]))
print("Upgoing numubar w/ track length > 2m and deposited energy > 1GeV and between 300GeV-1TeV", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, deposited_energy_mask, three_gev_mask])]))
print("Upgoing numubar w/ track length > 2m and deposited energy > 1GeV and between 1-2TeV", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, deposited_energy_mask, tev_mask])]))
print("Upgoing w/ cos(zenith) < -0.8", np.sum(weights[red([vup_mask])]))
print("Upgoing numu w/ cos(zenith) < -0.8", np.sum(weights[red([vup_mask, numu_mask])]))
print("Upgoing numubar w/ cos(zenith) < -0.8", np.sum(weights[red([vup_mask, numubar_mask])]))
print("Upgoing numubar w/ track length > 2m and deposited energy > 1GeV and between 100GeV-300GeV and cos(zenith) < -0.8", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, deposited_energy_mask, one_gev_mask, vup_mask])]))
print("Upgoing numubar w/ track length > 2m and deposited energy > 1GeV and between 300GeV-1TeV and cos(zenith) < -0.8", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, deposited_energy_mask, three_gev_mask, vup_mask])]))
print("Upgoing numubar w/ track length > 2m and deposited energy > 1GeV and between 1-2TeV and cos(zenith) < -0.8", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, deposited_energy_mask, tev_mask, vup_mask])]))

print("Upgoing through-going", np.sum(weights[red([up_mask, through_going_mask])]))
print("Upgoing through-going numu", np.sum(weights[red([up_mask, numu_mask, through_going_mask])]))
print("Upgoing through-going numubar", np.sum(weights[red([up_mask, numubar_mask, through_going_mask])]))
print("Upgoing through-going numubar between 100GeV-300GeV", np.sum(weights[red([up_mask, numubar_mask, one_gev_mask, through_going_mask])]))
print("Upgoing through-going numubar not between 100GeV-300GeV", np.sum(weights[red([up_mask, numubar_mask, one_gev_other_mask, through_going_mask])]))
print("Upgoing through-going numubar between 300GeV-1TeV", np.sum(weights[red([up_mask, numubar_mask, three_gev_mask, through_going_mask])]))
print("Upgoing through-going numubar not between 300GeV-1TeV", np.sum(weights[red([up_mask, numubar_mask, three_gev_other_mask, through_going_mask])]))
print("Upgoing through-going numubar between 1-2TeV", np.sum(weights[red([up_mask, numubar_mask, tev_mask, through_going_mask])]))
print("Upgoing through-going numubar not between 1-2TeV", np.sum(weights[red([up_mask, numubar_mask, tev_other_mask, through_going_mask])]))
print("Upgoing through-going numubar w/ track length > 2m", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, through_going_mask])]))
print("Upgoing through-going numubar w/ deposited energy > 1GeV", np.sum(weights[red([up_mask, numubar_mask, deposited_energy_mask, through_going_mask])]))
print("Upgoing through-going numubar w/ track length > 2m and deposited energy > 1GeV", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, deposited_energy_mask, through_going_mask])]))
print("Upgoing through-going numubar w/ track length > 2m and deposited energy > 1GeV and between 100GeV-300GeV", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, deposited_energy_mask, one_gev_mask, through_going_mask])]))
print("Upgoing through-going numubar w/ track length > 2m and deposited energy > 1GeV and between 300GeV-1TeV", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, deposited_energy_mask, three_gev_mask, through_going_mask])]))
print("Upgoing through-going numubar w/ track length > 2m and deposited energy > 1GeV and between 1-2TeV", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, deposited_energy_mask, tev_mask, through_going_mask])]))
print("Upgoing through-going w/ cos(zenith) < -0.8", np.sum(weights[red([vup_mask, through_going_mask])]))
print("Upgoing through-going numu w/ cos(zenith) < -0.8", np.sum(weights[red([vup_mask, numu_mask, through_going_mask])]))
print("Upgoing through-going numubar w/ cos(zenith) < -0.8", np.sum(weights[red([vup_mask, numubar_mask, through_going_mask])]))
print("Upgoing through-going numubar w/ track length > 2m and deposited energy > 1GeV and between 100GeV-300GeV and cos(zenith) < -0.8", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, deposited_energy_mask, one_gev_mask, vup_mask, through_going_mask])]))
print("Upgoing through-going numubar w/ track length > 2m and deposited energy > 1GeV and between 300GeV-1TeV and cos(zenith) < -0.8", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, deposited_energy_mask, three_gev_mask, vup_mask, through_going_mask])]))
print("Upgoing through-going numubar w/ track length > 2m and deposited energy > 1GeV and between 1-2TeV and cos(zenith) < -0.8", np.sum(weights[red([up_mask, numubar_mask, track_length_mask, deposited_energy_mask, tev_mask, vup_mask, through_going_mask])]))

#print("Expect", np.sum(weights[np.logical_and(cut_mask, m_mask)]), m, "events per year!")

print("With cuts:")
for m in np.unique(morphology):
    m_mask = morphology == m
    print("Expect", np.sum(weights[np.logical_and(cut_mask, m_mask)]), m, "events per year!")
print()

print("Without cuts:")
for m in np.unique(morphology):
    m_mask = morphology == m
    print("Expect", np.sum(weights[np.logical_and(zenith > np.pi/2., m_mask)]), m, "events per year!")
print()
#weights = weights[cut_mask]
"""

energy = energy[cut_mask]
zenith = zenith[cut_mask]
azimuth = azimuth[cut_mask]
bjorken_x = bjorken_x[cut_mask]
bjorken_y = bjorken_y[cut_mask]
final_type_0 = final_type_0[cut_mask]
final_type_1 = final_type_1[cut_mask]
particle = particle[cut_mask]
x = x[cut_mask]
y = y[cut_mask]
z = z[cut_mask]
total_column_depth = total_column_depth[cut_mask]
entry_energy = entry_energy[cut_mask]
entry_x = entry_x[cut_mask]
entry_y = entry_y[cut_mask]
entry_z = entry_z[cut_mask]
entry_zenith = entry_zenith[cut_mask]
entry_azimuth = entry_azimuth[cut_mask]
track_length = track_length[cut_mask]
entry_distance = entry_distance[cut_mask]
data = data[cut_mask]
"""
print("Expect", np.sum(weights), "events per year!")
print("Expect", np.sum(weights[energy>100]), "events per year!")

print(list(zip(energy[:100], entry_energy[:100])))
print(np.unique(final_type_0))

colors = ['#440154', '#440455', '#440655', '#440856', '#450957', '#450b57', '#450e58',
          '#451058', '#45135a', '#45145a', '#45165b', '#45175b', '#45185c', '#451a5d',
          '#461c5d', '#461c5e', '#461e5e', '#46205f', '#462160', '#462261', '#462461',
          '#462462', '#462663', '#462863', '#462864', '#462a65', '#462b65', '#462d66',
          '#462e67', '#462f67', '#462f68', '#463168', '#45336a', '#45346a', '#45346b',
          '#45366b', '#45366c', '#45386d', '#45396d', '#453a6e', '#453c6f', '#443d70',
          '#443e70', '#443e71', '#443f71', '#444072', '#444173', '#434474', '#434475',
          '#434575', '#434776', '#424776', '#424977', '#424978', '#414a78', '#414b79',
          '#414d7a', '#404e7b', '#404f7c', '#40507c', '#3f527d', '#3f537d', '#3e537e',
          '#3e557f', '#3e557f', '#3d5680', '#3d5881', '#3c5881', '#3c5a82', '#3b5b83',
          '#3a5d84', '#3a5e85', '#395f85', '#395f86', '#386187', '#376187', '#376388',
          '#366388', '#356489', '#34668a', '#34668a', '#33688b', '#32698c', '#306b8d',
          '#2f6b8e', '#2f6c8e', '#306e8d', '#316f8c', '#32708b', '#33718b', '#34718a',
          '#357389', '#357388', '#367487', '#377687', '#387785', '#387884', '#397a83',
          '#397b83', '#3a7c82', '#3a7c81', '#3b7e80', '#3b7e80', '#3c807f', '#3c807e',
          '#3c827d', '#3d827c', '#3d857b', '#3d867a', '#3e8779', '#3e8778', '#3e8977',
          '#3e8a77', '#3f8a76', '#3f8b75', '#3f8c74', '#3f8d73', '#3f9072', '#3f9071',
          '#3f9270', '#3f936f', '#3f946e', '#3f956d', '#3f966c', '#3f976c', '#3f986b',
          '#3f9a69', '#3f9a68', '#3f9b67', '#3e9d66', '#3e9e65', '#3e9f64', '#3ea063',
          '#3ea162', '#3da261', '#3da460', '#3ca45f', '#3ca65e', '#3ca75d', '#3ba75c',
          '#3ba85b', '#3aaa5a', '#3aab58', '#39ad57', '#38ae56', '#38af55', '#37b054',
          '#37b052', '#36b151', '#35b350', '#34b44e', '#33b54d', '#32b64c', '#31b74b',
          '#31b849', '#30b948', '#2ebb46', '#30bc46', '#30bd47', '#35be49', '#3abf4c',
          '#3ebf4e', '#3fc04e', '#43c051', '#47c254', '#4ac256', '#4bc356', '#4fc459',
          '#53c55c', '#55c55d', '#57c55e', '#5ac761', '#5ec763', '#5ec864', '#61c966',
          '#64c969', '#67ca6a', '#68cb6b', '#6bcc6e', '#6ecc70', '#6fcd71', '#71cd73',
          '#74ce76', '#76cf78', '#77d078', '#7ad17b', '#7dd27e', '#7ed27e', '#80d380',
          '#83d483', '#85d484', '#86d585', '#89d588', '#8cd68a', '#8cd78b', '#8fd78d',
          '#92d890', '#93d991', '#95da93', '#98da95', '#99db96', '#9bdc98', '#9ddc9b',
          '#9fdd9c', '#a0de9e', '#a3dfa0', '#a5dfa2', '#a6e0a3', '#a9e1a6', '#abe1a8',
          '#ace2a9', '#aee2ab', '#b1e3ad', '#b2e4ae', '#b4e5b1', '#b6e5b3', '#b7e6b4',
          '#bae7b6', '#bce7b9', '#bde8b9', '#bfe9bc', '#c2e9be', '#c2eabf', '#c5eac2',
          '#c7ebc4', '#c8ecc5', '#caedc7', '#cdedca', '#ceeecb', '#d0efcd', '#d2efcf',
          '#d3efd0', '#d6f1d3', '#d8f1d5', '#d8f1d6', '#dbf2d9', '#ddf3da', '#def3dc',
          '#e1f4df', '#e2f5e0', '#e4f5e2', '#e6f6e5', '#e7f7e6', '#e9f7e8', '#ecf8ea',
          '#ecf9eb', '#effaee', '#f1faef', '#f2faf1', '#f4fbf4', '#f6fcf5', '#f8fdf7',
          '#fafdf9', '#fbfefa', '#fefefd', '#ffffff']
rgb_colors = [matplotlib.colors.hex2color(c) for c in reversed(colors)]

cmap = {
        'red': [[float(i)/(len(colors)-1)] + [c[0]]*2 for i, c in enumerate(rgb_colors)],
        'green': [[float(i)/(len(colors)-1)] + [c[1]]*2 for i, c in enumerate(rgb_colors)],
        'blue': [[float(i)/(len(colors)-1)] + [c[2]]*2 for i, c in enumerate(rgb_colors)]
        }
custom_cm = matplotlib.colors.LinearSegmentedColormap('BlueGreen', cmap)

#energy_bins, zenith_bins = common.get_bins()
energy_bins = np.logspace(-1, 5, 30+1)
zenith_bins = np.arccos(np.linspace(-1,0,10+1))[::-1]

masks = common.get_bin_masks(energy, zenith, energy_bins, zenith_bins)

expect = np.array([np.sum(weights[m]) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T

m = np.max(expect)
mm = np.min(expect[expect>0])
cm = plt.get_cmap('plasma')
cm.set_under('black')
norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=10, clip=False)

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
print("neutrino_dist", np.sum(expect))
mesh = ax.pcolormesh(X,Y,expect, cmap=cm, norm=norm)
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
fig.savefig(outdir + 'neutrino_dist.pdf')
fig.savefig(outdir + 'neutrino_dist.png', dpi=200)
fig.clf()
plt.close(fig)

masks = common.get_bin_masks(energy, zenith, energy_bins, zenith_bins)

expect = np.array([np.sum(weights[np.logical_and(m, cut_mask)]) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T

m = np.max(expect)
mm = np.min(expect[expect>0])
norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=10, clip=False)

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
print("neutrino_dist", np.sum(expect))
mesh = ax.pcolormesh(X,Y,expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('High quality events per year per module')
cb.ax.minorticks_off()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'neutrino_dist_cut.pdf')
fig.savefig(outdir + 'neutrino_dist_cut.png', dpi=200)
fig.clf()
plt.close(fig)

###########

masks = common.get_bin_masks(energy, zenith, energy_bins, zenith_bins)

expect = np.array([np.sum(weights[red([m, cut_mask, starting_contained_mask])]) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T

m = np.max(expect)
mm = np.min(expect[expect>0])
norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=10, clip=False)

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
print("neutrino_dist", np.sum(expect))
mesh = ax.pcolormesh(X,Y,expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,1))
ax.set_ylabel('Neutrino Energy [GeV]')
ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('High quality events per year per module')
cb.ax.minorticks_off()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'starting_neutrino_dist_cut.pdf')
fig.savefig(outdir + 'starting_neutrino_dist_cut.png', dpi=200)
fig.clf()
plt.close(fig)

###########

masks = common.get_bin_masks(muon_start_energy, muon_start_zenith, energy_bins, zenith_bins)

expect = np.array([np.sum(weights[m]) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T

m = np.max(expect)
mm = np.min(expect[expect>0])
norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=10, clip=False)

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
print("muon_dist", np.sum(expect))
mesh = ax.pcolormesh(X,Y,expect, cmap=cm, norm=norm)
ax.set_yscale('log')
#ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,0))
ax.set_ylabel('Initial muon energy in detector [GeV]')
ax.set_xlabel(r'Initial muon $\cos\left(\theta_z\right)$ in detector')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Events per year per module')
cb.ax.minorticks_off()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'muon_dist.pdf')
fig.savefig(outdir + 'muon_dist.png', dpi=200)
fig.clf()
plt.close(fig)

masks = common.get_bin_masks(muon_start_energy, muon_start_zenith, energy_bins, zenith_bins)

expect = np.array([np.sum(weights[np.logical_and(m, cut_mask)]) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T

m = np.max(expect)
mm = np.min(expect[expect>0])
norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=10, clip=False)

fig, ax = plt.subplots(figsize=(7,5))
X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
Y = np.array([energy_bins]*(len(zenith_bins))).T
print("muon_dist", np.sum(expect))
mesh = ax.pcolormesh(X,Y,expect, cmap=cm, norm=norm)
ax.set_yscale('log')
#ax.set_ylim((1e2, 1e5))
#ax.set_xlim((-1,0))
ax.set_ylabel('Initial muon energy in detector [GeV]')
ax.set_xlabel(r'Initial muon $\cos\left(\theta_z\right)$ in detector')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('High quality events per year per module')
cb.ax.minorticks_off()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'muon_dist_cut.pdf')
fig.savefig(outdir + 'muon_dist_cut.png', dpi=200)
fig.clf()
plt.close(fig)

length_bins = np.logspace(0, 5, 20+1)
masks = common.get_bin_masks(energy, entry_distance, energy_bins, length_bins)

expect = np.array([np.sum(weights[np.logical_and(m, cut_mask)]) for m in masks]).reshape((len(length_bins)-1, len(energy_bins)-1)).T

m = np.max(expect)
mm = np.min(expect[expect>0])
norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=10, clip=False)

fig, ax = plt.subplots(figsize=(7,5))
X = np.array([length_bins]*(len(energy_bins)))
Y = np.array([energy_bins]*(len(length_bins))).T
print("neutrino_energy_distance", np.sum(expect))
mesh = ax.pcolormesh(X,Y,expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim((1e2, 1e5))
ax.set_xlim((1e0,1e5))
ax.set_ylabel('Neutrino energy [GeV]')
ax.set_xlabel(r'Distance to entry [m]')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('High quality events per year per module')
cb.ax.minorticks_off()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'neutrino_energy_distance.pdf')
fig.savefig(outdir + 'neutrino_energy_distance.png', dpi=200)
fig.clf()
plt.close(fig)

length_bins = np.logspace(-3, 3, 20+1)
masks = common.get_bin_masks(muon_start_energy, track_length, energy_bins, length_bins)

expect = np.array([np.sum(weights[m]) for m in masks]).reshape((len(length_bins)-1, len(energy_bins)-1)).T

m = np.max(expect)
mm = np.min(expect[expect>0])
norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=10, clip=False)

fig, ax = plt.subplots(figsize=(7,5))
X = np.array([length_bins]*(len(energy_bins)))
Y = np.array([energy_bins]*(len(length_bins))).T
print("muon_energy_geometric_length", np.sum(expect))
mesh = ax.pcolormesh(X,Y,expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim((1e-3,1e3))
ax.set_ylabel('Initial muon energy in detector [GeV]')
ax.set_xlabel(r'Path length in detector [m]')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Events per year per module')
cb.ax.minorticks_off()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'muon_energy_length.pdf')
fig.savefig(outdir + 'muon_energy_length.png', dpi=200)
fig.clf()
plt.close(fig)

length_bins = np.logspace(-3, 3, 20+1)
masks = common.get_bin_masks(deposited_energy, track_length, energy_bins, length_bins)

expect = np.array([np.sum(weights[m]) for m in masks]).reshape((len(length_bins)-1, len(energy_bins)-1)).T

m = np.max(expect)
mm = np.min(expect[expect>0])
norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=10, clip=False)

fig, ax = plt.subplots(figsize=(7,5))
X = np.array([length_bins]*(len(energy_bins)))
Y = np.array([energy_bins]*(len(length_bins))).T
print("muon_energy_geometric_length", np.sum(expect))
mesh = ax.pcolormesh(X,Y,expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim((1e-3,1e3))
ax.set_ylabel('Deposited energy [GeV]')
ax.set_xlabel(r'Path length in detector [m]')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('Events per year per module')
cb.ax.minorticks_off()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'muon_dep_energy_length.pdf')
fig.savefig(outdir + 'muon_dep_energy_length.png', dpi=200)
fig.clf()
plt.close(fig)

masks = common.get_bin_masks(muon_start_energy, energy, energy_bins, energy_bins)

expect = np.array([np.sum(weights[np.logical_and(cut_mask, m)]) for m in masks]).reshape((len(energy_bins)-1, len(energy_bins)-1)).T

m = np.max(expect)
mm = np.min(expect[expect>0])
norm = matplotlib.colors.LogNorm()

fig, ax = plt.subplots(figsize=(7,5))
X = np.array([energy_bins]*(len(energy_bins)))
Y = np.array([energy_bins]*(len(energy_bins))).T
print("muon_neutrino_energy", np.sum(expect))
mesh = ax.pcolormesh(X,Y,expect, cmap=cm, norm=norm)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim((1e2, 1e5))
ax.set_xlim((1e2, 1e5))
ax.set_ylabel('Muon entry energy [GeV]')
ax.set_xlabel(r'Neutrino energy [GeV]')
cb = fig.colorbar(mesh, ax=ax)
cb.ax.set_ylabel('High quality events per year per module')
cb.ax.minorticks_off()
plt.tight_layout()
font = FontProperties()
font.set_size('medium')
font.set_family('sans-serif')
font.set_weight('bold')
fig.savefig(outdir + 'muon_neutrino_energy.pdf')
fig.savefig(outdir + 'muon_neutrino_energy.png', dpi=200)
fig.clf()
plt.close(fig)


fig, ax = plt.subplots(figsize=(7,5))
energy_mapping = np.digitize(energy, bins=energy_bins) - 1
energy_masks = [energy_mapping == i for i in range(len(energy_bins)-1)]
for i, (energy_center, label) in enumerate(zip([100, 500, 1e3, 2e3], ["100GeV", "200GeV", "1TeV", "2TeV"])):
    mu_energy_mask = np.logical_and(muon_start_energy >= 0.9*energy_center, muon_start_energy <= 1.1*energy_center)
    slice_masks = [np.logical_and(emask, mu_energy_mask) for emask in energy_masks]
    expect = [np.sum(weights[np.logical_and(cut_mask, smask)]) for smask in slice_masks]
    center = lambda xx: (xx[:-1] + xx[1:]) / 2.0
    x = center(energy_bins)
    ax.hist(x, bins=energy_bins, weights=expect, histtype='step', color=cm(0.1 + 0.8/3*i), label=label)
ax.set_ylabel('High quality events per year per module')
ax.set_xlabel(r'Neutrino energy [GeV]')
ax.set_xscale('log')
ax.legend()
fig.tight_layout()
fig.savefig(outdir + 'muon_energy_slices.pdf')
fig.savefig(outdir + 'muon_energy_slices.png', dpi=200)
fig.clf()
plt.close(fig)

fig, ax = plt.subplots(figsize=(7,5))
energy_mapping = np.digitize(energy, bins=energy_bins) - 1
energy_masks = [energy_mapping == i for i in range(len(energy_bins)-1)]
slice_bins = np.logspace(0, 4, 8+1)
slice_mapping = np.digitize(muon_start_energy, bins=slice_bins) - 1
slice_masks = [slice_mapping == i for i in range(len(slice_bins)-1)]

slice_centers = center(slice_bins)
unit_labels = []
for _energy in slice_bins:
    unit = int(np.floor(np.log10(_energy) / 3.0))
    unit_label = ['GeV', 'TeV', 'PeV'][unit]
    unit_scale = 10.0**(unit*3.0)
    unit_label = '%.1f%s' % (_energy / unit_scale, unit_label)
    unit_labels.append(unit_label)

for i in range(len(slice_centers)):
    mu_energy_mask = slice_masks[i]
    emasks = [np.logical_and(emask, mu_energy_mask) for emask in energy_masks]
    expect = [np.sum(weights[np.logical_and(cut_mask, smask)]) for smask in emasks]
    x = center(energy_bins)
    label = unit_labels[i] + " - " + unit_labels[i+1]
    ax.hist(x, bins=energy_bins, weights=expect, histtype='step', color=cm(0.1 + 0.8/(len(slice_bins)-1)*i), label=label)

ax.set_ylabel('High quality events per year per module')
ax.set_xlabel(r'Neutrino energy [GeV]')
ax.set_xscale('log')
ax.set_xlim(1e2, 1e5)
ax.legend()
fig.tight_layout()
fig.savefig(outdir + 'muon_energy_slices_integral.pdf')
fig.savefig(outdir + 'muon_energy_slices_integral.png', dpi=200)
fig.clf()
plt.close(fig)

fig, ax = plt.subplots(figsize=(7,5))
energy_mapping = np.digitize(muon_start_energy, bins=energy_bins) - 1
energy_masks = [energy_mapping == i for i in range(len(energy_bins)-1)]
slice_bins = np.logspace(2, 5, 6+1  )
slice_mapping = np.digitize(energy, bins=slice_bins) - 1
slice_masks = [slice_mapping == i for i in range(len(slice_bins)-1)]

slice_centers = center(slice_bins)
unit_labels = []
for _energy in slice_bins:
    unit = int(np.floor(np.log10(_energy) / 3.0))
    unit_label = ['GeV', 'TeV', 'PeV'][unit]
    unit_scale = 10.0**(unit*3.0)
    unit_label = '%.1f%s' % (_energy / unit_scale, unit_label)
    unit_labels.append(unit_label)

for i in range(len(slice_centers)):
    energy_mask = slice_masks[i]
    emasks = [np.logical_and(emask, energy_mask) for emask in energy_masks]
    expect = [np.sum(weights[np.logical_and(cut_mask, smask)]) for smask in emasks]
    x = center(energy_bins)
    label = unit_labels[i] + " - " + unit_labels[i+1]
    ax.hist(x, bins=energy_bins, weights=expect, histtype='step', color=cm(0.1 + 0.8/(len(slice_bins)-1)*i), label=label)

ax.set_ylabel('High quality events per year per module')
ax.set_xlabel(r'Initial muon energy in detector [GeV]')
ax.set_xscale('log')
ax.set_xlim(1e0, 1e4)
ax.legend()
fig.tight_layout()
fig.savefig(outdir + 'neutrino_energy_slices_integral.pdf')
fig.savefig(outdir + 'neutrino_energy_slices_integral.png', dpi=200)
fig.clf()
plt.close(fig)

fig, ax = plt.subplots(figsize=(7,5))
energy_mapping = np.digitize(muon_start_energy, bins=energy_bins) - 1
energy_masks = [energy_mapping == i for i in range(len(energy_bins)-1)]
slice_bins = np.logspace(2, 5, 6+1  )
slice_mapping = np.digitize(energy, bins=slice_bins) - 1
slice_masks = [slice_mapping == i for i in range(len(slice_bins)-1)]

slice_centers = center(slice_bins)
unit_labels = []
for _energy in slice_bins:
    unit = int(np.floor(np.log10(_energy) / 3.0))
    unit_label = ['GeV', 'TeV', 'PeV'][unit]
    unit_scale = 10.0**(unit*3.0)
    unit_label = '%.1f%s' % (_energy / unit_scale, unit_label)
    unit_labels.append(unit_label)

expect_tot = np.zeros(len(energy_masks))
expectations = []
for i in range(len(slice_centers)):
    energy_mask = slice_masks[i]
    emasks = [np.logical_and(emask, energy_mask) for emask in energy_masks]
    expect = [np.sum(weights[np.logical_and(cut_mask, smask)]) for smask in emasks]
    new_expect = expect_tot + np.array(expect)
    expectations.append(new_expect)
    expect_tot = new_expect
for i in reversed(range(len(slice_centers))):
    x = center(energy_bins)
    label = unit_labels[i] + " - " + unit_labels[i+1]
    ax.hist(x, bins=energy_bins, weights=expectations[i], histtype='bar', lw = 2, label=label, color=cm(0.1 + 0.8/(len(slice_bins)-1)*i))
    #ax.hist(x, bins=energy_bins, weights=expect, histtype='step', color=cm(0.1 + 0.8/(len(slice_bins)-1)*i), label=label)

ax.set_ylabel('High quality events per year per module')
ax.set_xlabel(r'Initial muon energy in detector [GeV]')
ax.set_xscale('log')
ax.set_xlim(1e0, 1e4)
ax.legend()
fig.tight_layout()
fig.savefig(outdir + 'neutrino_energy_slices_integral_stacked.pdf')
fig.savefig(outdir + 'neutrino_energy_slices_integral_stacked.png', dpi=200)
fig.clf()
plt.close(fig)



upgoing_m08_mask = np.logical_and(cut_mask, np.cos(zenith) < -0.8)

fig, ax = plt.subplots(figsize=(7,5))
energy_mapping = np.digitize(muon_start_energy, bins=energy_bins) - 1
energy_masks = [energy_mapping == i for i in range(len(energy_bins)-1)]
slice_bins = np.logspace(2, 5, 6+1  )
slice_mapping = np.digitize(energy, bins=slice_bins) - 1
slice_masks = [slice_mapping == i for i in range(len(slice_bins)-1)]

slice_centers = center(slice_bins)
unit_labels = []
for _energy in slice_bins:
    unit = int(np.floor(np.log10(_energy) / 3.0))
    unit_label = ['GeV', 'TeV', 'PeV'][unit]
    unit_scale = 10.0**(unit*3.0)
    unit_label = '%.1f%s' % (_energy / unit_scale, unit_label)
    unit_labels.append(unit_label)

expect_tot = np.zeros(len(energy_masks))
expectations = []
for i in range(len(slice_centers)):
    energy_mask = slice_masks[i]
    emasks = [np.logical_and(emask, energy_mask) for emask in energy_masks]
    expect = [np.sum(weights[np.logical_and(upgoing_m08_mask, smask)]) for smask in emasks]
    new_expect = expect_tot + np.array(expect)
    expectations.append(new_expect)
    expect_tot = new_expect
for i in reversed(range(len(slice_centers))):
    x = center(energy_bins)
    label = unit_labels[i] + " - " + unit_labels[i+1]
    ax.hist(x, bins=energy_bins, weights=expectations[i], histtype='bar', lw = 2, label=label, color=cm(0.1 + 0.8/(len(slice_bins)-1)*i))
    #ax.hist(x, bins=energy_bins, weights=expect, histtype='step', color=cm(0.1 + 0.8/(len(slice_bins)-1)*i), label=label)

ax.set_ylabel('High quality events per year per module')
ax.set_xlabel(r'Initial muon energy in detector [GeV]')
ax.set_xscale('log')
ax.set_xlim(1e0, 1e4)
ax.legend()
fig.tight_layout()
fig.savefig(outdir + 'neutrino_energy_slices_integral_stacked_upm08.pdf')
fig.savefig(outdir + 'neutrino_energy_slices_integral_stacked_upm08.png', dpi=200)
fig.clf()
plt.close(fig)

upgoing_m08_nop5t1TeVanm_mask = np.logical_and(upgoing_m08_mask, np.logical_or(particle > 0, np.logical_or(energy <= 5e2, energy >= 1e3)))

fig, ax = plt.subplots(figsize=(7,5))
energy_mapping = np.digitize(muon_start_energy, bins=energy_bins) - 1
energy_masks = [energy_mapping == i for i in range(len(energy_bins)-1)]
slice_bins = np.logspace(2, 5, 6+1  )
slice_mapping = np.digitize(energy, bins=slice_bins) - 1
slice_masks = [slice_mapping == i for i in range(len(slice_bins)-1)]

slice_centers = center(slice_bins)
unit_labels = []
for _energy in slice_bins:
    unit = int(np.floor(np.log10(_energy) / 3.0))
    unit_label = ['GeV', 'TeV', 'PeV'][unit]
    unit_scale = 10.0**(unit*3.0)
    unit_label = '%.1f%s' % (_energy / unit_scale, unit_label)
    unit_labels.append(unit_label)

expect_tot = np.zeros(len(energy_masks))
expectations = []
for i in range(len(slice_centers)):
    energy_mask = slice_masks[i]
    emasks = [np.logical_and(emask, energy_mask) for emask in energy_masks]
    expect = [np.sum(weights[np.logical_and(upgoing_m08_nop5t1TeVanm_mask, smask)]) for smask in emasks]
    new_expect = expect_tot + np.array(expect)
    expectations.append(new_expect)
    expect_tot = new_expect
for i in reversed(range(len(slice_centers))):
    x = center(energy_bins)
    label = unit_labels[i] + " - " + unit_labels[i+1]
    ax.hist(x, bins=energy_bins, weights=expectations[i], histtype='bar', lw = 2, label=label, color=cm(0.1 + 0.8/(len(slice_bins)-1)*i))
    #ax.hist(x, bins=energy_bins, weights=expect, histtype='step', color=cm(0.1 + 0.8/(len(slice_bins)-1)*i), label=label)

ax.set_ylabel('High quality events per year per module')
ax.set_xlabel(r'Initial muon energy in detector [GeV]')
ax.set_xscale('log')
ax.set_xlim(1e0, 1e4)
ax.legend()
fig.tight_layout()
fig.savefig(outdir + 'neutrino_energy_slices_integral_stacked_upm08_nop5t1TeVanm.pdf')
fig.savefig(outdir + 'neutrino_energy_slices_integral_stacked_upm08_nop5t1TeVanm.png', dpi=200)
fig.clf()
plt.close(fig)

fig, ax = plt.subplots(figsize=(7,5))
energy_mapping = np.digitize(muon_start_energy, bins=energy_bins) - 1
energy_masks = [energy_mapping == i for i in range(len(energy_bins)-1)]
tot_expect = np.array([np.sum(weights[np.logical_and(upgoing_m08_mask, emask)]) for emask in energy_masks])
m_expect = np.array([np.sum(weights[np.logical_and(upgoing_m08_nop5t1TeVanm_mask, emask)]) for emask in energy_masks])
diff = tot_expect - m_expect
print("Deficit", np.sum(diff))
x = center(energy_bins)
ax.hist(x, bins=energy_bins, weights=diff, histtype='step', lw = 2, color='blue')

ax.set_ylabel('Event deficit per year per module')
ax.set_xlabel(r'Initial muon energy in detector [GeV]')
ax.set_xscale('log')
ax.set_xlim(1e0, 1e4)
ax.legend()
fig.tight_layout()
fig.savefig(outdir + 'neutrino_energy_slices_integral_stacked_upm08_nop5t1TeVanm_diff.pdf')
fig.savefig(outdir + 'neutrino_energy_slices_integral_stacked_upm08_nop5t1TeVanm_diff.png', dpi=200)
fig.clf()
plt.close(fig)



##########
# Starting and contained events
##########

fig, ax = plt.subplots(figsize=(7,5))
energy_mapping = np.digitize(muon_start_energy, bins=energy_bins) - 1
energy_masks = [energy_mapping == i for i in range(len(energy_bins)-1)]
slice_bins = np.logspace(2, 5, 6+1  )
slice_mapping = np.digitize(energy, bins=slice_bins) - 1
slice_masks = [slice_mapping == i for i in range(len(slice_bins)-1)]

slice_centers = center(slice_bins)
unit_labels = []
for _energy in slice_bins:
    unit = int(np.floor(np.log10(_energy) / 3.0))
    unit_label = ['GeV', 'TeV', 'PeV'][unit]
    unit_scale = 10.0**(unit*3.0)
    unit_label = '%.1f%s' % (_energy / unit_scale, unit_label)
    unit_labels.append(unit_label)

for i in range(len(slice_centers)):
    expect = [np.sum(weights[red([cut_mask, emask, slice_masks[i], starting_contained_mask])]) for emask in energy_masks]
    x = center(energy_bins)
    label = unit_labels[i] + " - " + unit_labels[i+1]
    ax.hist(x, bins=energy_bins, weights=expect, histtype='step', color=cm(0.1 + 0.8/(len(slice_bins)-1)*i), label=label)

ax.set_ylabel('High quality events per year per module')
ax.set_xlabel(r'Initial muon energy in detector [GeV]')
ax.set_xscale('log')
ax.set_xlim(1e0, 1e4)
ax.legend()
fig.tight_layout()
fig.savefig(outdir + 'starting_neutrino_energy_slices_integral.pdf')
fig.savefig(outdir + 'starting_neutrino_energy_slices_integral.png', dpi=200)
fig.clf()
plt.close(fig)

fig, ax = plt.subplots(figsize=(7,5))
energy_mapping = np.digitize(muon_start_energy, bins=energy_bins) - 1
energy_masks = [energy_mapping == i for i in range(len(energy_bins)-1)]
slice_bins = np.logspace(2, 5, 6+1  )
slice_mapping = np.digitize(energy, bins=slice_bins) - 1
slice_masks = [slice_mapping == i for i in range(len(slice_bins)-1)]

slice_centers = center(slice_bins)
unit_labels = []
for _energy in slice_bins:
    unit = int(np.floor(np.log10(_energy) / 3.0))
    unit_label = ['GeV', 'TeV', 'PeV'][unit]
    unit_scale = 10.0**(unit*3.0)
    unit_label = '%.1f%s' % (_energy / unit_scale, unit_label)
    unit_labels.append(unit_label)

expect_tot = np.zeros(len(energy_masks))
expectations = []
for i in range(len(slice_centers)):
    expect = [np.sum(weights[red([cut_mask, emask, slice_masks[i], starting_contained_mask])]) for emask in energy_masks]
    new_expect = expect_tot + np.array(expect)
    expectations.append(new_expect)
    expect_tot = new_expect
for i in reversed(range(len(slice_centers))):
    x = center(energy_bins)
    label = unit_labels[i] + " - " + unit_labels[i+1]
    ax.hist(x, bins=energy_bins, weights=expectations[i], histtype='bar', lw = 2, label=label, color=cm(0.1 + 0.8/(len(slice_bins)-1)*i))
    #ax.hist(x, bins=energy_bins, weights=expect, histtype='step', color=cm(0.1 + 0.8/(len(slice_bins)-1)*i), label=label)

ax.set_ylabel('High quality events per year per module')
ax.set_xlabel(r'Initial muon energy in detector [GeV]')
ax.set_xscale('log')
ax.set_xlim(1e0, 1e4)
ax.legend()
fig.tight_layout()
fig.savefig(outdir + 'starting_neutrino_energy_slices_integral_stacked.pdf')
fig.savefig(outdir + 'starting_neutrino_energy_slices_integral_stacked.png', dpi=200)
fig.clf()
plt.close(fig)



upgoing_m08_mask = np.logical_and(cut_mask, np.cos(zenith) < -0.8)

fig, ax = plt.subplots(figsize=(7,5))
energy_mapping = np.digitize(muon_start_energy, bins=energy_bins) - 1
energy_masks = [energy_mapping == i for i in range(len(energy_bins)-1)]
slice_bins = np.logspace(2, 5, 6+1  )
slice_mapping = np.digitize(energy, bins=slice_bins) - 1
slice_masks = [slice_mapping == i for i in range(len(slice_bins)-1)]

slice_centers = center(slice_bins)
unit_labels = []
for _energy in slice_bins:
    unit = int(np.floor(np.log10(_energy) / 3.0))
    unit_label = ['GeV', 'TeV', 'PeV'][unit]
    unit_scale = 10.0**(unit*3.0)
    unit_label = '%.1f%s' % (_energy / unit_scale, unit_label)
    unit_labels.append(unit_label)

expect_tot = np.zeros(len(energy_masks))
expectations = []
for i in range(len(slice_centers)):
    expect = [np.sum(weights[red([upgoing_m08_mask, emask, slice_masks[i], starting_contained_mask])]) for emask in energy_masks]
    new_expect = expect_tot + np.array(expect)
    expectations.append(new_expect)
    expect_tot = new_expect
for i in reversed(range(len(slice_centers))):
    x = center(energy_bins)
    label = unit_labels[i] + " - " + unit_labels[i+1]
    ax.hist(x, bins=energy_bins, weights=expectations[i], histtype='bar', lw = 2, label=label, color=cm(0.1 + 0.8/(len(slice_bins)-1)*i))
    #ax.hist(x, bins=energy_bins, weights=expect, histtype='step', color=cm(0.1 + 0.8/(len(slice_bins)-1)*i), label=label)

ax.set_ylabel('High quality events per year per module')
ax.set_xlabel(r'Initial muon energy in detector [GeV]')
ax.set_xscale('log')
ax.set_xlim(1e0, 1e4)
ax.legend()
fig.tight_layout()
fig.savefig(outdir + 'starting_neutrino_energy_slices_integral_stacked_upm08.pdf')
fig.savefig(outdir + 'starting_neutrino_energy_slices_integral_stacked_upm08.png', dpi=200)
fig.clf()
plt.close(fig)

upgoing_m08_nop5t1TeVanm_mask = np.logical_and(upgoing_m08_mask, np.logical_or(particle > 0, np.logical_or(energy <= 5e2, energy >= 1e3)))

fig, ax = plt.subplots(figsize=(7,5))
energy_mapping = np.digitize(muon_start_energy, bins=energy_bins) - 1
energy_masks = [energy_mapping == i for i in range(len(energy_bins)-1)]
slice_bins = np.logspace(2, 5, 6+1  )
slice_mapping = np.digitize(energy, bins=slice_bins) - 1
slice_masks = [slice_mapping == i for i in range(len(slice_bins)-1)]

slice_centers = center(slice_bins)
unit_labels = []
for _energy in slice_bins:
    unit = int(np.floor(np.log10(_energy) / 3.0))
    unit_label = ['GeV', 'TeV', 'PeV'][unit]
    unit_scale = 10.0**(unit*3.0)
    unit_label = '%.1f%s' % (_energy / unit_scale, unit_label)
    unit_labels.append(unit_label)

expect_tot = np.zeros(len(energy_masks))
expectations = []
for i in range(len(slice_centers)):
    expect = [np.sum(weights[red([upgoing_m08_nop5t1TeVanm_mask, emask, slice_masks[i], starting_contained_mask])]) for emask in energy_masks]
    new_expect = expect_tot + np.array(expect)
    expectations.append(new_expect)
    expect_tot = new_expect
for i in reversed(range(len(slice_centers))):
    x = center(energy_bins)
    label = unit_labels[i] + " - " + unit_labels[i+1]
    ax.hist(x, bins=energy_bins, weights=expectations[i], histtype='bar', lw = 2, label=label, color=cm(0.1 + 0.8/(len(slice_bins)-1)*i))
    #ax.hist(x, bins=energy_bins, weights=expect, histtype='step', color=cm(0.1 + 0.8/(len(slice_bins)-1)*i), label=label)

ax.set_ylabel('High quality events per year per module')
ax.set_xlabel(r'Initial muon energy in detector [GeV]')
ax.set_xscale('log')
ax.set_xlim(1e0, 1e4)
ax.legend()
fig.tight_layout()
fig.savefig(outdir + 'starting_neutrino_energy_slices_integral_stacked_upm08_nop5t1TeVanm.pdf')
fig.savefig(outdir + 'starting_neutrino_energy_slices_integral_stacked_upm08_nop5t1TeVanm.png', dpi=200)
fig.clf()
plt.close(fig)

fig, ax = plt.subplots(figsize=(7,5))
energy_mapping = np.digitize(muon_start_energy, bins=energy_bins) - 1
energy_masks = [energy_mapping == i for i in range(len(energy_bins)-1)]
tot_expect = np.array([np.sum(weights[red([upgoing_m08_mask, starting_contained_mask, emask])]) for emask in energy_masks])
m_expect = np.array([np.sum(weights[red([upgoing_m08_nop5t1TeVanm_mask, starting_contained_mask, emask])]) for emask in energy_masks])
diff = tot_expect - m_expect
print("Deficit", np.sum(diff))
x = center(energy_bins)
ax.hist(x, bins=energy_bins, weights=diff, histtype='step', lw = 2, color='blue')

ax.set_ylabel('Event deficit per year per module')
ax.set_xlabel(r'Initial muon energy in detector [GeV]')
ax.set_xscale('log')
ax.set_xlim(1e0, 1e4)
ax.legend()
fig.tight_layout()
fig.savefig(outdir + 'starting_neutrino_energy_slices_integral_stacked_upm08_nop5t1TeVanm_diff.pdf')
fig.savefig(outdir + 'starting_neutrino_energy_slices_integral_stacked_upm08_nop5t1TeVanm_diff.png', dpi=200)
fig.clf()
plt.close(fig)


fig, ax = plt.subplots(figsize=(7,5))
energy_mapping = np.digitize(energy, bins=energy_bins) - 1
energy_masks = [energy_mapping == i for i in range(len(energy_bins)-1)]
expect = np.array([np.sum(weights[red([starting_contained_mask, emask, cut_mask])]) for emask in energy_masks])
x = center(energy_bins)
ax.hist(x, bins=energy_bins, weights=expect, histtype='step', lw = 2, color='blue')

ax.set_ylabel('High quality starting events (all upgoing)')
ax.set_xlabel(r'Neutrino energy [GeV]')
ax.set_xscale('log')
ax.set_xlim(1e2, 1e5)
ax.legend()
fig.tight_layout()
fig.savefig(outdir + 'starting_neutrino_energy.pdf')
fig.savefig(outdir + 'starting_neutrino_energy.png', dpi=200)
fig.clf()
plt.close(fig)
