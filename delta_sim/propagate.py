import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import sys
import os
import os.path
base_path = os.environ['GOLEMSPACE']
table_path = base_path + '/local/share/PROPOSAL/tables'
config_path = base_path + '/sources/DUNEAtmo/proposal_config/'
import proposal as pp
import numpy as np
import h5py as h5
import LWpy
import LeptonInjector
import json
import sim_tools
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Propagate muons")
parser.add_argument('--config',
        type=str,
        dest='config',
        default='./lar_config.json')
#parser.add_argument('--energy',
#        type=float,
#        dest='energy',
#        required=True
#        )
parser.add_argument('--output',
        type=str,
        dest='output',
        required=True
        )
args = parser.parse_args()

interpolation_def = pp.InterpolationDef()
interpolation_def.path_to_tables = table_path
interpolation_def.path_to_tables_readonly = table_path

mu_minus_def = pp.particle.MuMinusDef()
prop_mu_minus = pp.Propagator(particle_def=mu_minus_def, config_file=args.config)

mu_plus_def = pp.particle.MuPlusDef()
prop_mu_plus = pp.Propagator(particle_def=mu_plus_def, config_file=args.config)

def make_p(e):
    type = 13
    position = pp.Vector3D(0,0,0)
    zenith = np.pi/2.0
    azimuth = 0.0
    nx = np.sin(zenith)*np.cos(azimuth)
    ny = np.sin(zenith)*np.sin(azimuth)
    nz = np.cos(zenith)
    direction = pp.Vector3D(nx, ny, nz)

    p = pp.particle.DynamicData(type)
    p.position = position
    p.direction = direction
    p.energy = e * 1e3
    p.time = 0
    p.propagated_distance = 0
    return p


max_distance = 15 * 1e2

n_muons = int(1e6)

particle_types = {
    "Particle"               : 1000000001,
    "Brems"                  : 1000000002,
    "DeltaE"                 : 1000000003,
    "Epair"                  : 1000000004,
    "NuclInt"                : 1000000005,
    "MuPair"                 : 1000000006,
    "Hadrons"                : 1000000007,
    "ContinuousEnergyLoss"   : 1000000008,
    "WeakInt"                : 1000000009,
    "Compton"                : 1000000010,
    "Decay"                  : 1000000011,
    ""                       : 0,
}
is_energy_loss = lambda p: int(p.type) > 1000000000 and int(p.type) < 1000000012

mu_part = {"particle": 13}


energies = np.logspace(2, 3, 10+1)
per_mu_slices_by_primary = dict()
all_energies_by_primary = dict()

cm = plt.get_cmap("plasma")

min_loss_energy = 50 / 1e3
min_loss_energy = 50 / 1e3
max_loss_energy = max(energies)
loss_energy_bins = np.logspace(np.log10(min_loss_energy), np.log10(max_loss_energy), int((np.log10(max_loss_energy) - np.log10(min_loss_energy)) * 3)+1)

for c_i,energy in enumerate(energies):
    per_mu_slices = dict()
    for key in particle_types.keys():
        per_mu_slices[key] = []
    all_energies = dict()
    for key in particle_types.keys():
        all_energies[key] = []

    for i in tqdm(range(n_muons)):
        pp_part = make_p(energy)
        if mu_part["particle"] == 13:
            secondaries = prop_mu_minus.propagate(pp_part, max_distance_cm=max_distance, minimal_energy=100)
        elif mu_part["particle"] == -13:
            secondaries = prop_mu_plus.propagate(pp_part, max_distance_cm=max_distance, minimal_energy=100)
        particles = secondaries.particles
        decay_products = [p for i,p in zip(range(max(len(particles)-3,0),len(particles)), particles[-3:]) if int(p.type) <= 1000000001]
        if len(decay_products) == 0:
            pass
        else:
            particles = particles[:-len(decay_products)]
        particle_energies = [energy] + [p.energy/1e3 for p in particles]
        particle_energies = np.array(particle_energies)
        assert(np.all(particle_energies > 0))
        assert(np.all(particle_energies[:-1] > particle_energies[1:]))
        particle_energies = -np.diff(particle_energies)

        key = ""
        key_energies = [float(e) for e, particle in zip(particle_energies,particles) if is_energy_loss(particle)]
        k = len(per_mu_slices)
        per_mu_slices[key].append(slice(k, k + len(key_energies)))
        all_energies[key].extend(key_energies)

        for key in particle_types.keys():
            key_energies = [float(e) for e, particle in zip(particle_energies,particles) if particle.type == particle_types[key] and is_energy_loss(particle)]
            k = len(per_mu_slices)
            per_mu_slices[key].append(slice(k, k + len(key_energies)))
            all_energies[key].extend(key_energies)
        del particles

    for key in particle_types.keys():
        all_energies[key] = np.array(all_energies[key])

    per_mu_slices_by_primary[energy] = per_mu_slices
    all_energies_by_primary[energy] = all_energies

for key in particle_types.keys():
    fig, ax = plt.subplots(figsize=(7,5))
    print(key)
    for c_i,energy in enumerate(energies):
        color = cm(c_i/(max(len(energies)-1, 1)))
        per_mu_slices = per_mu_slices_by_primary[energy]
        all_energies = all_energies_by_primary[energy]
        print(energy, np.sum(all_energies[key])/len(all_energies[key]))
        mapping = np.digitize(all_energies[key], bins=loss_energy_bins) - 1
        masks = mapping[None,:] == np.arange(len(loss_energy_bins) - 1)[:,None]
        #key_energies = [[all_energies[key][s][mask] for mask in masks[:,s]] for s in per_mu_slices[key]]
        hists = np.array([[np.sum(mask) for mask in masks[:,s]] for s in per_mu_slices[key]])
        all_hist = np.sum(hists, axis=0)/len(per_mu_slices[key])

        center = lambda x: (x[:-1] + x[1:]) / 2.0
        energy_centers = center(loss_energy_bins)

        ax.hist(energy_centers, weights=all_hist, bins=loss_energy_bins, histtype='step', label=str(energy), color=color)
        ax.set_xlabel(r"Stochastic loss energy [GeV]")
        ax.set_ylabel(r"Average losses per muon in 15 m")
        ax.set_title(key)
        ax.set_xscale("log")
        ax.legend()
        fig.savefig(args.output + "_" + key + ".png", dpi=200)
