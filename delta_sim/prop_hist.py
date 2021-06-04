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
        default=config_path + 'config.json')
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

def make_hist(energies, positions, energy_bins, pos_bins):
    mapping_e = np.digitize(energies, bins=energy_bins) - 1
    mapping_p = np.digitize(positions, bins=pos_bins) - 1
    masks_e = mapping_e[None,:] == np.arange(0,len(energy_bins)-1)[:,None]
    masks_p = mapping_p[None,:] == np.arange(0,len(pos_bins)-1)[:,None]
    return np.count_nonzero(np.logical_and(masks_e[:,:,None], masks_p[:,None,:]), axis=0)

max_distance = 62 * 1e2

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


energies = np.logspace(1, 5, 40+1)
loss_hists_by_primary = dict()

cm = plt.get_cmap("plasma")

min_loss_energy = 50 / 1e3
max_loss_energy = max(energies)
loss_energy_bins = np.logspace(np.log10(min_loss_energy), np.log10(max_loss_energy), int((np.log10(max_loss_energy) - np.log10(min_loss_energy)) * 10)+1)
pos_bins = np.linspace(0,62,62+1)

nE = len(loss_energy_bins)
nP = len(pos_bins)
nEm1 = nE-1
nPm1 = nP-1

max_data_len = 2000

def make_loss_hist(energies, positions):
    mapping_e = np.digitize(energies, bins=loss_energy_bins) - 1
    mapping_p = np.digitize(positions, bins=pos_bins) - 1
    masks_e = mapping_e[:,None] == np.arange(0,nEm1)[None,:]
    masks_p = mapping_p[:,None] == np.arange(0,nPm1)[None,:]
    return np.count_nonzero(np.logical_and(masks_e[:,:,None], masks_p[:,None,:]), axis=0)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_progress(max_e_i, max_mu_i, fname="progress.json", mode="w"):
    d_dict = dict()
    for c_i,energy in enumerate(energies):
        if energy not in loss_hists_by_primary:
            continue
        d_dict[energy] = dict()
        for key in particle_types.keys():
            if key not in loss_hists_by_primary[energy]:
                continue
            #d_dict[energy][key] = loss_hists_by_primary[energy][key].tolist()
            d_dict[energy][key] = loss_hists_by_primary[energy][key]

    j_dict = dict()
    j_dict["hists"] = d_dict
    #j_dict["muon energies"] = energies.tolist()
    j_dict["muon energies"] = energies
    #j_dict["energy loss bins"] = loss_energy_bins.tolist()
    j_dict["energy loss bins"] = loss_energy_bins
    #j_dict["length bins"] = pos_bins.tolist()
    j_dict["length bins"] = pos_bins
    j_dict["n muons"] = n_muons
    j_dict["max energy index"] = max_e_i
    j_dict["max muon index"] = max_mu_i

    f = open(fname, mode)
    #f.write(json.dumps(j_dict))
    json.dump(j_dict, f, cls=NumpyEncoder)

def load_progress(fname="progress.json"):
    if not os.path.exists("progress.json"):
        return
    f = open(fname, "r")
    j_dict = json.load(f)
    d_dict = j_dict["hists"]
    global max_e_i
    global max_mu_i
    l_energies = j_dict["muon energies"]
    l_loss_energy_bins = j_dict["energy loss bins"]
    l_pos_bins = j_dict["length bins"]
    l_n_muons = j_dict["n muons"]
    max_e_i = j_dict["max energy index"]
    max_mu_i = j_dict["max muon index"]

    e_string_list = list(d_dict.keys())
    e_data_list = [(float(s), s) for s in e_string_list]
    e_data_list = sorted(e_data_list, key=lambda x: x[0])
    e_string_list = [s for e,s in e_data_list]

    for c_i,(energy,l_energy) in enumerate(zip(energies, e_string_list)):
        loss_hists_by_primary[energy] = dict()
        if l_energy not in d_dict:
            continue
        for key in particle_types.keys():
            if key not in d_dict[l_energy]:
                continue
            loss_hists_by_primary[energy][key] = np.array(
                    d_dict[l_energy][key])

max_e_i = 0
max_mu_i = 0
load_progress()
print(max_e_i, max_mu_i)
for c_i,energy in enumerate(energies):
    if c_i < max_e_i:
        continue
    loss_hists = dict()
    temp_loss_data = dict()
    n_temp = 0
    for key in particle_types.keys():
        loss_hists[key] = 0.0
        temp_loss_data[key] = []

    for i in tqdm(range(n_muons)):
        if i < max_mu_i:
            continue
        while True:
            try:
                pp_part = make_p(energy)
                if mu_part["particle"] == 13:
                    secondaries = prop_mu_minus.propagate(pp_part, max_distance_cm=max_distance, minimal_energy=100)
                elif mu_part["particle"] == -13:
                    secondaries = prop_mu_plus.propagate(pp_part, max_distance_cm=max_distance, minimal_energy=100)
                particles = secondaries.particles
                decay_products = [p for p in particles[max(len(particles)-4,0):] if int(p.type) <= 1000000001 or int(p.type == 1000000011)]
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
                loss_data = [[float(e), float(particle.position.x)/1e2] for e, particle in zip(particle_energies,particles) if is_energy_loss(particle)]
                if len(loss_data) == 0:
                    loss_data = np.zeros(shape=(0,2))
                temp_loss_data[key].extend(loss_data)
                if n_temp >= max_data_len or i == n_muons-1:
                    loss_data = temp_loss_data[key]
                    loss_data = np.atleast_2d(np.array(loss_data))
                    hist = make_loss_hist(loss_data[:,0], loss_data[:,1])
                    loss_hists[key] += hist
                    temp_loss_data[key] = []

                for key in particle_types.keys():
                    loss_data = [[float(e), float(particle.position.x)/1e2] for e, particle in zip(particle_energies,particles) if particle.type == particle_types[key] and is_energy_loss(particle)]
                    temp_loss_data[key].extend(loss_data)
                    del loss_data
                    if n_temp >= max_data_len or i == n_muons-1:
                        loss_data = temp_loss_data[key]
                        if len(loss_data) == 0:
                            loss_data = np.zeros(shape=(0,2))
                        loss_data = np.atleast_2d(np.array(loss_data))
                        hist = make_loss_hist(loss_data[:,0], loss_data[:,1])
                        loss_hists[key] += hist
                        temp_loss_data[key] = []
                del particles
                if n_temp >= max_data_len or i == n_muons-1:
                    loss_hists_by_primary[energy] = loss_hists
                    save_progress(max_e_i, max_mu_i)
                    n_temp = 0
                else:
                    n_temp += 1
                max_mu_i += 1
                break
            except:
                pass
    if c_i < len(energies)-1:
        max_mu_i = 0
    max_e_i += 1

    loss_hists_by_primary[energy] = loss_hists

save_progress(len(energies)-1, n_muons-1, fname="hists.json", mode="w")

for key in particle_types.keys():
    fig, ax = plt.subplots(figsize=(7,5))
    print(key)
    for c_i,energy in enumerate(energies):
        color = cm(c_i/(max(len(energies)-1, 1)))
        loss_hist = loss_hists_by_primary[energy][key] / n_muons
        energy_hist = np.sum(loss_hist, axis=1)

        center = lambda x: (x[:-1] + x[1:]) / 2.0
        energy_centers = center(loss_energy_bins)

        ax.hist(energy_centers, weights=energy_hist, bins=loss_energy_bins, histtype='step', label=str(energy), color=color)
        ax.set_xlabel(r"Stochastic loss energy [GeV]")
        ax.set_ylabel(r"Average losses per muon in 15 m")
        ax.set_title(key)
        ax.set_xscale("log")
        ax.legend()
        fig.savefig(args.output + "_" + key + ".png", dpi=200)
