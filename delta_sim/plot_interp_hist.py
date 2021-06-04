import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.style

matplotlib.style.use("./paper.mplstyle")

import sys
import os
import os.path
base_path = os.environ['GOLEMSPACE']
table_path = base_path + '/local/share/PROPOSAL/tables'
config_path = base_path + '/sources/DUNEAtmo/proposal_config/'
import proposal as pp
import numpy as np
import h5py as h5
import json
import sim_tools
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Propagate muons")
parser.add_argument('--output',
        type=str,
        dest='output',
        required=True
        )
args = parser.parse_args()


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


energies = np.logspace(2, 3, 10+1)
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

max_data_len = 20000

def make_loss_hist(energies, positions):
    mapping_e = np.digitize(energies, bins=loss_energy_bins) - 1
    mapping_p = np.digitize(positions, bins=pos_bins) - 1
    masks_e = mapping_e[:,None] == np.arange(0,nEm1)[None,:]
    masks_p = mapping_p[:,None] == np.arange(0,nPm1)[None,:]
    return np.count_nonzero(np.logical_and(masks_e[:,:,None], masks_p[:,None,:]), axis=0)


max_entry = 0.0

def load_progress(fname="hists.json"):
    if not os.path.exists(fname):
        return
    f = open(fname, "r")
    j_dict = json.load(f)
    d_dict = j_dict["hists"]
    l_energies = j_dict["muon energies"]
    l_loss_energy_bins = j_dict["energy loss bins"]
    l_pos_bins = j_dict["length bins"]
    l_n_muons = j_dict["n muons"]

    global max_entry

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
            loss_hists_by_primary[energy][key] = np.array(d_dict[l_energy][key])
            max_entry = max(max_entry, np.amax(np.sum(loss_hists_by_primary[energy][key], axis=-1)))

load_progress()

max_entry = float(max_entry) / float(n_muons)

keys_to_plot = [["DeltaE"], ["Epair"], ["Brems"], ["DeltaE", "Epair", "Brems"]]
labels = ["DeltaE", "Epair", "Brems", "All"]

import mu_expectation

delta_data = np.empty((len(energies), len(loss_energy_bins)-1, len(pos_bins)-1))
epair_data = np.empty((len(energies), len(loss_energy_bins)-1, len(pos_bins)-1))
brems_data = np.empty((len(energies), len(loss_energy_bins)-1, len(pos_bins)-1))
for c_i,energy in enumerate(energies):
    delta_data[c_i] = loss_hists_by_primary[energy]["DeltaE"] / n_muons
    epair_data[c_i] = loss_hists_by_primary[energy]["Epair"] / n_muons
    brems_data[c_i] = loss_hists_by_primary[energy]["Brems"] / n_muons
all_data = delta_data + epair_data + brems_data

delta_interp = mu_expectation.MuExpect(delta_data, energies, loss_energy_bins, pos_bins)
epair_interp = mu_expectation.MuExpect(epair_data, energies, loss_energy_bins, pos_bins)
brems_interp = mu_expectation.MuExpect(brems_data, energies, loss_energy_bins, pos_bins)
all_interp = mu_expectation.MuExpect(all_data, energies, loss_energy_bins, pos_bins)

interps_by_key = {
        "DeltaE": delta_interp,
        "Epair": epair_interp,
        "Brems": brems_interp,
        "All": all_interp,
        }

pos_bins = np.linspace(14,16,5+1)
energies = np.logspace(2,3,20+1)

fig, ax = plt.subplots(figsize=(7,5))
for i, pos in enumerate(pos_bins):
    for keys,key_label in zip(keys_to_plot,labels):
        interp = interps_by_key[key_label]
        for c_i,energy in enumerate(energies):
            color = cm(c_i/(max(len(energies)-1, 1)))
            energy_hist = interp.expect(energy, pos)

            center = lambda x: (x[:-1] + x[1:]) / 2.0
            energy_centers = center(loss_energy_bins)

            label = ("%d"%energy) + " [GeV] : " + ("%.02f losses"%np.sum(energy_hist))
            ax.hist(energy_centers, weights=energy_hist, bins=loss_energy_bins, histtype='step', label=label, color=color)
            ax.set_xlabel(r"Stochastic loss energy [GeV]")
            ax.set_ylabel(r"Average losses per muon in %d m" % pos)
            ax.set_title(key_label)
            ax.set_xscale("log")
            ax.set_ylim((0,max_entry))
            ax.legend()
            fig.savefig(args.output + "_" + key_label + ("_%.02fm"%pos) + ".png", dpi=200)
        ax.cla()
