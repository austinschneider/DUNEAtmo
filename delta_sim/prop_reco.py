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

import mu_expectation

import scipy
import scipy.stats
import scipy.optimize

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
args = parser.parse_args()

max_entry = 0.0
n_muons = None
loss_hists_by_primary = dict()
energies = np.logspace(2, 3, 10+1)
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
min_loss_energy = 50 / 1e3
max_loss_energy = max(energies)
loss_energy_bins = np.logspace(np.log10(min_loss_energy), np.log10(max_loss_energy), int((np.log10(max_loss_energy) - np.log10(min_loss_energy)) * 10)+1)
pos_bins = np.linspace(0,62,62+1)
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
    global n_muons
    n_muons = l_n_muons

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

interpolation_def = pp.InterpolationDef()
interpolation_def.path_to_tables = table_path
interpolation_def.path_to_tables_readonly = table_path

mu_minus_def = pp.particle.MuMinusDef()
prop_mu_minus = pp.Propagator(particle_def=mu_minus_def, config_file=args.config)

mu_plus_def = pp.particle.MuPlusDef()
prop_mu_plus = pp.Propagator(particle_def=mu_plus_def, config_file=args.config)

def make_p(e):
    type = 13
    r = 100.*1e2
    zenith = np.arccos(np.random.random()*2-1)
    azimuth = np.random.random()*2*np.pi
    nx = np.sin(zenith)*np.cos(azimuth)
    ny = np.sin(zenith)*np.sin(azimuth)
    nz = np.cos(zenith)
    position = pp.Vector3D(nx*r,ny*r,nz*r)
    direction = pp.Vector3D(-nx, -ny, -nz)

    p = pp.particle.DynamicData(type)
    p.position = position
    p.direction = direction
    p.energy = e * 1e3
    p.time = 0
    p.propagated_distance = 0
    return p

nE = len(loss_energy_bins)
nP = len(pos_bins)
nEm1 = nE-1
nPm1 = nP-1
def make_loss_hist(energies, positions):
    #print("make_loss_hist")
    le = len(energies)
    lp = len(positions)
    #print("N Energies:", le)
    #print("N Positions:", lp)
    #print("Energy", np.amin(energies) if le > 0 else None, np.mean(energies), np.amax(energies) if le > 0 else None)
    #print("Positions", np.amin(positions) if lp > 0 else None, np.mean(positions), np.amax(positions) if lp > 0 else None)
    mapping_e = np.digitize(energies, bins=loss_energy_bins) - 1
    mapping_p = np.digitize(positions, bins=pos_bins) - 1
    masks_e = mapping_e[:,None] == np.arange(0,nEm1)[None,:]
    masks_p = mapping_p[:,None] == np.arange(0,nPm1)[None,:]
    res = np.count_nonzero(np.logical_and(masks_e[:,:,None], masks_p[:,None,:]), axis=0)
    #print("N in hist:", np.sum(res))
    return res

def get_likelihood(energy, data_hist, distance):
    mc_hist = all_interp.simple_expect(energy, distance)
    k = data_hist
    wsx = mc_hist

    res = np.zeros(np.shape(wsx))

    bad_mask = np.logical_and(wsx <= 0, k != 0)
    res[bad_mask] = -np.inf

    poisson_mask = wsx > 0
    k = k[poisson_mask]
    wsx = wsx[poisson_mask]

    logw = np.log(wsx)

    klogw = k*logw

    klogw_minus_w = klogw - wsx

    res[poisson_mask] = klogw_minus_w - scipy.special.loggamma(k + 1)

    res = -np.sum(res)

    return res

def get_2d_likelihood(energy, delta_data_hist, epair_data_hist, distance):
    delta_mc_hist = delta_interp.simple_expect(energy, distance)
    epair_mc_hist = epair_interp.simple_expect(energy, distance)
    mc_hist = np.concatenate([delta_mc_hist, epair_mc_hist])
    data_hist = np.concatenate([delta_data_hist, epair_data_hist])
    k = data_hist
    wsx = mc_hist

    res = np.zeros(np.shape(wsx))

    bad_mask = np.logical_and(wsx <= 0, k != 0)
    #res[bad_mask] = -np.inf
    res[bad_mask] = -np.finfo(float).max

    poisson_mask = wsx > 0
    k = k[poisson_mask]
    wsx = wsx[poisson_mask]

    logw = np.log(wsx)

    klogw = k*logw

    klogw_minus_w = klogw - wsx

    res[poisson_mask] = klogw_minus_w - scipy.special.loggamma(k + 1)

    res = -np.sum(res)

    return res

def get_3d_likelihood(energy, delta_data_hist, epair_data_hist, distance):
    delta_mc_hist = delta_interp.simple_2d_expect(energy, distance)
    epair_mc_hist = epair_interp.simple_2d_expect(energy, distance)
    mc_hist = np.concatenate([delta_mc_hist.flat, epair_mc_hist.flat])
    data_hist = np.concatenate([delta_data_hist.flat, epair_data_hist.flat])
    k = data_hist
    wsx = mc_hist

    res = np.zeros(np.shape(wsx))

    bad_mask = np.logical_and(wsx <= 0, k != 0)
    #res[bad_mask] = -np.inf
    res[bad_mask] = -np.finfo(float).max

    poisson_mask = wsx > 0
    k = k[poisson_mask]
    wsx = wsx[poisson_mask]

    logw = np.log(wsx)

    klogw = k*logw

    klogw_minus_w = klogw - wsx

    res[poisson_mask] = klogw_minus_w - scipy.special.loggamma(k + 1)

    res = -np.sum(res)

    return res

def do_reco(loss_hist, distance):
    energy_hist = np.sum(loss_hist, axis=1)
    best_res = None
    lenergies = np.log10(energies)
    print("losses:", np.sum(loss_hist))
    llhs = []
    for i in range(len(energies)-1):
        low = energies[i]
        high = energies[i+1]
        mid = 10**((lenergies[i]+lenergies[i+1])/2.0)
        res = scipy.optimize.minimize(lambda x: get_likelihood(x[0], energy_hist, distance), [mid], bounds=[[low, high]], method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-8, 'maxls': 20})
        llhs.append(res.fun)
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    print(llhs)
    return best_res.x

def do_2d_reco(delta_loss_hist, epair_loss_hist, distance):
    delta_energy_hist = np.sum(delta_loss_hist, axis=1)
    epair_energy_hist = np.sum(epair_loss_hist, axis=1)
    best_res = None
    lenergies = np.log10(energies)
    print("losses:", np.sum(delta_loss_hist), np.sum(epair_loss_hist))
    llhs = []
    best_res = scipy.optimize.minimize(lambda x: get_2d_likelihood(x[0], delta_energy_hist, epair_energy_hist, distance), [1e2], bounds=[[1e2, 1e3]], method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-8, 'maxls': 20})
    res = scipy.optimize.minimize(lambda x: get_2d_likelihood(x[0], delta_energy_hist, epair_energy_hist, distance), [1e3], bounds=[[1e2, 1e3]], method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-8, 'maxls': 20})
    if best_res is None or res.fun < best_res.fun:
        best_res = res
    for i in range(len(energies)-1):
        low = energies[i]
        high = energies[i+1]
        mid = 10**((lenergies[i]+lenergies[i+1])/2.0)
        res = scipy.optimize.minimize(lambda x: get_2d_likelihood(x[0], delta_energy_hist, epair_energy_hist, distance), [low], bounds=[[low, mid]], method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-8, 'maxls': 20})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
        res = scipy.optimize.minimize(lambda x: get_2d_likelihood(x[0], delta_energy_hist, epair_energy_hist, distance), [mid], bounds=[[low, high]], method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-8, 'maxls': 20})
        llhs.append(res.fun)
        if best_res is None or res.fun < best_res.fun:
            best_res = res
        res = scipy.optimize.minimize(lambda x: get_2d_likelihood(x[0], delta_energy_hist, epair_energy_hist, distance), [high], bounds=[[mid, high]], method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-8, 'maxls': 20})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    print(llhs)
    return best_res.x[0]

def do_2d_reco(delta_loss_hist, epair_loss_hist, distance):
    delta_energy_hist = np.sum(delta_loss_hist, axis=1)
    epair_energy_hist = np.sum(epair_loss_hist, axis=1)
    best_res = None
    loge = np.log10(energies)
    mine = np.amin(loge)
    maxe = np.amax(loge)
    these_energies = np.logspace(mine,maxe,200+1)
    best_e = None
    best_llh = None
    for energy in these_energies:
        llh = get_2d_likelihood(energy, delta_energy_hist, epair_energy_hist, distance)
        if best_llh is None or llh < best_llh:
            best_e = energy
            best_llh = llh
    return best_e

def do_3d_reco(delta_loss_hist, epair_loss_hist, distance):
    best_res = None
    loge = np.log10(energies)
    mine = np.amin(loge)
    maxe = np.amax(loge)
    these_energies = np.logspace(mine,maxe,200+1)
    best_e = None
    best_llh = None
    for energy in these_energies:
        llh = get_3d_likelihood(energy, delta_energy_hist, epair_energy_hist, distance)
        if best_llh is None or llh < best_llh:
            best_e = energy
            best_llh = llh
    return best_e

max_distance = 200 * 1e2

n_muons = int(1e3)

is_energy_loss = lambda p: int(p.type) > 1000000000 and int(p.type) < 1000000012
is_good_loss = lambda p: int(p.type) >= 1000000002 and int(p.type) <= 1000000004

mu_part = {"particle": 13}

my_energies = np.logspace(2,3,4)[1:]
my_energies = [1e3]
per_mu_slices_by_primary = dict()
all_energies_by_primary = dict()

cm = plt.get_cmap("plasma")

geo_det_single = []
geo_det_single.append(pp.geometry.Box(pp.Vector3D(), 13.9, 58.1, 11.9))

true_energies = []
reco_energies = []
true_lengths = []

for c_i,energy in enumerate(my_energies):
    print()
    print("#############################")
    print("#############################")
    print("#############################")
    print(energy)
    print("#############################")
    for i in tqdm(range(n_muons)):
        pp_part = make_p(energy)
        if mu_part["particle"] == 13:
            secondaries = prop_mu_minus.propagate(pp_part, max_distance_cm=max_distance, minimal_energy=100)
        elif mu_part["particle"] == -13:
            secondaries = prop_mu_plus.propagate(pp_part, max_distance_cm=max_distance, minimal_energy=100)
        particles = secondaries.particles
        entry = sim_tools.compute_sim_info(geo_det_single, pp_part, particles)
        morphology, deposited_energy, detector_track_length, start_info, end_info, path_pairs, secondaries_inside = entry
        if morphology == 2 or morphology == 3:
            pass
        else:
            continue
        start_energy = start_info[0]/1e3 if start_info is not None else np.nan
        start_pos = start_info[1] if start_info is not None else np.nan
        particles = secondaries_inside
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

        """
        key = ""
        loss_data = [[float(e), float(particle.position.x)/1e2] for e, particle in zip(particle_energies,particles) if is_good_loss(particle)]
        if len(loss_data) == 0:
            loss_data = np.zeros(shape=(0,2))
        loss_data = np.atleast_2d(np.array(loss_data))
        hist = make_loss_hist(loss_data[:,0], loss_data[:,1])
        print("# start", start_energy)
        reco_energy = do_reco(hist, detector_track_length/100.)
        print("# reco", reco_energy)
        """

        loss_data = [[float(e), float((particle.position - start_pos).magnitude())/1e2] for e, particle in zip(particle_energies,particles) if particle.type == particle_types["DeltaE"]]
        if len(loss_data) == 0:
            loss_data = np.zeros(shape=(0,2))
        loss_data = np.atleast_2d(np.array(loss_data))
        delta_hist = make_loss_hist(loss_data[:,0], loss_data[:,1])
        loss_data = [[float(e), float((particle.position - start_pos).magnitude())/1e2] for e, particle in zip(particle_energies,particles) if particle.type == particle_types["Epair"]]
        if len(loss_data) == 0:
            loss_data = np.zeros(shape=(0,2))
        loss_data = np.atleast_2d(np.array(loss_data))
        epair_hist = make_loss_hist(loss_data[:,0], loss_data[:,1])
        reco_energy = do_2d_reco(delta_hist, epair_hist, detector_track_length/100.)
        print("# %.02f, %.02f" %( start_energy, reco_energy))

        """
        for key in particle_types.keys():
            loss_data = [[float(e), float(particle.position.x)/1e2] for e, particle in zip(particle_energies,particles) if particle.type == particle_types[key] and is_energy_loss(particle)]
            if len(loss_data) == 0:
                loss_data = np.zeros(shape=(0,2))
            loss_data = np.atleast_2d(np.array(loss_data))
            hist = make_loss_hist(loss_data[:,0], loss_data[:,1])
        """
        true_energies.append(start_energy)
        reco_energies.append(reco_energy)
        true_lengths.append(detector_track_length/100.)

d = {
    "true_energy": true_energies,
    "reco_energy": reco_energies,
    "true_length": true_lengths,
    }

f = open("mu_reco.json", "w")
f.write(json.dumps(d))
f.close()



