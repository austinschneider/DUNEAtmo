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
parser.add_argument('--h5',
        type=str,
        dest='h5',
        required=True
        )
parser.add_argument('--output',
        type=str,
        dest='output',
        required=True
        )
args = parser.parse_args()

interpolation_def = pp.InterpolationDef()
print(table_path)
interpolation_def.path_to_tables = table_path
interpolation_def.path_to_tables_readonly = table_path

mu_minus_def = pp.particle.MuMinusDef()
prop_mu_minus = pp.Propagator(particle_def=mu_minus_def, config_file=args.config)

mu_plus_def = pp.particle.MuPlusDef()
prop_mu_plus = pp.Propagator(particle_def=mu_plus_def, config_file=args.config)

data_file = h5.File(args.h5, 'r')
injector_list = [i for i in data_file.keys()]
props = None
mu_props = None
injector_n = []
for i in injector_list:
    p = data_file[i]["properties"][:]
    p.dtype.names = (
            'energy',
            'zenith',
            'azimuth',
            'bjorken_x',
            'bjorken_y',
            'final_type_0',
            'final_type_1',
            'particle',
            'x',
            'y',
            'z',
            'total_column_depth'
            )
    mp = data_file[i]['final_1'][:]
    mp.dtype.names = ('_', 'particle', 'position', 'direction', 'energy')
    injector_n.append(len(p))
    if props is None:
        props = p
    else:
        props = np.concatenate([props, p])
    if mu_props is None:
        mu_props = mp
    else:
        mu_props = np.concatenate([mu_props, mp])

def make_p(p, mu_p):
    type = mu_p["particle"]
    pos = [pos_x*100. for pos_x in mu_p["position"]]
    position = pp.Vector3D(*pos)
    zenith, azimuth = mu_p["direction"]
    nx = np.sin(zenith)*np.cos(azimuth)
    ny = np.sin(zenith)*np.sin(azimuth)
    nz = np.cos(zenith)
    direction = pp.Vector3D(nx, ny, nz)
    energy = mu_p["energy"]*1e3
    parent_energy = p["energy"]*1e3
    time = 0
    propagated_distance = 0

    p = pp.particle.DynamicData(type)
    p.position = position
    p.direction = direction
    p.energy = energy
    p.parent_particle_energy = parent_energy
    p.time = time
    p.propagated_distance = propagated_distance
    return p

geo_det_single = []
geo_det_single.append(pp.geometry.Box(pp.Vector3D(), 13.9, 58.1, 11.9))

geo_det_double = []
geo_det_double.append(pp.geometry.Box(pp.Vector3D(12, 0, 0), 12, 58, 12))
geo_det_double.append(pp.geometry.Box(pp.Vector3D(-12, 0, 0), 12, 58, 12))

geo_det_triple = []
geo_det_triple.append(pp.geometry.Box(pp.Vector3D(), 12, 58, 12))
geo_det_triple.append(pp.geometry.Box(pp.Vector3D(24, 0, 0), 12, 58, 12))
geo_det_triple.append(pp.geometry.Box(pp.Vector3D(-24, 0, 0), 12, 58, 12))

geo_det_quad = []
geo_det_quad.append(pp.geometry.Box(pp.Vector3D(12, 0, 0), 12, 58, 12))
geo_det_quad.append(pp.geometry.Box(pp.Vector3D(-12, 0, 0), 12, 58, 12))
geo_det_quad.append(pp.geometry.Box(pp.Vector3D(36, 0, 0), 12, 58, 12))
geo_det_quad.append(pp.geometry.Box(pp.Vector3D(-36, 0, 0), 12, 58, 12))


det_geos = [geo_det_single, geo_det_double, geo_det_triple, geo_det_quad]
geo = geo_det_single[0]
geo_epsilon = pp.geometry.Box(pp.Vector3D(), 14, 58.2, 12)

min_energy = 0.2e3

order = np.arange(len(props))
np.random.shuffle(order)
injectors = np.zeros(len(props)).astype(int)
n_tot = 0
for i,n in enumerate(injector_n):
    injectors[n_tot:n_tot+n] = i
    n_tot += n

injector_count = np.zeros(len(injector_list))


mode = 'w'
batch_size = 0 
save_size = 5000
n_saved = 0
entries = []
entries_mask = []
particle_counts = dict()

for i in tqdm(range(len(props)), total=len(props)):
    injector_index = injectors[order[i]]
    injector_count[injector_index] += 1
    part, mu_part = props[order[i]], mu_props[order[i]]
    pp_part = make_p(part, mu_part)

    if pp_part.type not in particle_counts:
        particle_counts[pp_part.type] = 0
    particle_counts[pp_part.type] += 1

    d0, d1 = geo_epsilon.distance_to_border(pp_part.position, pp_part.direction)
    infront = d0 > 0 and d1 > 0
    inside = d0 > 0 and d1 < 0
    outside = d0 < 0 and d1 < 0
    if outside:
        entries_mask.append(False)
    else:
        if mu_part["particle"] == 13:
            secondaries = prop_mu_minus.propagate(pp_part, max_distance_cm=1e20, minimal_energy=100)
        elif mu_part["particle"] == -13:
            secondaries = prop_mu_plus.propagate(pp_part, max_distance_cm=1e20, minimal_energy=100)
        particles = secondaries.particles

        entry = sim_tools.compute_sim_info(geo_det_single, pp_part, particles)
        morphology, deposited_energy, detector_track_length, start_info, end_info, path_pairs = entry
        if morphology == sim_tools.EventMorphology.missing:
            entries_mask.append(False)
        else:
            entries.append(entry)
            entries_mask.append(True)
        del particles
        del secondaries
    batch_size += 1

    if i < len(props)-1 and batch_size < save_size:
        continue

    mask = order[n_saved:n_saved+len(entries_mask)][entries_mask]
    these_props = props[mask]
    mu_these_props = mu_props[mask]

    energy = these_props["energy"]
    zenith = these_props["zenith"]
    azimuth = these_props["azimuth"]
    bjorken_x = these_props["bjorken_x"]
    bjorken_y = these_props["bjorken_y"]
    final_type_0 = these_props["final_type_0"]
    final_type_1 = these_props["final_type_1"]
    particle = these_props["particle"]
    x = these_props["x"]
    y = these_props["y"]
    z = these_props["z"]
    total_column_depth = these_props["total_column_depth"]

    mu_energy = mu_these_props["energy"]
    mu_pos = mu_these_props["position"]
    mu_dir = mu_these_props["direction"]
    mu_x = np.array([pos[0] for pos in mu_pos])
    mu_y = np.array([pos[1] for pos in mu_pos])
    mu_z = np.array([pos[2] for pos in mu_pos])
    mu_zenith = np.array([d[0] for d in mu_dir])
    mu_azimuth = np.array([d[1] for d in mu_dir])

    morphology = np.array([int(entry[0]) for entry in entries])
    deposited_energy = np.array([entry[1] for entry in entries]) / 1e3
    track_length = np.array([entry[2] for entry in entries]) / 100.

    get = lambda x,i,j: x[i][j] if x[i] is not None else np.nan
    getx = lambda x,i,j: x[i][j].x if x[i] is not None else np.nan
    gety = lambda x,i,j: x[i][j].y if x[i] is not None else np.nan
    getz = lambda x,i,j: x[i][j].z if x[i] is not None else np.nan

    entry_energy = np.array([get(entry,3,0) for entry in entries]) / 1e3
    entry_x = np.array([getx(entry,3,1) for entry in entries]) / 100.
    entry_y = np.array([gety(entry,3,1) for entry in entries]) / 100.
    entry_z = np.array([getz(entry,3,1) for entry in entries]) / 100.
    entry_nx = np.array([getx(entry,3,2) for entry in entries])
    entry_ny = np.array([gety(entry,3,2) for entry in entries])
    entry_nz = np.array([getz(entry,3,2) for entry in entries])
    entry_d = np.sqrt(entry_nx**2 + entry_ny**2 + entry_nz**2)
    entry_nx /= entry_d
    entry_ny /= entry_d
    entry_nz /= entry_d
    entry_zenith = np.arccos(entry_nz)
    entry_azimuth = np.arctan2(entry_ny, entry_nx)

    exit_energy = np.array([get(entry,4,0) for entry in entries]) / 1e3
    exit_x = np.array([getx(entry,4,1) for entry in entries]) / 100.
    exit_y = np.array([gety(entry,4,1) for entry in entries]) / 100.
    exit_z = np.array([getz(entry,4,1) for entry in entries]) / 100.
    exit_nx = np.array([getx(entry,4,2) for entry in entries])
    exit_ny = np.array([gety(entry,4,2) for entry in entries])
    exit_nz = np.array([getz(entry,4,2) for entry in entries])
    exit_d = np.sqrt(exit_nx**2 + exit_ny**2 + exit_nz**2)
    exit_nx /= exit_d
    exit_ny /= exit_d
    exit_nz /= exit_d
    exit_zenith = np.arccos(exit_nz)
    exit_azimuth = np.arctan2(exit_ny, exit_nx)

    this_check = final_type_0.tolist()
    data = {
        'energy': energy.tolist(),
        'zenith': zenith.tolist(),
        'azimuth': azimuth.tolist(),
        'bjorken_x': bjorken_x.tolist(),
        'bjorken_y': bjorken_y.tolist(),
        'final_type_0': final_type_0.tolist(),
        'final_type_1': final_type_1.tolist(),
        'particle': particle.tolist(),
        'x': x.tolist(),
        'y': y.tolist(),
        'z': z.tolist(),
        'total_column_depth': total_column_depth.tolist(),
        'mu_energy': mu_energy.tolist(),
        'mu_x': mu_x.tolist(),
        'mu_y': mu_y.tolist(),
        'mu_z': mu_z.tolist(),
        'mu_zenith': mu_zenith.tolist(),
        'mu_azimuth': mu_azimuth.tolist(),
        'morphology': morphology.tolist(),
        'deposited_energy': deposited_energy.tolist(),
        'track_length': track_length.tolist(),
        'entry_energy': entry_energy.tolist(),
        'entry_x': entry_x.tolist(),
        'entry_y': entry_y.tolist(),
        'entry_z': entry_z.tolist(),
        'entry_zenith': entry_zenith.tolist(),
        'entry_azimuth': entry_azimuth.tolist(),
        'exit_energy': exit_energy.tolist(),
        'exit_x': exit_x.tolist(),
        'exit_y': exit_y.tolist(),
        'exit_z': exit_z.tolist(),
        'exit_zenith': exit_zenith.tolist(),
        'exit_azimuth': exit_azimuth.tolist(),
        'injector_count': injector_count.tolist(),
        }

    f = open(args.output, mode)
    json.dump(data, f)
    f.close()

    mode = 'a'
    n_saved += len(entries_mask)
    entries = []
    entries_mask = []
    particle_counts = dict()
    injector_count = np.zeros(len(injector_list))

