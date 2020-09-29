import proposal as pp
import numpy as np
import h5py as h5
import LWpy
import LeptonInjector
import json
import sim_tools
from tqdm import tqdm

medium = pp.medium.StandardRock()

mu_def = pp.particle.MuMinusDef()

interpolation_def = pp.InterpolationDef()
interpolation_def.path_to_tables = "/home/austin/.local/share/PROPOSAL/tables"
interpolation_def.path_to_tables_readonly = "/home/austin/.local/share/PROPOSAL/tables"

mu_def = pp.particle.MuMinusDef()
#prop = pp.Propagator(particle_def=mu_def, sector_defs=[sector], detector=geo_detector, interpolation_def=interpolation_def)
prop = pp.Propagator(particle_def=mu_def, config_file="./config.json")

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

data_file = h5.File("data_output_DUNE_2.h5")
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
    type = mu_def.particle_type
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
geo_det_single.append(pp.geometry.Box(pp.Vector3D(), 12, 58, 12))

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
geo_epsilon = pp.geometry.Box(pp.Vector3D(), 12+1, 58+1, 12+1)

min_energy = 0.2e3

order = np.arange(len(props))
np.random.shuffle(order)
injectors = np.zeros(len(props)).astype(int)
n_tot = 0
for i,n in enumerate(injector_n):
    injectors[n_tot:n_tot+n] = i
    n_tot += n

injector_count = np.zeros(len(injector_list))

entries = []
entries_mask = []

def save_entries():
    mask = np.zeros(len(props)).astype(bool)
    mask[:len(entries_mask)] = np.array(entries_mask)
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
    entry_zenith = np.arccos(entry_nz)
    entry_azimuth = np.arctan2(entry_ny, entry_nx)

    exit_energy = np.array([get(entry,3,0) for entry in entries]) / 1e3
    exit_x = np.array([getx(entry,3,1) for entry in entries]) / 100.
    exit_y = np.array([gety(entry,3,1) for entry in entries]) / 100.
    exit_z = np.array([getz(entry,3,1) for entry in entries]) / 100.
    exit_nx = np.array([getx(entry,3,2) for entry in entries])
    exit_ny = np.array([gety(entry,3,2) for entry in entries])
    exit_nz = np.array([getz(entry,3,2) for entry in entries])
    exit_zenith = np.arccos(exit_nz)
    exit_azimuth = np.arctan2(exit_ny, exit_nx)

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

    f = open('propagated.json', 'w')
    json.dump(data, f)
    f.close()

for i in tqdm(range(len(props)), total=len(props)):
    injector_index = injectors[order[i]]
    injector_count[injector_index] += 1
    part, mu_part = props[order[i]], mu_props[order[i]]
    pp_part = make_p(part, mu_part)
    d0, d1 = geo_epsilon.distance_to_border(pp_part.position, pp_part.direction)
    infront = d0 > 0 and d1 > 0
    inside = d0 > 0 and d1 < 0
    outside = d0 < 0 and d1 < 0
    if outside:
        entries_mask.append(False)
    else:
        secondaries = prop.propagate(pp_part, max_distance_cm=1e20, minimal_energy=100)
        particles = secondaries.particles

        entry = sim_tools.compute_sim_info(geo_det_single, pp_part, particles)
        morphology, deposited_energy, detector_track_length, start_info, end_info, path_pairs = entry
        if morphology == sim_tools.EventMorphology.missing:
            entries_mask.append(False)
        else:
            entries.append(entry)
            entries_mask.append(True)

    if i % 50000 == 0:
        save_entries()
save_entries()

