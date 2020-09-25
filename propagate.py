import proposal as pp
import numpy as np
import h5py as h5
import LWpy
import LeptonInjector
import json

medium = pp.medium.StandardRock()

mu_def = pp.particle.MuMinusDef()
geo_detector = pp.geometry.Box(pp.Vector3D(), 84, 58, 12)
det_sector = pp.SectorDefinition()
det_sector.medium = medium
det_sector.geometry = geo_detector
det_sector.particle_location = pp.ParticleLocation.inside_detector
det_sector.scattering_model = pp.scattering.ScatteringModel.Moliere
det_sector.cut_settings.ecut = 500
det_sector.cut_settings.vcut = 0.05

geo_outside = pp.geometry.Box(pp.Vector3D(0, 0, 0), 5000*100, 5000*100, 5000*100)
out_sector = pp.SectorDefinition()
out_sector.medium = medium
out_sector.geometry = geo_outside
out_sector.particle_location = pp.ParticleLocation.inside_detector
out_sector.scattering_model = pp.scattering.ScatteringModel.Moliere
out_sector.cut_settings.ecut = 500
out_sector.cut_settings.vcut = 0.05

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

data_file = h5.File("data_output_DUNE.h5")
injector_list = [i for i in data_file.keys()]
for i in injector_list:
    props = data_file[i]["properties"][:]
    props.dtype.names = (
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
    mu_props = data_file[i]['final_1'][:]
    mu_props.dtype.names = ('_', 'particle', 'position', 'direction', 'energy')

#    gen.prob_pos(props)
#    gen.prob_area(props)
#    gen.get_considered_range(props)
#    gen.prob_kinematics(props)

#nu_interactions_list = standard_interactions.get_standard_interactions()
#int_model = LWpy.interaction_model(nu_interactions_list, earth_model_params)
#int_model.prob_kinematics(props)

#first_pos, last_pos = gen.get_considered_range(props)
#phys_pos = int_model.prob_pos(props, first_pos, last_pos)
#gen_pos = gen.prob_pos(props)
#p_int = int_model.prob_interaction(props, first_pos, last_pos)

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

def get_particle_at_entry(geo, parent, particles):
    d0, d1 = geo.distance_to_border(parent.position, parent.direction)
    last_infront = d0 > 0 and d1 > 0
    last_inside = d0 > 0 and d1 < 0
    outside = d0 < 0 and d1 < 0

    was_infront = last_infront
    was_inside = last_inside

    last_infront_part = None
    last_infront_k = None
    if last_infront:
        last_infront_part = parent
        last_infront_k = -1

    entered = False
    index = None

    inside_record = [last_inside]
    outside_record = [outside]
    infront_record = [last_infront]
    index_record = [-1]

    for j, part in enumerate(particles):
        # Skip continuous losses since idk how their location is determined
        if int(part.type) == int(pp.particle.Interaction_Type.ContinuousEnergyLoss):
            continue
        d0, d1 = geo.distance_to_border(part.position, part.direction)
        if part.energy < min_energy:
            break
        outside = d0 < 0 and d1 < 0
        infront = d0 > 0 and d1 > 0
        inside = d0 > 0 and d1 < 0
        if last_infront and inside:
            entered = True
            index = j
            print("First")
            break
        last_inside = inside
        last_infront = infront
        if infront:
            last_infront_part = part
            last_infront_k = j
        inside_record.append(inside)
        outside_record.append(outside)
        infront_record.append(infront)
        index_record.append(j)

    if not entered:
        k = -1
        for j, infront in enumerate(infront_record):
            if infront and not np.any(infront_record[j:]) and j+1 < len(infront_record):
                last_infront_part = particles[index_record[j]]
                last_infront_k = index_record[j]
                k = j+1
                break
        if k > -1:
            index = index_record[k]
            entered = True

        if index is not None:
            assert(index > last_infront_k)

    if entered:
        j = index
        initial_energy = last_infront_part.energy
        initial_pos = last_infront_part.position

        part = particles[index]

        final_pos = part.position

        continuous_losses = particles[last_infront_k+1:j]
        if len(continuous_losses) > 0:
            final_energy = continuous_losses[-1].energy
        else:
            final_energy = part.energy


        final_initial_vector = final_pos - initial_pos
        total_distance = final_initial_vector.magnitude()
        final_initial_direction = final_pos - initial_pos
        final_initial_direction.normalize()
        assert(np.abs(final_initial_direction.magnitude() - 1.0) < 1e-8)
        assert(final_initial_direction*parent.direction > 0)
        #print(np.abs(final_initial_direction*parent.direction - 1.0))
        distance_to_entry = geo.distance_to_border(initial_pos, final_initial_direction)[0]

        assert(distance_to_entry > 0)
        #print(total_distance, distance_to_entry)
        d0, d1 = geo.distance_to_border(initial_pos, final_initial_direction)
        track_length = d1 - d0
        #print(d0,d1)
        #print(initial_pos, final_pos)
        assert(total_distance > distance_to_entry)

        delta_energy = final_energy - initial_energy

        entry_energy = delta_energy/total_distance*distance_to_entry + initial_energy
        entry_pos = initial_pos + distance_to_entry*final_initial_direction

        return entry_energy, entry_pos, final_initial_direction, track_length
    else:
        return None

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
geo_epsilon = pp.geometry.Box(pp.Vector3D(), 12+0.01, 58+0.01, 12+0.01)

min_energy = 0.2e3

entries = []
entries_mask = []
for i, (part, mu_part) in enumerate(zip(props, mu_props)):
    pp_part = make_p(part, mu_part)
    d0, d1 = geo_epsilon.distance_to_border(pp_part.position, pp_part.direction)
    infront = d0 > 0 and d1 > 0
    inside = d0 > 0 and d1 < 0
    outside = d0 < 0 and d1 < 0
    if outside:
        entries_mask.append(False)
        continue
    secondaries = prop.propagate(pp_part, max_distance_cm=1e20, minimal_energy=100)
    if secondaries.number_of_particles < 4:
        entries_mask.append(False)
        continue
    parent_energy = secondaries.parent_particle_energy[:-3]
    loss_energy = secondaries.energy[:-3]
    position = secondaries.position[:-3]
    direction = secondaries.direction[:-3]
    particles = secondaries.particles[:-3]

    entry = get_particle_at_entry(geo, pp_part, particles)
    if entry is not None:
        entry_energy, entry_position, entry_direction, track_length = entry
        entries.append((part, entry))
        print("Entered at", entry_energy/1e3, "GeV!")
        entries_mask.append(True)
    else:
        entries_mask.append(False)

entries_mask = np.array(entries_mask)
mask = np.zeros(len(props)).astype(bool)
mask[:len(entries_mask)] = entries_mask
props = props[mask]

energy = props["energy"]
zenith = props["zenith"]
azimuth = props["azimuth"]
bjorken_x = props["bjorken_x"]
bjorken_y = props["bjorken_y"]
final_type_0 = props["final_type_0"]
final_type_1 = props["final_type_1"]
particle = props["particle"]
x = props["x"]
y = props["y"]
z = props["z"]
total_column_depth = props["total_column_depth"]

entry_energy = np.array([entry[1][0] for entry in entries])
entry_x = np.array([entry[1][1].x for entry in entries]) / 100.
entry_y = np.array([entry[1][1].y for entry in entries]) / 100.
entry_z = np.array([entry[1][1].z for entry in entries]) / 100.
entry_nx = np.array([entry[1][2].x for entry in entries])
entry_ny = np.array([entry[1][2].y for entry in entries])
entry_nz = np.array([entry[1][2].z for entry in entries])
entry_zenith = np.arccos(entry_nz)
entry_azimuth = np.arctan2(entry_ny, entry_nx)
track_length = np.array([entry[1][3] for entry in entries]) / 100.

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
    'entry_energy': entry_energy.tolist(),
    'entry_x': entry_x.tolist(),
    'entry_y': entry_y.tolist(),
    'entry_z': entry_z.tolist(),
    'entry_zenith': entry_zenith.tolist(),
    'entry_azimuth': entry_azimuth.tolist(),
    'track_length': track_length.tolist(),
    }

json.dump(data, open('propagated.json', 'w'))
