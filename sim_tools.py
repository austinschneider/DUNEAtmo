import proposal as pp
import numpy as np
import h5py as h5
import LWpy
import LeptonInjector
import json
from enum import Enum

def get_particle_at_entry(geo, parent, particles):
    d0, d1 = geo.distance_to_border(parent.position, parent.direction)
    last_infront = d0 > 0 and d1 > 0
    last_inside = d0 > 0 and d1 < 0
    outside = d0 < 0 and d1 < 0

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
        assert(initial_energy > entry_energy)
        entry_pos = initial_pos + distance_to_entry*final_initial_direction

        return entry_energy, entry_pos, final_initial_direction, track_length
    else:
        return None

def get_particle_at_exit(geo, parent, particles):
    d0, d1 = geo.distance_to_border(parent.position, parent.direction)
    last_infront = d0 > 0 and d1 > 0
    last_inside = d0 > 0 and d1 < 0
    outside = d0 < 0 and d1 < 0

    last_infront_part = None
    last_infront_k = None
    if last_infront:
        last_infront_part = parent
        last_infront_k = -1

    exited = False
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
        outside = d0 < 0 and d1 < 0
        infront = d0 > 0 and d1 > 0
        inside = d0 > 0 and d1 < 0
        if (last_inside or last_infront) and outside:
            exited = True
            index = j
            print("First")
            break
        last_inside = inside
        last_infront = infront
        if infront or inside:
            last_infront_part = part
            last_infront_k = j
        inside_record.append(inside)
        outside_record.append(outside)
        infront_record.append(infront)
        index_record.append(j)

    if exited:
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
        distance_to_exit = geo.distance_to_border(initial_pos, final_initial_direction)[0]

        assert(distance_to_exit > 0)
        assert(total_distance > distance_to_exit)

        delta_energy = final_energy - initial_energy

        exit_energy = delta_energy/total_distance*distance_to_exit + initial_energy
        assert(initial_energy > exit_energy)
        exit_pos = initial_pos + distance_to_exit*final_initial_direction

        return exit_energy, exit_pos, final_initial_direction
    else:
        return None

def is_starting(geo, parent, particles):
    d0, d1 = geo.distance_to_border(parent.position, parent.direction)
    inside = d0 > 0 and d1 < 0
    return inside

class EventMorphology(Enum):
    missing = 0
    starting = 1
    stopping = 2
    through_going = 3
    contained = 4
    pseudo_contained = 5

def compute_through_going_event_props(parent, particles, entry_info, exit_info):
    entry_energy, entry_pos, _ = entry_info
    exit_energy, exit_pos, _ = exit_info
    energy_in_detector = entry_energy - exit_energy

    track_length = 0
    last_pos = entry_pos
    for i, p in enumerate(particles[:-3]):
        if p.energy < entry_energy and p.energy > exit_energy:
            track_length += (last_pos - p.position).magnitude()
            last_pos = p.position
        elif p.energy <= exit_energy:
            track_length += (last_pos - exit_pos).magnitude()
            break
    return energy_in_detector, track_length, entry_info, exit_info

def compute_starting_event_props(parent, particles, entry_info, exit_info):
    exit_energy, exit_pos, _ = exit_info
    energy_in_detector = parent.energy - exit_energy

    track_length = 0
    last_pos = parent.position
    for i, p in enumerate(particles[:-3]):
        if p.energy > exit_energy:
            track_length += (last_pos - p.position).magnitude()
            last_pos = p.position
        elif p.energy <= exit_energy:
            track_length += (last_pos - exit_pos).magnitude()
            break
    return energy_in_detector, track_length, entry_info, exit_info

def compute_contained_event_props(parent, particles, entry_info, exit_info):
    energy_in_detector = parent.energy - [p.energy for p in particles if abs(int(p.type)) in [12, 14, 16]]

    track_length = 0
    last_pos = parent.position
    for i, p in enumerate(particles[-2]):
        track_length += (last_pos - p.position).magnitude()
        last_pos = p.position
    return energy_in_detector, track_length, entry_info, exit_info

def compute_stopping_event_props(parent, particles, entry_info, exit_info):
    entry_energy, entry_pos, _ = entry_info
    energy_in_detector = entry_energy - [p.energy for p in particles if abs(int(p.type)) in [12, 14, 16]]

    track_length = 0
    last_pos = entry_pos
    for i, p in enumerate(particles[-2]):
        if p.energy < entry_energy:
            track_length += (last_pos - p.position).magnitude()
            last_pos = p.position
    return energy_in_detector, track_length, entry_info, exit_info

def compute_sim_info(geos, parent, particles):
    starts = False
    stops = False
    intersects = False
    contained = False
    deposited_energy = 0
    detector_track_length = 0
    info = []
    start_info = None
    end_info = None
    path_pairs = []
    for geo in geos:
        starting = is_starting(geo, parent, particles)
        entry_info = None
        if not starting:
            entry_info = get_particle_at_entry(geo, parent, particles)
        entering = entry_info is not None
        exit_info = None
        if entry_info is not None or starting:
            exit_info = get_particle_at_exit(geo, parent, particles)
        exiting = exit_info is not None

        through_going = entering and exiting
        stopping = (entering or starting) and (not exiting)
        missing = (not entering) and (not starting)

        starts |= starting
        stops |= starting
        intersects |= not missing
        contained |= starting and (not exiting)

        energy_in_detector = 0
        track_length = 0
        if through_going: # muon just passing through
            energy_in_detector, track_length, _, _ = compute_through_going_event_props(parent, particles, entry_info, exit_info)
        elif starting and exiting: # muon begins in the detector but doesn't stay
            energy_in_detector, track_length, _, _ = compute_starting_event_props(parent, particles, entry_info, exit_info)
        elif contained: # muon is only ever in the detector
            energy_in_detector, track_length, _, _ = compute_contained_event_props(parent, particles, entry_info, exit_info)
        elif entering and (not exiting): # muon comes from outside to stop in the detector
            energy_in_detector, track_length, _, _ = compute_starting_event_props(parent, particles, entry_info, exit_info)
        deposited_energy += energy_in_detector
        detector_track_length += track_length

        if entry_info is not None:
            if start_info is None or entry_info[0] > start_info[0]:
                start_info = entry_info

        if exit_info is not None:
            if end_info is None or exit_info[0] < end_info[0]:
                end_info = exit_info

        path_pairs.append((entry_info, exit_info))

    # contained = intersects and contained
    pseudo_contained = intersects and (not contained) and (starts and stops)
    through_going = intersects and (not stops) and (not starts)
    stopping = intersects and (stops) and (not starts)
    starting = intersects and (not stops) and starts
    missing = not intersects
    bin_arr = [missing, starting, stopping, through_going, contained, pseudo_contained]

    assert(np.sum(bin_arr) == 1)
    morphology = EventMorphology(bin_arr.index(1))

    return morphology, deposited_energy, detector_track_length, start_info, end_info, path_pairs

def intersects_geo(geo, particle):
    d0, d1 = geo_epsilon.distance_to_border(pp_part.position, pp_part.direction)
    outside = d0 < 0 and d1 < 0
    return not outside

def prep_dict(props):
    pass

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

entry_energy = np.array([entry[1][0] for entry in entries]) / 1e3
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
