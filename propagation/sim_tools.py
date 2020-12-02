import proposal as pp
import numpy as np
import h5py as h5
import LWpy
import LeptonInjector
import json
from enum import Enum, IntEnum

def interpolate_energy(last_particle, particle, continuous_losses, distance, d):
    initial_energy = last_particle.energy
    if len(continuous_losses) > 1:
        final_energy = continuous_losses[-1].energy
    else:
        final_energy = particle.energy
    delta_energy = final_energy - initial_energy
    point_energy = delta_energy/distance*d + initial_energy
    return point_energy

def get_particle_intersections(geo, parent, particles):
    last_particle = parent
    d0, d1 = geo.distance_to_border(parent.position, parent.direction)
    last_inside = d0 >= 0 and d1 < 0
    last_k = -1

    entries = []
    exits = []

    track_length = 0
    deposited_energy = 0

    for j, particle in enumerate(particles):
        # Skip the continuous energy losses
        if int(particle.type) == int(pp.particle.Interaction_Type.ContinuousEnergyLoss):
            continue

        direction = particle.position - last_particle.position
        distance = direction.magnitude()
        direction.normalize()
        if distance == 0:
            direction = last_particle.direction

        # Skip if the particle didn't move, but update the references
        #if distance == 0:
        #    last_particle = particle
        #    last_k = j
        #    d0, d1 = geo.distance_to_border(particle.position, direction)
        #    last_inside = d0 > 0 and d1 < 0
        #    continue

        if last_inside:
            # We started inside the detector
            # Don't worry about the entry position
            # Either the event started in the detector
            # or the entry point was handled in the last iteration
            d0, d1 = geo.distance_to_border(last_particle.position, direction)
            # We're inside so there should only be one intersection
            assert(d0 >= 0 and d1 < 0)

            # Check if we go far enough to exit
            if distance >= d0: # Exited the detector
                track_length += d0
                ## Compute exit position and energy
                continuous_losses = particles[last_k+1:j]
                exit_energy = interpolate_energy(last_particle, particle, continuous_losses, distance, d0)
                exit_pos = last_particle.position + direction * d0
                exit_dir = direction
                exits.append((exit_energy, exit_pos, exit_dir))
                deposited_energy += last_particle.energy - exit_energy
            else: # Not far enough
                track_length += distance
                deposited_energy += last_particle.energy - particle.energy
                # Do nothing
                pass
            pass
        else:
            # We started outside the detector
            # Check if the path could intersect the detector
            d0, d1 = geo.distance_to_border(last_particle.position, direction)
            # At the initial point we should not be inside the detector
            # We should either be outside pointed away, or outside pointed towards
            assert(np.sign(d0) == np.sign(d1))
            if d0 > 0 and d1 > 0: # We're outside pointed towards
                # Check if we went far enough to enter the detector
                if distance >= d1: # Entered AND exited the detector
                    track_length += d1 - d0
                    ## Compute entry position and entry energy
                    continuous_losses = particles[last_k+1:j]
                    entry_energy = interpolate_energy(last_particle, particle, continuous_losses, distance, d0)
                    entry_pos = last_particle.position + direction * d0
                    entry_dir = direction
                    entries.append((entry_energy, entry_pos, entry_dir))
                    ## Compute exit position and exit energy
                    exit_energy = interpolate_energy(last_particle, particle, continuous_losses, distance, d1)
                    exit_pos = last_particle.position + direction * d1
                    exit_dir = direction
                    exits.append((exit_energy, exit_pos, exit_dir))
                    deposited_energy += entry_energy - exit_energy
                elif distance >= d0: # Entered the detector
                    track_length += distance - d0
                    ## Compute entry position and entry energy
                    continuous_losses = particles[last_k+1:j]
                    entry_energy = interpolate_energy(last_particle, particle, continuous_losses, distance, d0)
                    entry_pos = last_particle.position + direction * d0
                    entry_dir = direction
                    entries.append((entry_energy, entry_pos, entry_dir))
                    deposited_energy += entry_energy - particle.energy
                else: # Not far enough
                    # Do nothing
                    pass
            else: # We're outside pointed away
                # Do nothing
                pass

        # Update the last particle information
        last_particle = particle
        last_k = j
        d0, d1 = geo.distance_to_border(particle.position, direction)
        last_inside = d0 >= 0 and d1 < 0

    # There can only be one! (unless the muon scatters in and out of the detector...)
    if len(entries) > 1 or len(exits) > 1:
        print("WARNING: muon found skimming the surface of the detector! This should be extremely rare!!!")

    # return the first entry
    if len(entries) > 0:
        entry_ret = entries[0]
    else:
        entry_ret = None

    # return the last exit
    if len(exits) > 0:
        exit_ret = exits[-1]
    else:
        exit_ret = None

    return entry_ret, exit_ret, track_length, deposited_energy

def is_starting(geo, parent, particles):
    d0, d1 = geo.distance_to_border(parent.position, parent.direction)
    inside = d0 > 0 and d1 < 0
    return inside

class EventMorphology(IntEnum):
    missing = 0
    starting = 1
    stopping = 2
    through_going = 3
    contained = 4
    pseudo_contained = 5

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
    decay_products = [p for i,p in zip(range(max(len(particles)-3,0),len(particles)), particles[-3:]) if int(p.type) <= 1000000001]
    if len(decay_products) == 0:
        print([p.type for p in particles])
    else:
        particles = particles[:-len(decay_products)]
        dummy_particle = pp.particle.DynamicData(1000000001)
        dummy_particle.position = decay_products[-1].position
        if len(particles) > 0:
            dummy_particle.direction = particles[-1].direction
        else:
            dummy_particle.direction = parent.direction
        dummy_particle.energy = np.sum([p.energy for p in decay_products if abs(int(p.type)) in [12, 14, 16]])
        particles.append(dummy_particle)
    for geo in geos:
        starting = is_starting(geo, parent, particles)
        entry_info, exit_info, track_length, energy_in_detector = get_particle_intersections(geo, parent, particles)
        entering = entry_info is not None
        exiting = exit_info is not None

        through_going = entering and exiting
        stopping = (entering or starting) and (not exiting)
        missing = (not entering) and (not starting)

        starts |= starting
        stops |= stopping
        intersects |= not missing
        contained |= starting and (not exiting)

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

