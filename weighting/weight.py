import proposal as pp
import numpy as np
import h5py as h5
import LWpy
import LeptonInjector
import nuSQUIDSpy as nsq
import json

units = nsq.Const()
kaon = nsq.nuSQUIDSAtm("./kaon_atmospheric_final.hdf5")
pion = nsq.nuSQUIDSAtm("./pion_atmospheric_final.hdf5")
prompt = nsq.nuSQUIDSAtm("./prompt_atmospheric_final.hdf5")

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

data = json.load(open('propagated.json', 'r'))
energy = np.array(data["energy"])
zenith = np.array(data["zenith"])
azimuth = np.array(data["azimuth"])
bjorken_x = np.array(data["bjorken_x"])
bjorken_y = np.array(data["bjorken_y"])
final_type_0 = np.array(data["final_type_0"]).astype(int)
final_type_1 = np.array(data["final_type_1"]).astype(int)
particle = np.array(data["particle"]).astype(int)
x = np.array(data["x"])
y = np.array(data["y"])
z = np.array(data["z"])
total_column_depth = np.array(data["total_column_depth"])
entry_energy = np.array(data["entry_energy"])
entry_x = np.array(data["entry_x"])
entry_y = np.array(data["entry_y"])
entry_z = np.array(data["entry_z"])
entry_zenith = np.array(data["entry_zenith"])
entry_azimuth = np.array(data["entry_azimuth"])
track_length = np.array(data["track_length"])

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
      ("entry_energy", entry_energy.dtype),
      ("entry_x", entry_x.dtype),
      ("entry_y", entry_y.dtype),
      ("entry_z", entry_z.dtype),
      ("entry_zenith", entry_zenith.dtype),
      ("entry_azimuth", entry_azimuth.dtype),
      ("track_length", track_length.dtype),
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
data["entry_energy"] = entry_energy
data["entry_x"] = entry_x
data["entry_y"] = entry_y
data["entry_z"] = entry_z
data["entry_zenith"] = entry_zenith
data["entry_azimuth"] = entry_azimuth
data["track_length"] = track_length

props = data

nu_interactions_list = LWpy.get_standard_interactions()
int_model = LWpy.interaction_model(nu_interactions_list, earth_model_params)
gen_prob = np.zeros(len(props))
for gen in generators:
    prob = gen.prob(props)
    p = gen.prob_final_state(props)
    p *= gen.prob_stat(props)
    nonzero = p != 0
    p[nonzero] *= gen.prob_dir(props[nonzero])
    nonzero = p != 0
    p[nonzero] *= gen.prob_e(props[nonzero])
    nonzero = p != 0
    p[nonzero] *= gen.prob_area(props[nonzero])
    nonzero = p != 0
    p[nonzero] *= gen.prob_pos(props[nonzero])
    nonzero = p != 0
    p[nonzero] *= gen.prob_kinematics(props[nonzero])
    nonzero = p != 0
    first_pos, last_pos = gen.get_considered_range(props[nonzero])
    pos_prob = int_model.prob_pos(props[nonzero], first_pos, last_pos)
    int_prob = int_model.prob_interaction(props[nonzero], first_pos, last_pos)
    p[nonzero] /= pos_prob * int_prob
    gen_prob += p
k_prob = int_model.prob_kinematics(props)
fs_prob = int_model.prob_final_state(props)
#print(gen.prob_final_state(props))
#print(gen.prob_stat(props))
#print(gen.prob_dir(props))
#print(gen.prob_e(props))
#print(gen.prob_area(props))
#print(gen.prob_pos(props))
#print(gen.prob_kinematics(props))
#print(fs_prob)
#print(k_prob/gen.prob_kinematics(props))
#print(pos_prob/gen.prob_pos(props))
#print(track_length)
gen_prob /= k_prob * fs_prob

flux = np.empty(len(props))

flavors = (np.abs(particle) / 2 - 6).astype(int)

for i, (flavor, zenith, energy, particle_type) in enumerate(zip(flavors, props["zenith"], props["energy"], props["particle"])):
    flavor = int(flavor)
    zenith = float(zenith)
    energy = float(energy)
    particle_type = int(particle_type)
    particle_type = 0 if particle_type > 0 else 1
    prompt_flux = prompt.EvalFlavor(flavor, np.cos(zenith), energy*units.GeV, particle_type)
    pion_flux = pion.EvalFlavor(flavor, np.cos(zenith), energy*units.GeV, particle_type)
    kaon_flux = kaon.EvalFlavor(flavor, np.cos(zenith), energy*units.GeV, particle_type)
    flux[i] = prompt_flux + pion_flux + kaon_flux

livetime = 365.25 * 24 * 3600

mask = track_length > 2
mask = np.logical_and(mask, entry_energy > 100)
mask = np.logical_and(mask, entry_zenith > np.pi/2.)

w = flux * livetime / gen_prob
print(np.sum(w[mask]))

#first_pos, last_pos = gen.get_considered_range(props)
#phys_pos = int_model.prob_pos(props, first_pos, last_pos)
#gen_pos = gen.prob_pos(props)
#p_int = int_model.prob_interaction(props, first_pos, last_pos)
