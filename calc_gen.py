import numpy as np
import LWpy
import LeptonInjector
import json

outdir = './plots/oscillations_alt/'

s = LWpy.read_stream('./injected/config_DUNE.lic')
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

json_data = json.load(open('./propagated/propagated.json', 'r'))
energy = np.array(json_data["energy"])
zenith = np.array(json_data["zenith"])
azimuth = np.array(json_data["azimuth"])
bjorken_x = np.array(json_data["bjorken_x"])
bjorken_y = np.array(json_data["bjorken_y"])
final_type_0 = np.array(json_data["final_type_0"]).astype(int)
final_type_1 = np.array(json_data["final_type_1"]).astype(int)
particle = np.array(json_data["particle"]).astype(int)
x = np.array(json_data["x"])
y = np.array(json_data["y"])
z = np.array(json_data["z"])
total_column_depth = np.array(json_data["total_column_depth"])

props = np.empty(len(energy), dtype=[
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
      ])

props["energy"] = energy
props["zenith"] = zenith
props["azimuth"] = azimuth
props["bjorken_x"] = bjorken_x
props["bjorken_y"] = bjorken_y
props["final_type_0"] = final_type_0
props["final_type_1"] = final_type_1
props["particle"] = particle
props["x"] = x
props["y"] = y
props["z"] = z
props["total_column_depth"] = total_column_depth
def print_stats(name, v, z):
    return
    m0 = np.cos(z) < 0
    m1 = np.cos(z) > 0
    v0 = v[m0]
    v1 = v[m1]
    v_min0 = np.amin(v0)
    v_max0 = np.amax(v0)
    v_mean0 = np.sum(v0)/len(v0)
    v_stddev0 = np.sqrt(np.sum((v0-v_mean0)**2)/len(v0))

    v_min1 = np.amin(v1)
    v_max1 = np.amax(v1)
    v_mean1 = np.sum(v1)/len(v1)
    v_stddev1 = np.sqrt(np.sum((v1-v_mean1)**2)/len(v1))
    print(name + ":")
    print("\t"+"min:", v_min0, v_min1)
    print("\t"+"max:", v_max0, v_max1)
    print("\t"+"mean:", v_mean0, v_mean1)
    print("\t"+"stddev:", v_stddev0, v_stddev1)
    print()

nu_interactions_list = LWpy.get_standard_interactions()
int_model = LWpy.interaction_model(nu_interactions_list, earth_model_params)
gen_prob = np.zeros(len(props))
for i, gen in enumerate(generators):
    print(gen.block_type)
    p = gen.prob_final_state(props)
    print_stats("final state", p, zenith)
    pp = gen.prob_stat(props)
    p *= pp
    #print("stat:", pp)
    nonzero = p != 0
    if np.any(nonzero):
        pp = gen.prob_dir(props[nonzero])
        print_stats("direction", pp, zenith[nonzero])
        p[nonzero] *= pp
        nonzero = p != 0
    if np.any(nonzero):
        pp = gen.prob_e(props[nonzero])
        print_stats("energy", pp, zenith[nonzero])
        p[nonzero] *= pp
        nonzero = p != 0
    if np.any(nonzero):
        pp = gen.prob_area(props[nonzero])
        if type(pp) is float:
            print("area:", pp)
        else:
            print_stats("area", pp, zenith[nonzero])
        p[nonzero] *= pp
        nonzero = p != 0
    if np.any(nonzero):
        pp = gen.prob_pos(props[nonzero])
        print_stats("pos", pp, zenith[nonzero])
        p[nonzero] *= pp
        nonzero = p != 0
    if np.any(nonzero):
        pp = gen.prob_kinematics(props[nonzero])
        print_stats("kinematics", pp, zenith[nonzero])
        p[nonzero] *= pp
        nonzero = p != 0
    if np.any(nonzero):
        first_pos, last_pos = gen.get_considered_range(props[nonzero])
        pos_prob = int_model.prob_pos(props[nonzero], first_pos, last_pos)
        int_prob = int_model.prob_interaction(props[nonzero], first_pos, last_pos)
        print_stats("physical pos", pos_prob, zenith[nonzero])
        print_stats("physical int", int_prob, zenith[nonzero])
        p[nonzero] /= pos_prob * int_prob
    gen_prob += p
k_prob = int_model.prob_kinematics(props)
fs_prob = int_model.prob_final_state(props)
print_stats("physical kinematics", k_prob, zenith)
print_stats("physical final state", fs_prob, zenith)
gen_prob /= k_prob * fs_prob
print_stats("gen_prob", gen_prob, zenith)
print_stats("1/gen_prob", 1./gen_prob, zenith)

json_data["gen_prob"] = gen_prob.tolist()

f = open('./weighted/weighted.json', 'w')
json.dump(json_data, f)
f.close()
