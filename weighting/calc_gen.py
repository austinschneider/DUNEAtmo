import argparse

parser = argparse.ArgumentParser(description="Calculate generation probabilities")
parser.add_argument('--propagated',
        type=str,
        dest='propagated',
        required=True,
        )
parser.add_argument('--lic',
        type=str,
        dest='lic',
        required=True,
        )
parser.add_argument('--outdir',
        type=str,
        dest='outdir',
        required=True,
        )
parser.add_argument('--output',
        type=str,
        dest='output',
        required=True,
        )
args = parser.parse_args()

import os
import os.path
import numpy as np
import LWpy
import LeptonInjector
import json
import glob
base_path = os.environ['GOLEMSPACE']

gen_earth_model_params = [
    "DUNE",
    base_path + "/sources/LWpy/LWpy/resources/earthparams/",
    ["PREM_dune"],
    ["Standard"],
    "NoIce",
    20.0*LeptonInjector.Constants.degrees,
    1480.0*LeptonInjector.Constants.m]

lic_files = sorted(glob.glob(args.lic))
prop_files = sorted(glob.glob(args.propagated))

print("lic files:")
print(lic_files)
print()
print("prop_files:")
print(prop_files)
print()

by_prefix = dict()
for lic in lic_files:
    base = os.path.basename(lic)
    if not base.endswith('.lic'):
        continue
    prefix = base[:-len('.lic')]
    by_prefix[prefix] = lic

print(len(by_prefix))

joint_by_prefix = dict()
for prop in prop_files:
    base = os.path.basename(prop)
    if not base.endswith('.json'):
        continue
    prefix = base[:-len('.json')]
    if prefix not in by_prefix:
        continue
    joint_by_prefix[prefix] = (by_prefix[prefix], prop)

print(len(joint_by_prefix))

def load_gen(fname):
    s = LWpy.read_stream(fname, spline_dir='./splines/')
    blocks = s.read()
    return blocks

def blocks_to_gen(blocks):
    generators = []
    for block in blocks:
        block_name, block_version, _ = block
        if block_name == 'EnumDef':
            continue
        elif block_name == 'VolumeInjectionConfiguration':
            gen = LWpy.volume_generator(block, spline_dir='./splines/')
        elif block_name == 'RangedInjectionConfiguration':
            gen = LWpy.ranged_generator(block, gen_earth_model_params, spline_dir='./splines/')
        else:
            raise ValueError("Unrecognized block! " + block_name)
        generators.append(gen)
    return generators

def count_gen_events(blocks):
    n = 0
    for block in blocks:
        block_name, block_version, block_data = block
        print(block_name)
        if block_name == 'VolumeInjectionConfiguration' or block_name == 'RangedInjectionConfiguration':
            n += block_data["events"]
    return n

earth_model_params = [
    "DUNE",
    base_path + "/sources/LWpy/LWpy/resources/earthparams/",
    ["PREM_dune"],
    ["Standard"],
    "NoIce",
    20.0*LeptonInjector.Constants.degrees,
    1480.0*LeptonInjector.Constants.m]

nu_interactions_list = LWpy.get_standard_interactions()
int_model = LWpy.interaction_model(nu_interactions_list, earth_model_params)

lic_output = os.path.join(args.outdir, args.output + '.lic')
print("lic_output:", lic_output)

def merge_json(j0, j1):
    if j0 is None:
        return j1
    if j1 is None:
        return j0
    res = dict()
    keys = list(j0.keys())
    for k in keys:
        if k == "injector_count":
            continue
        v0 = j0[k]
        v1 = j1[k]
        res[k] = np.concatenate([v0, v1])
    v0 = j0["injector_count"]
    v1 = j1["injector_count"]
    c = [c1 for c0, c1 in zip(v0, v1)]
    res["injector_count"] = c
    
    return res

def read_file(fname):
    f = open(fname, "r")
    dec = json.JSONDecoder()
    data = None
    for json_str in f.readlines():
        pos = 0
        while not pos == len(str(json_str)):
            j, json_len = dec.raw_decode(str(json_str)[pos:])
            pos += json_len
            data = merge_json(data, j)
    return data

def read_files(fnames):
    data = None
    for fname in tqdm.tqdm(fnames):
        data = merge_json(data, read_file(fname))
    return data

all_blocks = []
if os.path.exists(lic_output):
    all_blocks = load_gen(lic_output)
else:
    for prefix, (lic, prop) in joint_by_prefix.items():
        json_data = read_file(prop)
        n_prop_events = np.sum(json_data["injector_count"])
        del json_data
        blocks = load_gen(lic)
        n_gen_events = count_gen_events(blocks)
        print("N gen:", n_gen_events, "N prop:", n_prop_events)
        if n_prop_events != n_gen_events:
            continue
        all_blocks.extend(blocks)
        print(len(all_blocks))
        all_blocks = LWpy.merge_blocks(all_blocks)
        print(len(all_blocks))
        print()

    s = LWpy.write_stream(os.path.join(args.outdir, args.output + '.lic'), spline_dir='./splines/')
    s.write(all_blocks)

print(all_blocks)
generators = blocks_to_gen(all_blocks)

def calc_gen_prob(json_data):
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

    gen_prob = np.zeros(len(props))
    for i, gen in enumerate(generators):
        p = gen.prob_final_state(props)
        pp = gen.prob_stat(props)
        p *= pp
        nonzero = p != 0
        if np.any(nonzero):
            pp = gen.prob_dir(props[nonzero])
            p[nonzero] *= pp
            nonzero = p != 0
        if np.any(nonzero):
            pp = gen.prob_e(props[nonzero])
            p[nonzero] *= pp
            nonzero = p != 0
        if np.any(nonzero):
            pp = gen.prob_area(props[nonzero])
            p[nonzero] *= pp
            nonzero = p != 0
        if np.any(nonzero):
            pp = gen.prob_pos(props[nonzero])
            p[nonzero] *= pp
            nonzero = p != 0
        if np.any(nonzero):
            pp = gen.prob_kinematics(props[nonzero])
            p[nonzero] *= pp
            nonzero = p != 0
        if np.any(nonzero):
            first_pos, last_pos = gen.get_considered_range(props[nonzero])
            pos_prob = int_model.prob_pos(props[nonzero], first_pos, last_pos)
            int_prob = int_model.prob_interaction(props[nonzero], first_pos, last_pos)
            p[nonzero] /= pos_prob * int_prob
        gen_prob += p
    k_prob = int_model.prob_kinematics(props)
    fs_prob = int_model.prob_final_state(props)
    gen_prob /= k_prob * fs_prob
    return gen_prob

#keep_keys = ['energy', 'zenith', 'azimuth', 'particle', 'mu_energy', 'mu_zenith', 'entry_energy', 'entry_zenith', 'morphology']
gen_prob = []
all_json = dict()
tot_prop_events = 0
for prefix, (lic, prop) in joint_by_prefix.items():
    json_data = read_file(prop)
    gen_prob.extend(calc_gen_prob(json_data).tolist())
    n_prop_events = np.sum(json_data["injector_count"])
    tot_prop_events += n_prop_events
    for k in json_data.keys():
        #if k not in keep_keys:
        #    continue
        if k not in all_json:
            all_json[k] = list(json_data[k])
        else:
            all_json[k] += list(json_data[k])

n_gen_events = count_gen_events(all_blocks)
if n_gen_events != tot_prop_events:
    print("Generated event count does not match propagated event count!!")
    print("Generated events:", n_gen_events)
    print("Propagated events:", tot_prop_events)
    print("Applying approximate rescaling!")
    gen_prob = (np.array(gen_prob) * n_gen_events / tot_prop_events).tolist()
    os.remove(lic_output)

json_data = all_json
json_data["gen_prob"] = gen_prob

f = open(os.path.join(args.outdir, args.output + '.json'), 'w')
json.dump(json_data, f)
f.close()
