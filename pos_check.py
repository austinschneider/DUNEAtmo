import numpy as np
import LWpy
import LeptonInjector
import json
import common
import matplotlib
import matplotlib.pyplot as plt
import os

base_outdir = './plots/pos_check/'
outdir = base_outdir

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
        print(gen.earthModel.PrintEarthParams())
    else:
        raise ValueError("Unrecognized block! " + block_name)
    generators.append(gen)

class accumulator:
    def __init__(self):
        self.arr = []
    def __mul__(self, x):
        self.arr.append(x)
        return self
    def __add__(self, x):
        self.arr = self.arr*x
        return self

class make_array(accumulator):
    def __rshift__(self, other):
        res = np.array(other.arr)
        other.arr = []
        return res

a = accumulator()
b = make_array()

n = 2
energy = b>>a*1e4+n
zenith = b>>a*np.pi+n
azimuth = b>>a*0.0+n
bjorken_x = b>>a*0.2+n
bjorken_y = b>>a*0.5+n
final_type_0 = (b>>a*13+n).astype(int)
final_type_1 = (b>>a*-2000001006+n).astype(int)
particle = (b>>a*14+n).astype(int)
x = b>>a*0.0+n
y = b>>a*0.0+n
z = b>>a*0.0*1500.0
print(z)
total_column_depth = b>>a*0.0+n

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

nu_interactions_list = LWpy.get_standard_interactions()
int_model = LWpy.interaction_model(nu_interactions_list, earth_model_params)
gen_prob = np.zeros(len(props))
for i, gen in enumerate(generators):
    outdir = base_outdir + gen.block_type + str(i) + '/'
    try:
        os.mkdir(outdir)
    except:
        pass
    print(gen.block_type)
    probs = dict()
    nonzeros = dict()

    p = gen.prob_final_state(props)
    name = "final_state"
    nonzeros[name] = None
    probs[name] = p

    pp = gen.prob_stat(props)
    name = "stat"
    nonzeros[name] = None
    probs[name] = pp
    p *= pp

    nonzero = p != 0
    if np.any(nonzero):
        pp = gen.prob_dir(props[nonzero])
        name = "direction"
        nonzeros[name] = nonzero
        probs[name] = pp

        p[nonzero] *= pp
        nonzero = p != 0
    if np.any(nonzero):
        pp = gen.prob_e(props[nonzero])
        name = "energy"
        nonzeros[name] = nonzero
        probs[name] = pp

        p[nonzero] *= pp
        nonzero = p != 0
    if np.any(nonzero):
        pp = gen.prob_area(props[nonzero])
        name = "area"
        nonzeros[name] = nonzero
        probs[name] = pp

        p[nonzero] *= pp
        nonzero = p != 0
    if np.any(nonzero):
        pp = gen.prob_pos(props[nonzero])
        name = "position"
        nonzeros[name] = nonzero
        probs[name] = pp

        p[nonzero] *= pp
        nonzero = p != 0
    if np.any(nonzero):
        pp = gen.prob_kinematics(props[nonzero])
        name = "kinematics"
        nonzeros[name] = nonzero
        probs[name] = pp

        p[nonzero] *= pp
        nonzero = p != 0
    if np.any(nonzero):
        first_pos, last_pos = gen.get_considered_range(props[nonzero])
        name = "first_pos"
        nonzeros[name] = nonzero
        probs[name] = first_pos
        name = "last_pos"
        nonzeros[name] = nonzero
        probs[name] = last_pos
        pos_prob = int_model.prob_pos(props[nonzero], first_pos, last_pos)
        name = "physical_position"
        nonzeros[name] = nonzero
        probs[name] = pos_prob
        int_prob = int_model.prob_interaction(props[nonzero], first_pos, last_pos)
        name = "physical_interaction"
        nonzeros[name] = nonzero
        probs[name] = int_prob

        p[nonzero] /= pos_prob * int_prob
    gen_prob += p
    if "first_pos" in probs and gen.block_type != 'VolumeInjectionConfiguration':
        first_pos = probs["first_pos"]
        nonzero = nonzeros["first_pos"]
        r = np.array([gen.earthModel.GetEarthCoordPosFromDetCoordPos(p).Magnitude() for p in first_pos])
        #plot_weight("first_pos_radius_"+str(i), r, nonzero)
    if "last_pos" in probs and gen.block_type != 'VolumeInjectionConfiguration':
        last_pos = probs["last_pos"]
        nonzero = nonzeros["last_pos"]
        r = np.array([gen.earthModel.GetEarthCoordPosFromDetCoordPos(p).Magnitude() for p in last_pos])
        #plot_weight("last_pos_radius_"+str(i), r, nonzero)
    if "position" in probs:
        prob = probs["position"]
        nonzero = nonzeros["position"]
        #plot_weight("position_"+str(i), prob, nonzero)
    if "physical_position" in probs:
        prob = probs["physical_position"]
        nonzero = nonzeros["physical_position"]
        #plot_weight("physical_position_"+str(i), prob, nonzero)
    if "position" in probs and "physical_position" in probs:
        pos_prob = probs["position"]
        phys_pos_prob = probs["physical_position"]

        pos_nonzero = nonzeros["position"]
        phys_pos_nonzero = nonzeros["physical_position"]

        big_pos = np.zeros(len(pos_nonzero))
        big_phys_pos = np.zeros(len(phys_pos_nonzero))

        big_pos[pos_nonzero] = pos_prob
        big_phys_pos[phys_pos_nonzero] = phys_pos_prob

        nonzero = np.logical_and(pos_nonzero, phys_pos_nonzero)
        pos_prob = big_pos[nonzero]
        phys_pos_prob = big_phys_pos[nonzero]

        #plot_weight("position_ratio_"+str(i), phys_pos_prob/pos_prob, nonzero)

k_prob = int_model.prob_kinematics(props)
fs_prob = int_model.prob_final_state(props)
outdir = base_outdir
try:
    os.mkdir(outdir)
except:
    pass
#print_stats("physical_kinematics", k_prob, zenith)
#plot_weight("physical_kinematics", k_prob)
#print_stats("physical_final_state", fs_prob, zenith)
#plot_weight("physical_final_state", fs_prob)
gen_prob /= k_prob * fs_prob
#print_stats("gen_prob", gen_prob, zenith)
#plot_weight("gen_prob", gen_prob)
#print_stats("1/gen_prob", 1./gen_prob, zenith)
#plot_weight("inv_gen_prob", 1./gen_prob)

#json_data["gen_prob"] = gen_prob.tolist()

#f = open('weighted.json', 'w')
#json.dump(json_data, f)
#f.close()
