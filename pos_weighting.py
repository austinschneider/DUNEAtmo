import numpy as np
import LWpy
import LeptonInjector
import json
import common
import matplotlib
import matplotlib.pyplot as plt
import os

base_outdir = './plots/pos_weighting/'
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

def plot_weight(name, w, nonzero=None):
    if nonzero is not None:
        weights = np.zeros(len(nonzero))
        weights[nonzero] = w
    else:
        nonzero = np.ones(len(w)).astype(bool)
        weights = w

    energy_bins = np.logspace(-1, 5, 60+1)
    zenith_bins = np.arccos(np.linspace(-1,1,50+1))[::-1]

    masks = common.get_bin_masks(energy, zenith, energy_bins, zenith_bins)
    expect_funcs = [
            lambda w, m: np.sum(w[m])/np.sum(m),
            lambda w, m: np.amin(w[m]) if len(w[m])>0 else 0,
            lambda w, m: np.amax(w[m]) if len(w[m])>0 else 0,
            lambda w, m: np.sum(w[m]),
            lambda w, m: np.sum(1./w[m]),
            ]
    suffixes = ['avg', 'min', 'max', 'sum', 'invsum']
    expects = []
    if len(w) > 0:
        for f in expect_funcs:
            expect = np.array([f(weights,np.logical_and(m, nonzero)) for m in masks]).reshape((len(zenith_bins)-1, len(energy_bins)-1)).T
            expects.append(expect)
    else:
        for f in expect_funcs:
            expect = np.zeros((len(zenith_bins)-1, len(energy_bins)-1)).T
            expects.append(expect)

    for suffix, expect in zip(suffixes, expects):
        cm = plt.get_cmap('plasma_r')
        cm.set_under('white')
        norm = matplotlib.colors.LogNorm()
        fig, ax = plt.subplots(figsize=(7,5))
        X = np.cos(np.array([zenith_bins]*(len(energy_bins))))
        Y = np.array([energy_bins]*(len(zenith_bins))).T
        mesh = ax.pcolormesh(X, Y, expect, cmap=cm, norm=norm)
        ax.set_yscale('log')
        ax.set_ylim((1e2, 1e5))
        #ax.set_xlim((-1,1))
        ax.set_ylabel('Neutrino Energy [GeV]')
        ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.set_ylabel(name)
        cb.ax.minorticks_on()
        plt.tight_layout()
        fig.savefig(outdir + 'weight_'+name+'_'+suffix+'.png', dpi=200)
        fig.clf()
        plt.close(fig)

plot_weight("total_column_depth", total_column_depth)

nu_interactions_list = LWpy.get_standard_interactions()
int_model = LWpy.interaction_model(nu_interactions_list, earth_model_params)
print(int_model.earthModel.PrintEarthParams())
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
        plot_weight("first_pos_radius_"+str(i), r, nonzero)
    if "last_pos" in probs and gen.block_type != 'VolumeInjectionConfiguration':
        last_pos = probs["last_pos"]
        nonzero = nonzeros["last_pos"]
        r = np.array([gen.earthModel.GetEarthCoordPosFromDetCoordPos(p).Magnitude() for p in last_pos])
        plot_weight("last_pos_radius_"+str(i), r, nonzero)
    if "position" in probs:
        prob = probs["position"]
        nonzero = nonzeros["position"]
        plot_weight("position_"+str(i), prob, nonzero)
    if "physical_position" in probs:
        prob = probs["physical_position"]
        nonzero = nonzeros["physical_position"]
        plot_weight("physical_position_"+str(i), prob, nonzero)
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

        plot_weight("position_ratio_"+str(i), phys_pos_prob/pos_prob, nonzero)

k_prob = int_model.prob_kinematics(props)
fs_prob = int_model.prob_final_state(props)
outdir = base_outdir
try:
    os.mkdir(outdir)
except:
    pass
print_stats("physical_kinematics", k_prob, zenith)
plot_weight("physical_kinematics", k_prob)
print_stats("physical_final_state", fs_prob, zenith)
plot_weight("physical_final_state", fs_prob)
gen_prob /= k_prob * fs_prob
print_stats("gen_prob", gen_prob, zenith)
plot_weight("gen_prob", gen_prob)
print_stats("1/gen_prob", 1./gen_prob, zenith)
plot_weight("inv_gen_prob", 1./gen_prob)

json_data["gen_prob"] = gen_prob.tolist()

f = open('weighted.json', 'w')
json.dump(json_data, f)
f.close()
