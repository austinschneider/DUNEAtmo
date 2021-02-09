import sys
import os
import os.path
base_path = os.environ['GOLEMSPACE']
core_path = base_path + '/sources/DUNEAtmo/likelihood/core/'
sys.path.insert(0, core_path)
analysis_path = base_path + '/sources/DUNEAtmo/likelihood/analysis/'
default_flux_path = '/n/holyscratch01/arguelles_delgado_lab/Lab/DUNEAnalysis/store/fluxes/'
default_weight_path = '/n/holyscratch01/arguelles_delgado_lab/Lab/DUNEAnalysis/store/simulation/weighted/'
sys.path.insert(0, analysis_path)
import numpy as np
import json
import scipy
import scipy.optimize
import lv_analysis_grad
import likelihood_grad
import autodiff as ad

import argparse                                                                                                            
parser = argparse.ArgumentParser(description="LV scan")
parser.add_argument('--flux-path',
        type=str,
        dest='flux_path',
        required=False,
        default=default_flux_path,
        )
parser.add_argument('--weight-path',
        type=str,
        dest='weight_path',
        required=False,
        default=default_weight_path,
        )
parser.add_argument('--chunks',
        type=int,
        dest="chunks",
        default=0,
        required=False,
        )
parser.add_argument('--chunk-number',
        type=int,
        dest="chunk_number",
        default=0,
        required=False,
        )
parser.add_argument('--output',
        type=str,
        dest='output',
        required=True
        )
args = parser.parse_args()

default_asimov_params = {
    "operator_dimension": 3,
    "lv_emu_re": 0,
    "lv_emu_im": 0,
    "lv_mutau_re": 0,
    "lv_mutau_im": 0,
    "lv_etau_re": 0,
    "lv_etau_im": 0,
    "lv_ee": 0,
    "lv_mumu": 0,
    "convNorm": 1.0,
    "CRDeltaGamma": 0.0,
}

def build_grid(rho, f1, f2):
    one = np.ones((len(rho), len(f1), len(f2)))
    a = one*rho[:,None,None]*f1[None,:,None]
    b = one*rho[:,None,None]*f2[None,None,:]*np.sqrt(1-f1**2)[None,:,None]
    c = one*rho[:,None,None]*(np.sqrt(1-f2**2)[None,None,:])*np.sqrt(1-f1**2)[None,:,None]
    points = np.array([a.flatten(), b.flatten(), c.flatten()]).T
    return np.unique(points, axis=0)

rho_3_grid = np.concatenate([[0.0], np.logspace(-25, -19, 5*6+1)])
f1_3_grid = np.linspace(-1, 1, 50+1)
f2_3_grid = np.linspace(0, 1, 25+1)

rho_4_grid = np.concatenate([[0.0], np.logspace(-29, -23, 5*6+1)])
f1_4_grid = np.linspace(-1, 1, 50+1)
f2_4_grid = np.linspace(0, 1, 25+1)

parameter_points = []

for diag3, re3, im3 in build_grid(rho_3_grid, f1_3_grid, f2_3_grid):
    params = (3, 0, 0, re3, im3, 0, 0, 0, diag3)
    parameter_points.append(params)

#for diag4, re4, im4 in build_grid(rho_4_grid, f1_4_grid, f2_4_grid):
#    params = (4, 0, 0, re4, im4, 0, 0, 0, diag4)
#    parameter_points.append(params)

items = parameter_points

if args.chunks > 0:
    a = int(np.floor(float(len(items)) / float(args.chunks)))
    b = int(np.ceil(float(len(items)) / float(args.chunks)))
    x = len(items) - a*args.chunks
    if args.chunk_number < x:
        n = b
        n0 = n*args.chunk_number
    else:
        n = a
        n0 = b*x + a*(args.chunk_number - x)
    n1 = min(n0 + n, len(items))
    if n0 >= len(items): 
        exit(0)
    items = items[n0:n1]

parameter_points = items
print(len(parameter_points), "points to process")

output = args.output

seeds = []
norm_seeds = np.linspace(0.9, 1.1, 2)
cr_seeds = np.linspace(-0.02, 0.02, 2)
seeds.append([1.0, 0.0])
#for ns in norm_seeds:
#    for crs in cr_seeds:
#        seeds.append([ns, crs])

def lv_2d_scan(asimov_params=default_asimov_params, parameter_points=parameter_points, output="./test.json", weight_path=default_weight_path, flux_path=default_flux_path):
    the_store = lv_analysis_grad.setup_lv_analysis(weight_path=weight_path, flux_path=flux_path)

    asimov_expect = the_store.get_prop("asimov_expect", asimov_params)

    def asimov_binned_likelihood(parameters):
        expect = the_store.get_prop("expect", parameters)
        expect_sq = the_store.get_prop("expect_sq", parameters)
        return likelihood_grad.LEff(asimov_expect, 
                ad.unpack(np.array(expect)), 
                ad.unpack(np.array(expect_sq)))


    def asimov_likelihood(parameters):
        return ad.sum(asimov_binned_likelihood(parameters))

    def eval_priors(params, priors):
        PriorLLH = 0.0
        PriorD = np.zeros(len(priors))
        for i, (param, prior) in enumerate(zip(params, priors)):
            mu, sigma, low, high = prior
            if param > high or param < low or np.isinf(param):
                PriorLLH = -np.inf
                break
            if mu == None:
                continue

            LLH = -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - (param - mu)**2 / (2.0 * sigma ** 2)
            PriorLLH += LLH
            PriorD[i] = -(param - mu) / sigma ** 2
        if np.isnan(PriorLLH):
            print(params)
            print(priors)
        assert(not np.isnan(PriorLLH))
        return PriorLLH, PriorD

    priors = [
            (1.0, 0.05, 0.0, np.inf), # convNorm prior
            (0.0, 0.01, -np.inf, np.inf), # CRDeltaGamma prior
            ]

    physical_params = dict(asimov_params)
    print("Setting up fit:")
    print("\tlv_mutau_re =", 0)
    print("\tlv_mutau_im =", 0)
    def f(x):
        convNorm, CRDeltaGamma = x
        p = {"convNorm": convNorm, "CRDeltaGamma": CRDeltaGamma}
        physical_params.update(p)
        prior = eval_priors((convNorm, CRDeltaGamma), priors)
        if np.isinf(prior[0]):
            return (-prior[0], -prior[1])
        else:
            llh = asimov_likelihood(physical_params)
            return (-(llh[0] + prior[0]), -(llh[1] + np.array(prior[1])))
            #return -(prior + asimov_likelihood(physical_params))
    best = None
    for seed in seeds:
        res = scipy.optimize.minimize(f, seed, bounds=[priors[0][2:4], priors[1][2:4]], method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-8, 'maxls': 20}, jac=True)
        #print("Evaluate at:", res.x)
        #print("Result:", f(res.x))
        if best is None or res.fun < best.fun:
            best = res
    convNorm, CRDeltaGamma = res.x
    llh = res.fun
    entry = {
        "llh": llh,
        "operator_dimension": 3,
        "lv_emu_re": 0,
        "lv_emu_im": 0,
        "lv_mutau_re": 0,
        "lv_mutau_im": 0,
        "lv_etau_re": 0,
        "lv_etau_im": 0,
        "lv_ee": 0,
        "lv_mumu": 0,
        "convNorm": convNorm,
        "CRDeltaGamma": CRDeltaGamma,
    }
    json_file = open(output, "w")
    json.dump(entry, json_file)
    json_file.write("\n")
    json_file.close()
    print("\tfit convNorm     =", convNorm)
    print("\tfit CRDeltaGamma =", CRDeltaGamma)
    print("\tLLH =", llh)
    print()

    order = np.arange(len(parameter_points))
    np.random.shuffle(order)
    for i in order:
        operator_dimension, lv_emu_re, lv_emu_im, lv_mutau_re, lv_mutau_im, lv_etau_re, lv_etau_im, lv_ee, lv_mumu = parameter_points[i]
        print("Setting up fit:")
        print("\toperator_dimension =", operator_dimension)
        print("\tlv_mutau_re =", lv_mutau_re)
        print("\tlv_mutau_im =", lv_mutau_im)
        print("\tlv_etau_re  =", lv_etau_re)
        print("\tlv_etau_im  =", lv_etau_im)
        print("\tlv_ee       =", lv_ee)
        print("\tlv_mumu     =", lv_mumu)
        
        physical_params = {
            "operator_dimension": operator_dimension,
            "lv_emu_re": 0,
            "lv_emu_im": 0,
            "lv_mutau_re": lv_mutau_re,
            "lv_mutau_im": lv_mutau_im,
            "lv_etau_re": lv_etau_re,
            "lv_etau_im": lv_etau_im,
            "lv_ee": lv_ee,
            "lv_mumu": lv_mumu,
            "convNorm": 1.0,
            "CRDeltaGamma": 0.0,
        }
        np.set_printoptions(precision=20)
        def f(x):
            convNorm, CRDeltaGamma = x
            p = {"convNorm": convNorm, "CRDeltaGamma": CRDeltaGamma}
            physical_params.update(p)
            prior = eval_priors((convNorm, CRDeltaGamma), priors)
            if np.isinf(prior[0]):
                res = (-prior[0], -prior[1])
            else:
                llh = asimov_likelihood(physical_params)
                res = (-(llh[0] + prior[0]), -(llh[1] + np.array(prior[1])))
            print(x, res)
            assert(not np.isnan(res[0]))
            return res
        best = None
        for seed in seeds:
            res = scipy.optimize.minimize(f, seed, bounds=[priors[0][2:4], priors[1][2:4]], method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-18, 'maxls': 20}, jac=True)
            #print("Evaluate at:", res.x)
            #print("Result:", f(res.x))
            if best is None or res.fun < best.fun:
                best = res
        res = best
        convNorm, CRDeltaGamma = res.x
        llh = res.fun
        entry = {
                "llh": llh,
                "operator_dimension": operator_dimension,
                "lv_emu_re": 0,
                "lv_emu_im": 0,
                "lv_mutau_re": lv_mutau_re,
                "lv_mutau_im": lv_mutau_im,
                "lv_etau_re": lv_etau_re,
                "lv_etau_im": lv_etau_im,
                "lv_ee": lv_ee,
                "lv_mumu": lv_mumu,
                "convNorm": convNorm,
                "CRDeltaGamma": CRDeltaGamma,
                }
        json_file = open(output, "a")
        json.dump(entry, json_file)
        json_file.write("\n")
        json_file.close()
        print("\tfit convNorm     =", convNorm)
        print("\tfit CRDeltaGamma =", CRDeltaGamma)
        print("\tLLH =", llh)
        #print("\tSuccess: ", res.success)
        #print("\tStatus: ", res.status)
        #print("\tMessage: ", res.message)
        #print("\tnit: ", res.nit)
        #print("\tnfev: ", res.nfev)
        print()

lv_2d_scan(asimov_params=default_asimov_params, parameter_points=parameter_points, output=args.output, weight_path=args.weight_path, flux_path=args.flux_path)
