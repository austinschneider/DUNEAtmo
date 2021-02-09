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
import sterile_analysis_grad
import likelihood_grad
import argparse
import autodiff as ad

parser = argparse.ArgumentParser(description="Sterile scan")
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
    "numnu": 3,
    "dm2": 0.0,
    "th14": 0.0,
    "th24": 0.0,
    "th34": 0.0,
    "cp": 0.0,
    "convNorm": 1.0,
    "CRDeltaGamma": 0.0,
}

dm2_grid = np.concatenate([[0.0], np.logspace(-1, 1, 10*2+1)])
th14 = 0.0
s22th24_grid = np.concatenate([[0.0], np.logspace(-2, 0, 10*2+1)])
th24_grid = np.arcsin(np.sqrt(s22th24_grid)) / 2.0
s22th34_grid = np.concatenate([[0.0], np.logspace(-2, 0, 10*2+1)])
th34_grid = np.arcsin(np.sqrt(s22th34_grid)) / 2.0
cp = 0.0

parameter_points = []

for dm2 in dm2_grid:
    for th24 in th24_grid:
        for th34 in th34_grid:
            parameter_points.append((dm2, th24, th34))

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

def sterile_3d_scan(asimov_params=default_asimov_params, parameter_points=parameter_points, output="./test.json", weight_path=default_weight_path, flux_path=default_flux_path):
    the_store = sterile_analysis_grad.setup_sterile_analysis()
    asimov_expect = the_store.get_prop("asimov_expect", asimov_params)
    def asimov_binned_likelihood(parameters):
        expect = the_store.get_prop("expect", parameters)
        expect_sq = the_store.get_prop("expect_sq", parameters)
        return likelihood_grad.LEff(asimov_expect, ad.unpack(np.array(expect)), ad.unpack(np.array(expect_sq)))

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
    print("\tdm2  =", 0)
    print("\tth24 =", 0)
    def f(x):
        convNorm, CRDeltaGamma = x
        p = {"convNorm": convNorm, "CRDeltaGamma": CRDeltaGamma}
        physical_params.update(p)
        prior = eval_priors((convNorm, CRDeltaGamma), priors)
        if np.isinf(prior[0]):
            res = (-prior[0], -prior[1])
            return prior
        else:
            llh = asimov_likelihood(physical_params)
            return (-(llh[0] + prior[0]), -(llh[1] + np.array(prior[1])))
    res = scipy.optimize.minimize(f, [1.0, 0.0], bounds=[priors[0][2:4], priors[1][2:4]], method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-18, 'maxls': 20}, jac=True)
    convNorm, CRDeltaGamma = res.x
    llh = res.fun
    entry = {
            "llh": llh,
            "dm2": 0,
            "th24": 0,
            "th34": 0,
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
        dm2, th24, th34 = parameter_points[i]
        print("Setting up fit:")
        print("\tdm2  =", dm2)
        print("\tth24 =", th24)
        print("\tth34 =", th34)
        physical_params = {
            "numnu": 4,
            "dm2": dm2,
            "th14": th14,
            "th24": th24,
            "th34": th34,
            "cp": cp,
            "convNorm": 1.0,
            "CRDeltaGamma": 0.0,
        }
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
        res = scipy.optimize.minimize(f, [1.0, 0.0], bounds=[priors[0][2:4], priors[1][2:4]], method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-18, 'maxls': 20}, jac=True)
        convNorm, CRDeltaGamma = res.x
        llh = res.fun
        entry = {
                "llh": llh,
                "dm2": dm2,
                "th24": th24,
                "th34": th34,
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

sterile_3d_scan(asimov_params=default_asimov_params, parameter_points=parameter_points, output=args.output, weight_path=args.weight_path, flux_path=args.flux_path)
