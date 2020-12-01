import sys
import os
import os.path
base_path = os.environ['GOLEMSPACE']
core_path = base_path + '/sources/DUNEAtmo/likelihood/core/'
sys.path.insert(0, core_path)
analysis_path = base_path + '/sources/DUNEAtmo/likelihood/analysis/'
sys.path.insert(0, analysis_path)
import numpy as np
import json
import scipy
import scipy.optimize
import lv_analysis
import likelihood


default_asimov_params = {
    "operator_dimension": 3,
    "lv_emu_re": 0,
    "lv_emu_im": 0,
    "lv_mutau_re": 0,
    "lv_mutau_im": 0,
    "convNorm": 1.0,
    "CRDeltaGamma": 0.0,
}

re3_grid = np.logspace(-25, -22, 5*3+1)
im3_grid = np.logspace(-25, -22, 5*3+1)

re4_grid = np.logspace(-29, -26, 5*3+1)
im4_grid = np.logspace(-29, -26, 5*3+1)

def lv_2d_scan(asimov_params=default_asimov_params, re3_grid=re3_grid, im3_grid=im3_grid, re4_grid=re4_grid, im4_grid=im4_grid, output=output):
    the_store = analysis_lv.setup_lv_analysis()

    asimov_expect = the_store.get_prop("asimov_expect", asimov_params)
    def asimov_binned_likelihood(parameters):
        expect = the_store.get_prop("expect", parameters)
        expect_sq = the_store.get_prop("expect_sq", parameters)
        return likelihood.LEff(asimov_expect, expect, expect_sq)

    def asimov_likelihood(parameters):
        return np.sum(asimov_binned_likelihood(parameters))

    def eval_priors(params, priors):
        PriorLLH = 0.0
        for i, (param, prior) in enumerate(zip(params, priors)):
            mu, sigma, low, high = prior
            if param > high or param < low or np.isinf(param):
                PriorLLH = -np.inf
                break
            if mu == None:
                continue

            LLH = -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - (param - mu)**2 / (2.0 * sigma ** 2)
            PriorLLH += LLH
        if np.isnan(PriorLLH):
            print(params)
            print(priors)
        assert(not np.isnan(PriorLLH))
        return PriorLLH

    priors = [
            (1.0, 0.05, 0.0, np.inf), # convNorm prior
            (0.0, 0.01, -np.inf, np.inf), # CRDeltaGamma prior
            ]

    entries = []
    physical_params = dict(asimov_params)
    print("Setting up fit:")
    print("\tlv_mutau_re  =", 0)
    print("\tlv_mutau_im =", 0)
    def f(x):
        convNorm, CRDeltaGamma = x
        p = {"convNorm": convNorm, "CRDeltaGamma": CRDeltaGamma}
        physical_params.update(p)
        prior = eval_priors((convNorm, CRDeltaGamma), priors)
        if np.isinf(prior):
            return prior
        else:
            return -(prior + asimov_likelihood(physical_params))
    res = scipy.optimize.minimize(f, [1.0, 0.0], bounds=[priors[0][2:4], priors[1][2:4]], method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-18})
    convNorm, CRDeltaGamma = res.x
    llh = res.fun
    entry = {
        "llh": llh,
        "operator_dimension": 3,
        "lv_emu_re": 0,
        "lv_emu_im": 0,
        "lv_mutau_re": 0,
        "lv_mutau_im": 0,
        "convNorm": convNorm,
        "CRDeltaGamma": CRDeltaGamma,
    }
    entries.append(entry)
    json_file = open(output, "w")
    json.dump(entries, json_file)
    json_file.close()
    print("\tfit convNorm     =", convNorm)
    print("\tfit CRDeltaGamma =", CRDeltaGamma)
    print("\tLLH =", llh)
    print()

    pairs = []
    for re3 in re3_grid:
        for im3 in im3_grid:
            pairs.append((3, re3, im3))

    for re4 in re4_grid:
        for im4 in im4_grid:
            pairs.append((4, re4, im4))
    order = np.arange(len(pairs))
    np.random.shuffle(order)
    for i in order:
        operator_dimension, lv_mutau_re, lv_mutau_im = pairs[i]
        print("Setting up fit:")
        print("\toperator_dimension =", operator_dimension)
        print("\tlv_mutau_re =", lv_mutau_re)
        print("\tlv_mutau_im =", lv_mutau_im)
        physical_params = {
            "operator_dimension": operator_dimension,
            "lv_emu_re": 0,
            "lv_emu_im": 0,
            "lv_mutau_re": lv_mutau_re,
            "lv_mutau_im": lv_mutau_im,
            "convNorm": 1.0,
            "CRDeltaGamma": 0.0,
        }
        def f(x):
            convNorm, CRDeltaGamma = x
            p = {"convNorm": convNorm, "CRDeltaGamma": CRDeltaGamma}
            physical_params.update(p)
            prior = eval_priors((convNorm, CRDeltaGamma), priors)
            if np.isinf(prior):
                res = -prior
            else:
                res0 = -prior
                res1 = -asimov_likelihood(physical_params)
                res = res0 + res1
            assert(not np.isnan(res))
            return res
        res = scipy.optimize.minimize(f, [1.0, 0.0], bounds=[priors[0][2:4], priors[1][2:4]], method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-18})
        convNorm, CRDeltaGamma = res.x
        llh = res.fun
        entry = {
                "llh": llh,
                "operator_dimension": operator_dimension,
                "lv_emu_re": 0,
                "lv_emu_im": 0,
                "lv_mutau_re": lv_mutau_re,
                "lv_mutau_im": lv_mutau_im,
                "convNorm": convNorm,
                "CRDeltaGamma": CRDeltaGamma,
                }
        entries.append(entry)
        json_file = open(output, "w")
        json.dump(entries, json_file)
        json_file.close()
        print("\tfit convNorm     =", convNorm)
        print("\tfit CRDeltaGamma =", CRDeltaGamma)
        print("\tLLH =", llh)
        print()
    json_file = open(output, "w")
    json.dump(entries, json_file)
    json_file.close()

