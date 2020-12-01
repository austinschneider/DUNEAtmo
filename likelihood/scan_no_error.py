import numpy as np
import json
import scipy
import scipy.optimize
import analysis
import likelihood

the_store = analysis.setup_sterile_analysis()

asimov_params = {
    "numnu": 3,
    "dm2": 0.0,
    "th14": 0.0,
    "th24": 0.0,
    "th34": 0.0,
    "cp": 0.0,
    "convNorm": 1.0,
    "CRDeltaGamma": 0.0,
}

dm2_grid = np.logspace(-1, 1, 10*2+1)
th14 = 0.0
s22th24_grid = np.logspace(-2, 0, 10*2+1)
th24_grid = np.arcsin(np.sqrt(s22th24_grid)) / 2.0
th34 = 0.0
cp = 0.0

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
        if param > high or param < low:
            PriorLLH = -np.inf
            break
        if mu == None:
            continue

        LLH = -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - (param - mu)**2 / (2.0 * sigma ** 2)
        PriorLLH += LLH
    return PriorLLH

priors = [
        (1.0, 0.05, 0.0, np.inf), # convNorm prior
        (0.0, 0.05, -np.inf, np.inf), # CRDeltaGamma prior
        ]

entries = []
physical_params = dict(asimov_params)
print("Setting up fit:")
print("\tdm2  =", 0)
print("\tth24 =", 0)
def f(x):
    convNorm, CRDeltaGamma = x
    p = {"convNorm": convNorm, "CRDeltaGamma": CRDeltaGamma}
    physical_params.update(p)
    prior = eval_priors((convNorm, CRDeltaGamma), priors)
    if np.isinf(prior):
        return -prior
    else:
        return -(prior + asimov_likelihood(physical_params))
#res = scipy.optimize.minimize(f, [1.0, 0.0], bounds=[priors[0][2:4], priors[1][2:4]], method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-18})

convNorm, CRDeltaGamma = [1.0, 0.0]
llh = f([1.0, 0.0])
entry = {
        "llh": llh,
        "dm2": 0,
        "th24": 0,
        "convNorm": convNorm,
        "CRDeltaGamma": CRDeltaGamma,
        }
entries.append(entry)
json_file = open("sterile_scan.json", "w")
json.dump(entries, json_file)
json_file.close()
print("\tfit convNorm     =", convNorm)
print("\tfit CRDeltaGamma =", CRDeltaGamma)
print("\tLLH =", llh)
print()

pairs = []
for dm2 in dm2_grid:
    for th24 in th24_grid:
        pairs.append((dm2, th24))
order = np.arange(len(pairs))
np.random.shuffle(order)
for i in order:
    dm2, th24 = pairs[i]
    print("Setting up fit:")
    print("\tdm2  =", dm2)
    print("\tth24 =", th24)
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
        if np.isinf(prior):
            return prior
        else:
            return -(prior + asimov_likelihood(physical_params))
    #res = scipy.optimize.minimize(f, [1.0, 0.0], bounds=[priors[0][2:4], priors[1][2:4]], method="L-BFGS-B", options={'ftol': 1e4*np.finfo(float).eps, 'gtol': 1e-18})
    convNorm, CRDeltaGamma = [1.0, 0.0]
    llh = f([1.0, 0.0])
    entry = {
            "llh": llh,
            "dm2": dm2,
            "th24": th24,
            "convNorm": convNorm,
            "CRDeltaGamma": CRDeltaGamma,
            }
    entries.append(entry)
    json_file = open("sterile_scan_no_errors.json", "w")
    json.dump(entries, json_file)
    json_file.close()
    print("\tfit convNorm     =", convNorm)
    print("\tfit CRDeltaGamma =", CRDeltaGamma)
    print("\tLLH =", llh)
    print()
json_file = open("sterile_scan_no_errors.json", "w")
json.dump(entries, json_file)
json_file.close()

