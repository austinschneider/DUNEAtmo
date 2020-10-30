import os
import os.path
import collections
import numpy as np
import nuflux
import nuSQUIDSpy as nsq
import nuSQUIDSTools

flux = nuflux.makeFlux('H3a_SIBYLL23C')

units = nsq.Const()
dm2 = 1.
th14 = 0.
th24 = np.arcsin(np.sqrt(0.1))/2.
th34 = 0.0
cp = 0.
ebins = np.logspace(1, 6, 100+1)*units.GeV
czbins = np.linspace(-1, 1, 100+1)

import oscillator
osc = oscillator.oscillator('H3a_SIBYLL23C',flux,ebins,czbins,'sterile','./fluxes/',cache_size = 10)

import prop_store

the_store = prop_store.store()

# How to load the mc
def load_mc():
    import data_loader
    import binning
    print("Loading the mc")
    mc = data_loader.load_data('../weighted/weighted.json')
    mc, bin_slices = binning.bin_data(mc)
    return mc, bin_slices
the_store.add_prop('sorted_mc', None, load_mc, cache_size=1)

# Bin the mc and sort by
def get_mc(sorted_mc):
    return sorted_mc[0]
the_store.add_prop('mc', ['sorted_mc'], get_mc, cache_size=1)

def get_binning(sorted_mc):
    return sorted_mc[1]
the_store.add_prop('mc_binning', ['sorted_mc'], get_binning, cache_size=1)
the_store.initialize()

# Force loading of the MC
mc = the_store.get_prop('mc')
binning = the_store.get_prop('mc_binning')

# Convenience aliases for the MC parameters
def get_mc_f(name):
    s = str(name)
    def f(mc):
        return mc[s]
    return f
for name in mc.dtype.names:
    s = str(name)
    f = get_mc_f(s)
    the_store.add_prop('mc_'+s, ['mc'], f, cache_size=1)

def nsq_flux(numnu, dm2, th14, th24, th34, cp):
    flux = osc[(numnu, dm2, th14, th24, th34, cp)]
    return flux
the_store.add_prop('nsq_flux', ['numnu', 'dm2', 'th14', 'th24', 'th34', 'cp'], nsq_flux)

def baseline_flux_conv(flux, mc_energy, mc_zenith, mc_particle_type):
    print("baseline_flux_conv")
    res = np.empty(mc_energy.shape)
    mc_nsq_particle_type = np.array(mc_particle_type < 0).astype(int)
    mc_nsq_flavor = (np.abs(mc_particle_type) / 2 - 6).astype(int)
    for i, (ptype, flavor, energy, zenith) in enumerate(zip(
        mc_nsq_particle_type,
        mc_nsq_flavor,
        mc_energy,
        mc_zenith)):
        if mc_particle_type[i] == 0:
            print(i, mc_energy[i], mc_zenith[i], mc_particle_type[i])
        res[i] = flux.EvalFlavor(int(flavor), float(np.cos(zenith)), float(energy*units.GeV), int(ptype))
    return res
the_store.add_prop('baseline_flux_conv', ['nsq_flux', 'mc_energy', 'mc_zenith', 'mc_particle'], baseline_flux_conv)

def conv_tilt_correction(mc_energy, CRDeltaGamma, pivot_energy=500):
    return (mc_energy / pivot_energy) ** -CRDeltaGamma
the_store.add_prop('conv_tilt_correction', ['mc_energy', 'CRDeltaGamma'], conv_tilt_correction)

def conv_flux_tilt_corrected(baseline_flux_conv, conv_tilt_correction):
    return baseline_flux_conv * conv_tilt_correction
the_store.add_prop('conv_flux_tilt_corrected', ['baseline_flux_conv', 'conv_tilt_correction'], conv_flux_tilt_corrected)

def flux_conv(convNorm, conv_flux_tilt_corrected):
    return convNorm * conv_flux_tilt_corrected
the_store.add_prop('flux_conv', ["convNorm", "conv_flux_tilt_corrected"], flux_conv)

the_store.initialize(keep_cache=True)

dm2 = 1.
th14 = 0.
th24 = np.arcsin(np.sqrt(0.1))/2.
th34 = 0.0
cp = 0.

physical_params = {
        'numnu': 4,
        "dm2": dm2,
        "th14": th14,
        "th24": th24,
        "th34": th34,
        "cp": cp,
        "convNorm": 1.0,
        "CRDeltaGamma": 0.0,
        }

the_store.get_prop("flux_conv", physical_params)

