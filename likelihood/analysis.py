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

def load_mc():
    import data_loader
    print("Loading the mc")
    mc = data_loader.load_data('../weighted/weighted.json')
    return mc
the_store.add_prop('mc', None, load_mc, cache_size=1)
the_store.initialize()
mc = the_store.get_prop('mc')
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

def flux_conv(flux, mc_energy, mc_zenith, mc_particle_type):
    res = np.empty(mc_energy.shape)
    mc_nsq_particle_type = np.array(mc_particle_type < 0).astype(int)
    mc_nsq_flavor = (np.abs(mc_particle_type) / 2 - 6).astype(int)
    for i, (ptype, flavor, energy, zenith) in enumerate(zip(
        mc_nsq_particle_type,
        mc_nsq_flavor,
        mc_energy,
        mc_zenith)):
        res[i] = flux.EvalFlavor(int(flavor), float(np.cos(zenith)), float(energy*units.GeV), int(ptype))
    return res
the_store.add_prop('flux_conv', ['nsq_flux', 'mc_energy', 'mc_zenith', 'mc_particle'], flux_conv)

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
        }

the_store.get_prop("flux_conv", physical_params)

