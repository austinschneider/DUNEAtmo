import os
import os.path
import collections
import functools
import numpy as np
import nuflux
import nuSQUIDSpy as nsq
import nuSQUIDSTools
import oscillator
import prop_store
import likelihood


### How to build an analysis
## Instantiate the store
#
# the_store = prop_store.store()
#
## The store computes values for you and caches the result so you don't compute anything twice.
## If you need a computed value, you go to the store:
#
# my_expensive_numbers = the_store.get_prop("expensive_numbers", physics_parameters)
#
## But first you have to tell the store how to do things.
## So you define a function:
#
# def my_func(foo, bar, physics_param):
#     ret = ...
#     ... compute things ...
#     return ret
#
## Once defined, you must register the function with the store.
## This requires a name for the output of the function,
## the ordered names of parameters it depends on,
## and the function itself.
#
# the_store.add_prop("my_value", ["value_of_foo", "value_of_bar", "physics_value"], my_func)
#
## Now you can register other functions that use the output and so on...
#
# def my_other_func(my_value):
#     return (my_value + 1)/2.0
# the_store.add_prop("my_other_value", ["my_value"], my_other_func)
#
## This implicitly depends on "physics_value", but that is handled by the store.
## Just please don't define an infinite loop via interdependency...
#
## Finally you have to initialize the store so it can work out the implicit dependencies of the
## things you defined, figure out what the physics parameters are, and spin up the object caches
#
# the_store.initialize()
#
## If you are reinitializing the store after adding props and you want to keep the caches
#
# the_store.initialize(keep_cache=True)
#
## Now you can ask the store for values as long as you give it the appropriate physics parameters
## It will work out all the details and try not to recompute anything if it can help it
#
# physics_parameters = {"physics_value": np.pi/4.}
# value = the_store.get_prop("my_value", physics_parameters)

def setup_lv_analysis():
    units = nsq.Const()
    ebins = np.logspace(1, 6, 100 + 1) * units.GeV
    czbins = np.linspace(-1, 0, 100 + 1)

    flux = nuflux.makeFlux("H3a_SIBYLL23C")
    osc = oscillator.oscillator(
        "H3a_SIBYLL23C", flux, ebins, czbins, "lv", "./fluxes/", cache_size=10
    )

    the_store = prop_store.store()


    # How to load the MC with precomputed generation probabilities
    # This actually has the binning as well
    def load_mc():
        import data_loader
        import binning

        print("Loading the mc")
        mc = data_loader.load_data("../weighted/weighted.json")
        tg_emin = 1e2,
        tg_emax = 1e5,
        start_emin = 1e2,
        start_emax = 1e5,
        tg_ebins = 30
        start_ebins = 30
        tg_czbins = 20
        start_czbins = 40
        mc, bin_slices = binning.bin_data(mc, tg_ebins=tg_ebins, start_ebins=start_ebins, tg_czbins=tg_czbins, start_czbins=start_czbins, tg_emin=tg_emin, tg_emax=tg_emax, start_emin=start_emin, start_emax=start_emax)
        return mc, bin_slices
    the_store.add_prop("sorted_mc", None, load_mc, cache_size=1)

    # Actually get just the MC
    def get_mc(sorted_mc):
        return sorted_mc[0]
    the_store.add_prop("mc", ["sorted_mc"], get_mc, cache_size=1)

    # Just get the binning
    def get_binning(sorted_mc):
        return sorted_mc[1]
    the_store.add_prop("mc_binning", ["sorted_mc"], get_binning, cache_size=1)
    the_store.initialize()

    # Force loading of the MC
    # So we can look at what is inside
    mc = the_store.get_prop("mc")
    binning = the_store.get_prop("mc_binning")

    # Produce an accessor function
    def get_mc_f(name):
        s = str(name)

        def f(mc):
            return mc[s]

        return f

    # Convenience aliases for the MC parameters
    # Things like "mc_energy", "mc_zenith", etc. are registered with "mc" as a dependent
    for name in mc.dtype.names:
        s = str(name)
        f = get_mc_f(s)
        the_store.add_prop("mc_" + s, ["mc"], f, cache_size=1)

    # Convenience function for calling the nusquids flux repository
    # This nusquids repository does the 3+1 scenario, but the concept is easily extensible
    def nsq_flux(operator_dimension, lv_emu_re, lv_emu_im, lv_mutau_re, lv_mutau_im):
        flux = osc[(operator_dimension, lv_emu_re, lv_emu_im, lv_mutau_re, lv_mutau_im)]
        return flux
    the_store.add_prop("nsq_flux", ["operator_dimension", "lv_emu_re", "lv_emu_im", "lv_mutau_re", "lv_mutau_im"], nsq_flux)

    # How to get the flux for MC events from nusquids
    # That is before we modify it with additional parameters
    def baseline_flux_conv(flux, mc_energy, mc_zenith, mc_particle_type):
        res = np.empty(mc_energy.shape)
        mc_nsq_particle_type = np.array(mc_particle_type < 0).astype(int)
        mc_nsq_flavor = (np.abs(mc_particle_type) / 2 - 6).astype(int)
        for i, (ptype, flavor, energy, zenith) in enumerate(
            zip(mc_nsq_particle_type, mc_nsq_flavor, mc_energy, mc_zenith)
        ):
            if mc_particle_type[i] == 0:
                print(i, mc_energy[i], mc_zenith[i], mc_particle_type[i])
            res[i] = flux.EvalFlavor(
                int(flavor), float(np.cos(zenith)), float(energy * units.GeV), int(ptype),
                True
            )
            assert(not np.isnan(res[i]))
        assert(not np.any(np.isnan(res)))
        return res


    the_store.add_prop(
        "baseline_flux_conv",
        ["nsq_flux", "mc_energy", "mc_zenith", "mc_particle"],
        baseline_flux_conv,
    )

    # Now we can modify the flux according to cosmic ray uncertainties
    def conv_tilt_correction(mc_energy, CRDeltaGamma, pivot_energy=500):
        res = (mc_energy / pivot_energy) ** -CRDeltaGamma
        if np.any(np.isnan(res)):
            print(CRDeltaGamma, pivot_energy)
        assert(not np.any(np.isnan(res)))
        return res


    the_store.add_prop(
        "conv_tilt_correction", ["mc_energy", "CRDeltaGamma"], conv_tilt_correction
    )

    # Combining the baseline flux and the cosmic ray correction
    def conv_flux_tilt_corrected(baseline_flux_conv, conv_tilt_correction):
        res = baseline_flux_conv * conv_tilt_correction
        assert(not np.any(np.isnan(res)))
        return res


    the_store.add_prop(
        "conv_flux_tilt_corrected",
        ["baseline_flux_conv", "conv_tilt_correction"],
        conv_flux_tilt_corrected,
    )

    # Changing the conventional normalization
    def flux_conv(convNorm, conv_flux_tilt_corrected):
        res = convNorm * conv_flux_tilt_corrected
        assert(not np.any(np.isnan(res)))
        return res
    the_store.add_prop("flux_conv", ["convNorm", "conv_flux_tilt_corrected"], flux_conv)

    # Just the sample livetime
    def get_livetime():
        return 365.25 * 24 * 3600 * 9
    the_store.add_prop("livetime", None, get_livetime)

    # Combine the livetime and generation probability
    # We do this because neither will ever change
    def livetime_gen_prob(livetime, mc_gen_prob):
        return livetime / mc_gen_prob
    the_store.add_prop("livetime_gen_prob", ["livetime", "mc_gen_prob"], livetime_gen_prob)

    # Compute the weight of MC events
    def mc_weight(flux_conv, livetime_gen_prob):
        res = flux_conv * livetime_gen_prob
        assert(not np.any(np.isnan(res)))
        return res
    the_store.add_prop("mc_weight", ["flux_conv", "livetime_gen_prob"], mc_weight)

    # Bin the weights
    def binned_mc_weight(mc_weight, mc_binning):
        return [mc_weight[b] for b in mc_binning]
    the_store.add_prop("binned_mc_weight", ["mc_weight", "mc_binning"], binned_mc_weight)

    # Compute the MC expectation per bin
    def expect(binned_mc_weight):
        res = [np.sum(w) for w in binned_mc_weight]
        assert(not np.any(np.isnan(res)))
        return res
    the_store.add_prop("expect", ["binned_mc_weight"], expect)

    # Compute the MC square expectation per bin
    def expect_sq(binned_mc_weight):
        res = [np.sum(w ** 2) for w in binned_mc_weight]
        assert(not np.any(np.isnan(res)))
        return res
    the_store.add_prop("expect_sq", ["binned_mc_weight"], expect_sq)

    # Compute the weights for an asimov dataset
    # This is kept separate so that it has its own cache
    def asimov_data(mc_weight):
        return mc_weight
    the_store.add_prop("asimov_data", ["mc_weight"], asimov_data)


    def binned_asimov_data(asimov_data, mc_binning):
        return [asimov_data[b] for b in mc_binning]
    the_store.add_prop(
        "binned_asimov_data", ["asimov_data", "mc_binning"], binned_asimov_data
    )


    def asimov_expect(binned_asimov_data):
        res = [np.sum(w) for w in binned_asimov_data]
        assert(not np.any(np.isnan(res)))
        return res
    the_store.add_prop("asimov_expect", ["binned_asimov_data"], asimov_expect)


    def asimov_expect_sq(binned_asimov_data):
        res = [np.sum(w ** 2) for w in binned_asimov_data]
        assert(not np.any(np.isnan(res)))
        return res
    the_store.add_prop("asimov_expect_sq", ["binned_asimov_data"], asimov_expect_sq)

    # Spin up the caches
    the_store.initialize(keep_cache=True)

    return the_store

if __name__ == "__main__":
    the_store = setup_lv_analysis()

    # Now we can define the likelihood
    # For now we are ignoring the fact that we could have data (asimov only)

    def asimov_binned_likelihood(parameters, asimov_parameters):
        asimov_expect = the_store.get_prop("asimov_expect", asimov_parameters)
        expect = the_store.get_prop("expect", parameters)
        expect_sq = the_store.get_prop("expect_sq", parameters)
        return likelihood.LEff(asimov_expect, expect, expect_sq)


    def asimov_likelihood(parameters, asimov_parameters):
        return -np.sum(asimov_binned_likelihood(parameters, asimov_parameters))

    operator_dimension = 3
    lv_emu_re = 0
    lv_emu_im = 0
    lv_mutau_re = 1e-22
    lv_mutau_im = 1e-22

    physical_params = {
        "operator_dimension": operator_dimension,
        "lv_emu_re": lv_emu_re,
        "lv_emu_im": lv_emu_im,
        "lv_mutau_re": lv_mutau_re,
        "lv_mutau_im": lv_mutau_im,
        "convNorm": 1.0,
        "CRDeltaGamma": 0.0,
    }
    asimov_params = {
        "operator_dimension": 3,
        "lv_emu_re": 0,
        "lv_emu_im": 0,
        "lv_mutau_re": 0,
        "lv_mutau_im": 0,
        "convNorm": 1.0,
        "CRDeltaGamma": 0.0,
    }

    # Compute the delta LLH between the null (3 neutrino) and an alternative (3+1) scenario
    print(asimov_likelihood(physical_params, asimov_params))
    print(asimov_likelihood(asimov_params, asimov_params))
    print(
        asimov_likelihood(physical_params, asimov_params) -
        asimov_likelihood(asimov_params, asimov_params)
    )

    # Now you can hook this up to a minimizer, etc. ...

