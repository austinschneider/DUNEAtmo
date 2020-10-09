import nuflux
import nuSQUIDSpy as nsq
import nuSQUIDSTools

import numpy as np
from scipy import integrate
import time

def get_nusquids_obj(numneu, dm2, th14, th24, th34, cp, flux):

    assert numneu == 3 or numneu == 4, "numneu must be equal to 3 or 4"

    units = nsq.Const()

    interactions = True

    Emin = 1.e1*units.GeV
    Emax = 1.e6*units.GeV
    czmin = -1.
    czmax = 1.

    cz_nodes = nsq.linspace(czmin, czmax, 101)
    energy_nodes = nsq.logspace(Emin, Emax, 101)

    nuSQ = nsq.nuSQUIDSAtm(cz_nodes, energy_nodes, numneu, nsq.NeutrinoType.both, interactions)

    nuSQ.Set_MixingAngle(0,1,0.563942)
    nuSQ.Set_MixingAngle(0,2,0.154085)
    nuSQ.Set_MixingAngle(1,2,0.785398)

    nuSQ.Set_SquareMassDifference(1,7.65e-05)
    nuSQ.Set_SquareMassDifference(2,0.00247)

    nuSQ.Set_CPPhase(0,2,cp)
    if numneu > 3:
        nuSQ.Set_SquareMassDifference(3,dm2)
        nuSQ.Set_MixingAngle(0,3, th14)
        nuSQ.Set_MixingAngle(1,3, th24)
        nuSQ.Set_MixingAngle(2,3, th34)

    nuSQ.Set_rel_error(1.0e-08)
    nuSQ.Set_abs_error(1.0e-08)
    # nuSQ.Set_GSL_step(gsl_odeiv2_step_rk4)

    inistate = np.zeros( (nuSQ.GetNumCos(), nuSQ.GetNumE(), nuSQ.GetNumRho(), nuSQ.GetNumNeu()) )

    e_range = nuSQ.GetERange()
    cz_range = nuSQ.GetCosthRange()

    for ci in range(nuSQ.GetNumCos()):
        for ei in range(nuSQ.GetNumE()):

            inistate[ci][ei][0][0] = flux.getFlux(nuflux.NuE, e_range[ei]/units.GeV, cz_range[ci])
            inistate[ci][ei][0][1] = flux.getFlux(nuflux.NuMu, e_range[ei]/units.GeV, cz_range[ci])
            inistate[ci][ei][0][2] = flux.getFlux(nuflux.NuTau, e_range[ei]/units.GeV, cz_range[ci])

            inistate[ci][ei][1][0] = flux.getFlux(nuflux.NuEBar, e_range[ei]/units.GeV, cz_range[ci])
            inistate[ci][ei][1][1] = flux.getFlux(nuflux.NuMuBar, e_range[ei]/units.GeV, cz_range[ci])
            inistate[ci][ei][1][2] = flux.getFlux(nuflux.NuTauBar, e_range[ei]/units.GeV, cz_range[ci])

            if nuSQ.GetNumNeu() == 4:
                inistate[ci][ei][0][3] = 0.
                inistate[ci][ei][1][3] = 0.

    nuSQ.Set_initial_state(inistate, nsq.Basis.flavor)

    nuSQ.Set_ProgressBar(True)
    nuSQ.Set_IncludeOscillations(True)
    nuSQ.Set_GlashowResonance(True);
    nuSQ.Set_TauRegeneration(True);

    return nuSQ

flux = nuflux.makeFlux('honda2006')
flux.knee_reweighting_model = 'gaisserH3a_elbert'

dm2 = 1.
th14 = 0.
th24 = np.arcsin(np.sqrt(0.1))/2.
th34 = 0.0
cp = 0.

nus_atm = get_nusquids_obj(4, dm2, th14, th24, th34, cp, flux)
nus_atm.EvolveState()
nus_atm.WriteStateHDF5('./fluxes/conv_sin22th_0p1.h5', True)

nus_atm = get_nusquids_obj(3, 0, 0, 0, 0, 0, flux)
nus_atm.EvolveState()
nus_atm.WriteStateHDF5('./fluxes/conv.h5', True)
