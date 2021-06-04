import argparse                                                                                                            
parser = argparse.ArgumentParser(description="LV scan")
#parser.add_argument('-i', '--initial-flux',
#        type=str,
#        dest='initial_flux',
#        required=True,
#        )
#parser.add_argument('-f', '--final-flux',
#        type=str,
#        dest='final_flux',
#        required=True,
#        )
parser.add_argument('-o', '--output',
        type=str,
        dest='output',
        required=True,
        )
parser.add_argument('--granularity',
        type=int,
        dest='granularity',
        default=1,
        )
for l in ["e", "mu", "tau"]:
    nu = "nu" + l
    nubar = nu + "bar"
    for n in [nu, nubar]:
        parser.add_argument('--'+n,
                dest=n,
                action='store_true',
                default=False,
                )
args = parser.parse_args()

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import nuSQuIDS as nsq
import nuflux
import time

flux = nuflux.makeFlux("H3a_SIBYLL23C")

units = nsq.Const()
emin = 2
emax = 5
ebins = np.logspace(emin, emax, 20*(emax-emin) + 1) * units.GeV
czbins = np.linspace(-1, 0, 50 + 1)

init_state = np.zeros((len(czbins), len(ebins), 2, 3))
for ci in range(len(czbins)):
    cz = czbins[ci]
    for ei in range(len(ebins)):
        e = ebins[ei]
        init_state[ci][ei][0][0] = flux.getFlux(nuflux.NuE, e/units.GeV, cz)
        init_state[ci][ei][0][1] = flux.getFlux(nuflux.NuMu, e/units.GeV, cz)
        init_state[ci][ei][0][2] = flux.getFlux(nuflux.NuTau, e/units.GeV, cz)

        init_state[ci][ei][1][0] = flux.getFlux(nuflux.NuEBar, e/units.GeV, cz)
        init_state[ci][ei][1][1] = flux.getFlux(nuflux.NuMuBar, e/units.GeV, cz)
        init_state[ci][ei][1][2] = flux.getFlux(nuflux.NuTauBar, e/units.GeV, cz)

        #init_state[ci][ei][0][3] = 0.
        #init_state[ci][ei][1][3] = 0.

def build_grid(rho, f1, f2):
    one = np.ones((len(rho), len(f1), len(f2)))
    a = one*rho[:,None,None]*f1[None,:,None]
    b = one*rho[:,None,None]*f2[None,None,:]*np.sqrt(1-f1**2)[None,:,None]
    c = one*rho[:,None,None]*(np.sqrt(1-f2**2)[None,None,:])*np.sqrt(1-f1**2)[None,:,None]
    points = np.array([a.flatten(), b.flatten(), c.flatten()]).T
    return np.unique(points, axis=0)

def setup_nsq(nuSQ):
    nuSQ.Set_ProgressBar(True)
    nuSQ.Set_IncludeOscillations(True)
    nuSQ.Set_GlashowResonance(True);
    nuSQ.Set_TauRegeneration(True);

    nuSQ.Set_MixingAngle(0,1,0.563942)
    nuSQ.Set_MixingAngle(0,2,0.154085)
    nuSQ.Set_MixingAngle(1,2,0.785398)

    nuSQ.Set_SquareMassDifference(1,7.65e-05)
    nuSQ.Set_SquareMassDifference(2,0.00247)

    nuSQ.Set_CPPhase(0,2,0.0)
    nuSQ.Set_rel_error(1.0e-08)
    nuSQ.Set_abs_error(1.0e-08)

    dm2 = 0.5
    th14 = 0.0
    th24 = 40./180.*np.pi
    th34 = 0.0

    operator_dimension = 4
    lv_power = operator_dimension - 3                                              
    rho_3_grid = np.array([10**-23])
    f1_3_grid = np.array([0.0])
    f2_3_grid = np.array([0.2])
    grid = build_grid(rho_3_grid, f1_3_grid, f2_3_grid)
    (diag3,), (re3,), (im3,) = grid.T
    lv_emu_re = 0
    lv_emu_im = 0
    lv_mutau_re = re3
    lv_mutau_im = im3
    lv_etau_re = 0
    lv_etau_im = 0
    lv_ee = 0
    lv_mumu = diag3
    units = nsq.Const()
    u = (units.GeV/units.eV)**(-lv_power + 1)
    lv_emu_re *= u
    lv_emu_im *= u
    lv_mutau_re *= u 
    lv_mutau_im *= u
    lv_etau_re *= u 
    lv_etau_im *= u
    lv_ee *= u
    lv_mumu *= u 

    nuSQ.Set_LV_EnergyPower(lv_power)

    nuSQ.Set_LV_OpMatrix(lv_emu_re, lv_emu_im, lv_mutau_re, lv_mutau_im, lv_etau_re, lv_etau_im, lv_ee, lv_mumu)
    nuSQ.Set_initial_state(init_state, nsq.Basis.flavor)

standard_nusq = nsq.nuSQUIDSLVAtm(czbins, ebins, 3, nsq.NeutrinoType.both, True)
setup_nsq(standard_nusq)
tic = time.perf_counter()
standard_nusq.EvolveState()
toc = time.perf_counter()
normal_time = toc-tic
print("Normal time:", normal_time)
exit()


#standard_nusq = nsq.nuSQUIDSLVAtm("./standard.h5")

nuSQ = nsq.nuSQUIDSLVAtm(czbins, ebins, 3, nsq.NeutrinoType.both, True)
setup_nsq(nuSQ)
f = 8
fp = f*2
tic = time.perf_counter()
nuSQ.Set_AutoEvolLowPass(np.pi/f, np.pi/fp, False)
nuSQ.EvolveState()
toc = time.perf_counter()
lowpass_time = toc-tic
print("Lowpass time:", lowpass_time)

fast = nsq.nuSQUIDSLVAtm(czbins, ebins, 3, nsq.NeutrinoType.both, True)
setup_nsq(fast)
tic = time.perf_counter()
fast.Set_AutoEvolLowPass(np.pi/f, np.pi/fp, True)
fast.EvolveState()
toc = time.perf_counter()
fast_lowpass_time = toc-tic
print("Fast lowpass time:", fast_lowpass_time)

print("Normal time:", normal_time)
print("Lowpass time:", lowpass_time)
print("Fast lowpass time:", fast_lowpass_time)

before = nsq.nuSQUIDSLVAtm(czbins, ebins, 3, nsq.NeutrinoType.both, True)
setup_nsq(before)

initial = standard_nusq
final = nuSQ

cos_theta_grid = np.unique(list(initial.GetCosthRange()) + list(final.GetCosthRange()))
np.sort(cos_theta_grid)
cos_theta_grid = np.unique(np.concatenate([np.linspace(cos_theta_grid[i], cos_theta_grid[i+1], args.granularity+1) for i in range(len(cos_theta_grid)-1)]))
np.sort(cos_theta_grid)

energy_grid = np.unique(list(initial.GetERange()) + list(initial.GetERange()))
np.sort(energy_grid)
energy_grid = np.unique(np.concatenate([np.logspace(np.log10(energy_grid[i]), np.log10(energy_grid[i+1]), args.granularity+1) for i in range(len(energy_grid)-1)]))
np.sort(energy_grid)
energy_grid = energy_grid / units.GeV

initial_flux = dict()
final_flux = dict()
before_flux = dict()
fast_flux = dict()

for flavor,l in [(0,"e"), (1,"mu"), (2,"tau")]:
    nu = "nu" + l
    nubar = nu + "bar"
    for rho,n in [(0,nu),(1,nubar)]:
        if not getattr(args, n):
            continue
        initial_flux[n] = np.zeros((len(energy_grid)-1, len(cos_theta_grid)-1))
        final_flux[n] = np.zeros((len(energy_grid)-1, len(cos_theta_grid)-1))
        fast_flux[n] = np.zeros((len(energy_grid)-1, len(cos_theta_grid)-1))
        before_flux[n] = np.zeros((len(energy_grid)-1, len(cos_theta_grid)-1))
        for j in range(len(cos_theta_grid)-1):
            for k in range(len(energy_grid)-1):
                cz = (cos_theta_grid[j] + cos_theta_grid[j+1])/2.0
                e = 10**((np.log10(energy_grid[k]) + np.log10(energy_grid[k+1]))/2.0)
                initial_flux[n][k,j] = initial.EvalFlavor(flavor, cz, e*units.GeV, rho)
                final_flux[n][k,j] = final.EvalFlavor(flavor, cz, e*units.GeV, rho)
                fast_flux[n][k,j] = fast.EvalFlavor(flavor, cz, e*units.GeV, rho)
                before_flux[n][k,j] = before.EvalFlavor(flavor, cz, e*units.GeV, rho)

        cm = plt.get_cmap('plasma')
        cm.set_under('black')
        vmin = min(np.amin(initial_flux[n][initial_flux[n] > 0]), np.amin(final_flux[n][final_flux[n] > 0]))
        vmax = max(np.amax(initial_flux[n][initial_flux[n] > 0]), np.amax(final_flux[n][final_flux[n] > 0]))
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(figsize=(7,5))
        X = np.array([cos_theta_grid]*(len(energy_grid)))
        Y = np.array([energy_grid]*(len(cos_theta_grid))).T
        mesh = ax.pcolormesh(X,Y,initial_flux[n], cmap=cm, norm=norm)
        #mesh = ax.pcolormesh(X,Y,final_flux[n], cmap=cm, norm=norm)
        #mesh = ax.pcolormesh(X,Y,initial_flux[n], cmap=cm, norm=norm)
        ax.set_yscale('log')
        #ax.set_ylim((1e2, 1e5))
        #ax.set_xlim((-1,1))
        ax.set_ylabel('Neutrino Energy [GeV]')
        ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.set_ylabel('Flux (nominal)' + n)
        cb.ax.minorticks_on()
        plt.tight_layout()
        fig.savefig(args.output + '_nominal_flux_' + n +'.png', dpi=200)
        fig.clf()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7,5))
        X = np.array([cos_theta_grid]*(len(energy_grid)))
        Y = np.array([energy_grid]*(len(cos_theta_grid))).T
        mesh = ax.pcolormesh(X,Y,final_flux[n], cmap=cm, norm=norm)
        #mesh = ax.pcolormesh(X,Y,final_flux[n], cmap=cm, norm=norm)
        #mesh = ax.pcolormesh(X,Y,initial_flux[n], cmap=cm, norm=norm)
        ax.set_yscale('log')
        #ax.set_ylim((1e2, 1e5))
        #ax.set_xlim((-1,1))
        ax.set_ylabel('Neutrino Energy [GeV]')
        ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.set_ylabel('Flux (lowpass)' + n)
        cb.ax.minorticks_on()
        plt.tight_layout()
        fig.savefig(args.output + '_lowpass_flux_' + n +'.png', dpi=200)
        fig.clf()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7,5))
        X = np.array([cos_theta_grid]*(len(energy_grid)))
        Y = np.array([energy_grid]*(len(cos_theta_grid))).T
        mesh = ax.pcolormesh(X,Y,fast_flux[n], cmap=cm, norm=norm)
        #mesh = ax.pcolormesh(X,Y,final_flux[n], cmap=cm, norm=norm)
        #mesh = ax.pcolormesh(X,Y,initial_flux[n], cmap=cm, norm=norm)
        ax.set_yscale('log')
        #ax.set_ylim((1e2, 1e5))
        #ax.set_xlim((-1,1))
        ax.set_ylabel('Neutrino Energy [GeV]')
        ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.set_ylabel('Flux (lowpass approx)' + n)
        cb.ax.minorticks_on()
        plt.tight_layout()
        fig.savefig(args.output + '_lowpass_approx_flux_' + n +'.png', dpi=200)
        fig.clf()
        plt.close(fig)


        cm = plt.get_cmap('plasma')
        cm.set_under('black')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

        fig, ax = plt.subplots(figsize=(7,5))
        X = np.array([cos_theta_grid]*(len(energy_grid)))
        Y = np.array([energy_grid]*(len(cos_theta_grid))).T
        mesh = ax.pcolormesh(X,Y,initial_flux[n]/before_flux[n], cmap=cm, norm=norm)
        #mesh = ax.pcolormesh(X,Y,final_flux[n], cmap=cm, norm=norm)
        #mesh = ax.pcolormesh(X,Y,initial_flux[n], cmap=cm, norm=norm)
        ax.set_yscale('log')
        #ax.set_ylim((1e2, 1e5))
        #ax.set_xlim((-1,1))
        ax.set_ylabel('Neutrino Energy [GeV]')
        ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.set_ylabel('Transmission probability (nominal)' + n)
        cb.ax.minorticks_on()
        plt.tight_layout()
        fig.savefig(args.output + '_nominal_P_' + n +'.png', dpi=200)
        fig.clf()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7,5))
        X = np.array([cos_theta_grid]*(len(energy_grid)))
        Y = np.array([energy_grid]*(len(cos_theta_grid))).T
        mesh = ax.pcolormesh(X,Y,final_flux[n]/before_flux[n], cmap=cm, norm=norm)
        #mesh = ax.pcolormesh(X,Y,final_flux[n], cmap=cm, norm=norm)
        #mesh = ax.pcolormesh(X,Y,initial_flux[n], cmap=cm, norm=norm)
        ax.set_yscale('log')
        #ax.set_ylim((1e2, 1e5))
        #ax.set_xlim((-1,1))
        ax.set_ylabel('Neutrino Energy [GeV]')
        ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.set_ylabel('Transmission probability (lowpass)' + n)
        cb.ax.minorticks_on()
        plt.tight_layout()
        fig.savefig(args.output + '_lowpass_P_' + n +'.png', dpi=200)
        fig.clf()
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7,5))
        X = np.array([cos_theta_grid]*(len(energy_grid)))
        Y = np.array([energy_grid]*(len(cos_theta_grid))).T
        mesh = ax.pcolormesh(X,Y,fast_flux[n]/before_flux[n], cmap=cm, norm=norm)
        #mesh = ax.pcolormesh(X,Y,final_flux[n], cmap=cm, norm=norm)
        #mesh = ax.pcolormesh(X,Y,initial_flux[n], cmap=cm, norm=norm)
        ax.set_yscale('log')
        #ax.set_ylim((1e2, 1e5))
        #ax.set_xlim((-1,1))
        ax.set_ylabel('Neutrino Energy [GeV]')
        ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
        cb = fig.colorbar(mesh, ax=ax)
        cb.ax.set_ylabel('Transmission probability (lowpass)' + n)
        cb.ax.minorticks_on()
        plt.tight_layout()
        fig.savefig(args.output + '_lowpass_approx_P_' + n +'.png', dpi=200)
        fig.clf()
        plt.close(fig)

        try:
            cm = plt.get_cmap('plasma')
            cm.set_under('black')
            #norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            norm = matplotlib.colors.LogNorm()

            fig, ax = plt.subplots(figsize=(7,5))
            X = np.array([cos_theta_grid]*(len(energy_grid)))
            Y = np.array([energy_grid]*(len(cos_theta_grid))).T
            mesh = ax.pcolormesh(X,Y,np.abs(initial_flux[n]-final_flux[n]), cmap=cm, norm=norm)
            #mesh = ax.pcolormesh(X,Y,final_flux[n], cmap=cm, norm=norm)
            #mesh = ax.pcolormesh(X,Y,initial_flux[n], cmap=cm, norm=norm)
            ax.set_yscale('log')
            #ax.set_ylim((1e2, 1e5))
            #ax.set_xlim((-1,1))
            ax.set_ylabel('Neutrino Energy [GeV]')
            ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
            cb = fig.colorbar(mesh, ax=ax)
            cb.ax.set_ylabel('Absolute flux difference ' + n)
            cb.ax.minorticks_on()
            plt.tight_layout()
            fig.savefig(args.output + '_fluxdiff_' + n +'.png', dpi=200)
            fig.clf()
            plt.close(fig)
        except:
            pass

        try:
            cm = plt.get_cmap('plasma')
            cm.set_under('black')
            #norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            norm = matplotlib.colors.LogNorm()

            fig, ax = plt.subplots(figsize=(7,5))
            X = np.array([cos_theta_grid]*(len(energy_grid)))
            Y = np.array([energy_grid]*(len(cos_theta_grid))).T
            mesh = ax.pcolormesh(X,Y,np.abs(initial_flux[n]-final_flux[n])/final_flux[n], cmap=cm, norm=norm)
            #mesh = ax.pcolormesh(X,Y,final_flux[n], cmap=cm, norm=norm)
            #mesh = ax.pcolormesh(X,Y,initial_flux[n], cmap=cm, norm=norm)
            ax.set_yscale('log')
            #ax.set_ylim((1e2, 1e5))
            #ax.set_xlim((-1,1))
            ax.set_ylabel('Neutrino Energy [GeV]')
            ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
            cb = fig.colorbar(mesh, ax=ax)
            cb.ax.set_ylabel('Relative flux difference ' + n)
            cb.ax.minorticks_on()
            plt.tight_layout()
            fig.savefig(args.output + '_fluxreldiff_' + n +'.png', dpi=200)
            fig.clf()
            plt.close(fig)
        except:
            pass


        try:
            cm = plt.get_cmap('plasma')
            cm.set_under('black')
            #norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            norm = matplotlib.colors.LogNorm()

            fig, ax = plt.subplots(figsize=(7,5))
            X = np.array([cos_theta_grid]*(len(energy_grid)))
            Y = np.array([energy_grid]*(len(cos_theta_grid))).T
            mesh = ax.pcolormesh(X,Y,np.abs(initial_flux[n]-fast_flux[n]), cmap=cm, norm=norm)
            #mesh = ax.pcolormesh(X,Y,fast_flux[n], cmap=cm, norm=norm)
            #mesh = ax.pcolormesh(X,Y,initial_flux[n], cmap=cm, norm=norm)
            ax.set_yscale('log')
            #ax.set_ylim((1e2, 1e5))
            #ax.set_xlim((-1,1))
            ax.set_ylabel('Neutrino Energy [GeV]')
            ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
            cb = fig.colorbar(mesh, ax=ax)
            cb.ax.set_ylabel('Absolute flux difference ' + n)
            cb.ax.minorticks_on()
            plt.tight_layout()
            fig.savefig(args.output + '_fluxdiff_approx_' + n +'.png', dpi=200)
            fig.clf()
            plt.close(fig)
        except:
            pass

        try:
            cm = plt.get_cmap('plasma')
            cm.set_under('black')
            #norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            norm = matplotlib.colors.LogNorm()

            fig, ax = plt.subplots(figsize=(7,5))
            X = np.array([cos_theta_grid]*(len(energy_grid)))
            Y = np.array([energy_grid]*(len(cos_theta_grid))).T
            mesh = ax.pcolormesh(X,Y,np.abs(initial_flux[n]-fast_flux[n])/fast_flux[n], cmap=cm, norm=norm)
            #mesh = ax.pcolormesh(X,Y,fast_flux[n], cmap=cm, norm=norm)
            #mesh = ax.pcolormesh(X,Y,initial_flux[n], cmap=cm, norm=norm)
            ax.set_yscale('log')
            #ax.set_ylim((1e2, 1e5))
            #ax.set_xlim((-1,1))
            ax.set_ylabel('Neutrino Energy [GeV]')
            ax.set_xlabel(r'Neutrino $\cos\left(\theta_z\right)$')
            cb = fig.colorbar(mesh, ax=ax)
            cb.ax.set_ylabel('Relative flux difference ' + n)
            cb.ax.minorticks_on()
            plt.tight_layout()
            fig.savefig(args.output + '_fluxreldiff_approx_' + n +'.png', dpi=200)
            fig.clf()
            plt.close(fig)
        except:
            pass
print("Normal time:", normal_time)
print("Lowpass time:", lowpass_time)
print("Fast lowpass time:", fast_lowpass_time)
