import os
import os.path
import collections
import numpy as np
import nuflux
import nuSQUIDSpy as nsq
import nuSQUIDSTools

class memodict_(collections.OrderedDict):
    def __init__(self, f, maxsize=1):
        collections.OrderedDict.__init__(self)
        self.f = f
        self.maxsize = maxsize
    def __missing__(self, key):
        if len(self) == self.maxsize:
            self.popitem(last=False)
        ret = self[key] = self.f(key)
        return ret
    def __call__(self, *args):
        return self.__getitem__(args)

def param_to_string(param, log=True, n=8):
    if log:
        if param == 0:
            s = 'z'
        else:
            if param < 0:
                s = 'm'
            else:
                s = 'p'
            s += ('%.' + ('%d' % n) + 'f') % np.log10(abs(param))
    else:
        s = ('%.' + ('%d' % n) + 'f') % np.log10(param)
    return s

def string_to_param(s):
    if s[0] in ['z', 'p', 'm']:
        if s[0] == 'z':
            return 0.0
        else:
            x = 10**float(s[1:])
            if s[0] == 'm':
                x *= -1.0
            return x
    else:
        return float(s)

def params_to_fname(parameters, store_dir='./'):
    baseline, scenario, params = parameters
    if scenario == 'sterile':
        suffix = '.h5'
        baseline = os.path.basename(baseline)
        if baseline.endswith(suffix):
            baseline = baseline[:-len(suffix)]
        num_nus, dm2, th14, th12, th34, cp = params
        if num_nus == 3:
            fname = baseline + suffix
        elif num_nus == 4:
            fname = baseline + '_' + '_'.join([param_to_string(p) for p in params[1:]]) + suffix
    elif scenario == 'lv':
        pass
    elif scenario == 'standard':
        fname = baseline + suffix
    return os.path.join(store_dir, scenario, fname)

def fname_to_params(fname):
    ss = fname.split('/')
    store_dir, scenario, fname = '/'.join(ss[:-2]), ss[-2], ss[-1]
    suffix = '.h5'
    fname = os.path.basename(fname)
    if fname.endswith(suffix):
        fname = fname[:-len(suffix)]
    if scenario == 'sterile' or scenario == 'baseline':
        try:
            ss = fname.split('_')
            params = [4] + [string_to_param(s) for s in ss[-6:]]
            baseline = ss[0]
            return baseline, scenario, params
        except:
            return fname, scenario, (3, 0,0,0,0,0)
    else:
        raise ValueError("Scenario fname to params conversion not implemented:", '"'+scenario+'"')

class oscillator:
    def __init__(
            self,
            name,
            flux,
            ebins,
            czbins,
            scenario,
            store_dir,
            cache_size = 10,
            ):
        self.name = name
        self.flux = flux
        self.init_state = None
        self.ebins = ebins
        self.czbins = czbins
        self.scenario = scenario
        self.store_dir = store_dir
        self.cache_size = cache_size
        self._build_object = lambda params: self.build_object(params)
        self.object_store = memodict_(self._build_object, maxsize=self.cache_size)

    def __getitem__(self, item):
        return self.object_store[item]

    def build_object(self, args):
        fname = params_to_fname((self.name, self.scenario, args), store_dir=self.store_dir)
        params = fname_to_params(fname)
        if os.path.exists(fname):
            if self.scenario == 'baseline' or self.scenario == 'sterile':
                obj = nsq.nuSQUIDSAtm(fname)
        else:
            self.prepare_init_state()
            obj = self.prepare_init_object(args)
            obj.EvolveState()
            obj.WriteStateHDF5(fname, True)
        return obj

    def prepare_init_state(self, force=False):
        if self.init_state is not None and not force:
            return
        if self.scenario == 'baseline' or self.scenario == 'sterile':
            init_state = np.zeros((len(self.czbins), len(self.ebins), 2, 4))
            units = nsq.Const()

            for ci in range(len(self.czbins)):
                cz = self.czbins[ci]
                for ei in range(len(self.ebins)):
                    e = self.ebins[ei]
                    init_state[ci][ei][0][0] = flux.getFlux(nuflux.NuE, e/units.GeV, cz)
                    init_state[ci][ei][0][1] = flux.getFlux(nuflux.NuMu, e/units.GeV, cz)
                    init_state[ci][ei][0][2] = flux.getFlux(nuflux.NuTau, e/units.GeV, cz)

                    init_state[ci][ei][1][0] = flux.getFlux(nuflux.NuEBar, e/units.GeV, cz)
                    init_state[ci][ei][1][1] = flux.getFlux(nuflux.NuMuBar, e/units.GeV, cz)
                    init_state[ci][ei][1][2] = flux.getFlux(nuflux.NuTauBar, e/units.GeV, cz)

                    init_state[ci][ei][0][3] = 0.
                    init_state[ci][ei][1][3] = 0.

            self.init_state = init_state
        else:
            raise ValueError("Scenario state initialization not implemented:", '"'+self.scenario+'"')

    def prepare_init_object(self, args):
        if self.scenario == 'baseline' or self.scenario == 'sterile':
            numneu, dm2, th14, th24, th34, cp = args

            assert numneu == 3 or numneu == 4, "numneu must be equal to 3 or 4"

            interactions = True

            nuSQ = nsq.nuSQUIDSAtm(self.czbins, self.ebins, numneu, nsq.NeutrinoType.both, interactions)

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


            if numneu > 3:
                nuSQ.Set_initial_state(self.init_state, nsq.Basis.flavor)
            else:
                nuSQ.Set_initial_state(self.init_state[:,:,:,:numneu], nsq.Basis.flavor)

            nuSQ.Set_ProgressBar(False)
            nuSQ.Set_IncludeOscillations(True)
            nuSQ.Set_GlashowResonance(True);
            nuSQ.Set_TauRegeneration(True);

            return nuSQ
        else:
            raise ValueError("Scenario state initialization not implemented:", '"'+self.scenario+'"')

