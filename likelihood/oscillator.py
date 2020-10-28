import os
import os.path

class memodict_(collections.OrderedDict):
    def __init__(self, f, maxsize=1):
        collections.OrderedDict.__init__(self)
        self.f = f
        self.maxsize = maxsize
    def __missing__(self, key):
        if len(self) == self.maxsize:
            self.popitem(last=False)
        ret = self[key] = self.f(*key)
        return ret
    def __call__(self, *args):
        return self.__getitem__(args)

class _cache_(collections.OrderedDict):
    def __init__(self, maxsize=1):
        collections.OrderedDict.__init__(self)
        self.maxsize = maxsize
    def __missing__(self, key):
        if len(self) == self.maxsize:
            self.popitem(last=False)
        ret = self[key] = self.f(*key)
        return ret
    def __call__(self, *args):
        return self.__getitem__(args)

class oscillator:
    def __init__(
            self,
            name,
            flux,
            ebins,
            zbins,
            scenario,
            store_dir,
            cache_size = 10,
            ):
        self.name = name
        self.flux = flux
        self.initial_flux = None
        self.ebins = ebins
        self.zbins = zbins
        self.scenario = scenario
        self.store_dir = store_dir
        self.cache_size = cache_size
        self._build_object = lambda params: self.build_object(params)
        self.object_store = memodict_(self._build_object, maxsize=self.cache_size)
    def build_object(self, params):




def param_to_string(param, log=True, n=8):
    if log:
        if param == 0:
            s = 'z'
        else:
            if param < 0:
                s = 'p'
            else:
                s = 'm'
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
    if scenario == 'sterile':
        try:
            ss = fname.split('_')
            params = [4] + [string_to_param(s) for s in ss[-6:]]
            baseline = ss[0]
            return baseline, scenario, params
        except:
            return fname, scenario, (3, 0,0,0,0,0)

def load_baseline_flux()

def get_object(numneu, dm2, th14, th24, th34, cp, flux):
    args = (numneu, dm2, th14, th24, th34, cp, flux)
    fname = params_to_fname((flux, 'sterile', args))
    params = fname_to_params(fname)
    if os.path.exists(fname):
        obj = nsq.nuSQUIDSAtm(fname)
    else:
        baseline_fname = params[0] + '/' + params[1] + '/' + flux + '.h5'
        obj = nsq.nuSQUIDSAtm(baseline_fname)


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

def get_nusquids_object(parameters):
    if in_store(parameters):
        return from_store(parameters)
    if file_exists(parameters):
        obj = load_file(parameters)
        add_to_store(obj, parameters)
    else:
        obj = run_nusquids(parameters)
        add_to_store(obj, parameters)
        save_file(obj, parameters)
    return obj

