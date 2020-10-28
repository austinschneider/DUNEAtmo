import numpy as np
import scipy.stats
import scipy.interpolate

class LV_scenario:
    @staticmethod
    def construct_interp(x, y, z):
        xx = np.sort(np.unique(x))
        xr = np.arange(len(xx))
        yy = np.sort(np.unique(y))
        yr = np.arange(len(yy))
        zz = np.full((len(xx), len(yy)), np.nan)
        print("Setting grid")
        for zi,(xv, yv, zv) in enumerate(zip(x,y,z)):
            xi = xr[xx == xv].item()
            yi = yr[yy == yv].item()
            zz[xi, yi] = zv

        res = scipy.interpolate.interp2d(xx, yy, zz.T, fill_value=np.nan)
        return res

    @staticmethod
    def interps_from_data(data, n_pre_columns=1):
        data = np.asarray(data)
        assert(len(data.shape) == 2)
        n = data.shape[1]
        assert(n > (n_pre_columns+2))
        assert((n-(n_pre_columns+2)) % 2 == 0)
        x = data[:,(0+n_pre_columns)]
        y = data[:,(1+n_pre_columns)]
        interps = [LV_scenario.construct_interp(x, y, data[:,i]) for i in range((n_pre_columns+2),n)]
        return interps

    def __init__(self, nu_fname, nubar_fname, n_pre_columns=1):
        self.nu_data = np.loadtxt(fname=nu_fname)
        self.nubar_data = np.loadtxt(fname=nubar_fname)
        self.nu_interps = LV_scenario.interps_from_data(self.nu_data, n_pre_columns)
        self.nubar_interps = LV_scenario.interps_from_data(self.nubar_data, n_pre_columns)
        self.n_neutrinos = len(self.nu_interps)/2
        self.n_antineutrinos = len(self.nubar_interps)/2

    def get_interps(self, flavortype, mattertype):
        assert(abs(mattertype) < 2)
        n = [self.n_neutrinos, self.n_antineutrinos][mattertype]
        interp_set = [self.nu_interps, self.nubar_interps][mattertype]
        assert(flavortype < n)
        null_i = 2 * int(flavortype) + 1
        bsm_i = 2 * int(flavortype)
        null_interp = interp_set[null_i]
        bsm_interp = interp_set[bsm_i]

        return null_interp, bsm_interp

    def correction(self, x, y, flavortype, mattertype):
        null_interp, bsm_interp = self.get_interps(flavortype, mattertype)
        null = null_interp(x, y)
        bsm = bsm_interp(x, y)
        res = bsm/null
        res[np.isnan(res)] = 0.0
        return res
