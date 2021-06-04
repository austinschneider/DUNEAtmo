import numpy as np
import json

#np.seterr(all="raise")

def cmf(pmf,axis=None):
    if axis is None:
        axis = -1
    shape = np.shape(pmf)
    z_shape = np.copy(shape)
    z_shape[axis] = 1
    z = np.zeros(z_shape)
    return np.concatenate([z, np.cumsum(pmf, axis=axis)], axis=axis)

def pmf(cmf, axis=-1):
    return np.diff(cmf, axis=axis)

def interp(x, x0, x1, y0, y1):
    return y0+(x-x0)*(y1-y0)/(x1-x0)

def cmf_interp_helper(cmf_0, edges_0, y_val, i_0):
    y_val = min(1, max(y_val, 0))
    x0s = []
    # Determine where we are with respect to cmf_0
    while i_0 < len(cmf_0)-1 and cmf_0[i_0] < y_val:
        i_0 += 1
    # cmf[i_0] is greater than or equal to y_val
    if cmf_0[i_0] == y_val:
        # Check for duplicate values
        i_0_upper = i_0
        while i_0_upper < len(cmf_0)-1 and cmf_0[i_0_upper] == y_val:
            i_0_upper += 1
        i_0_upper -= 1
        if i_0 == i_0_upper:
            # We don't have duplicate values
            x0s.append(edges_0[i_0])
        else:
            # We have duplicate values
            x0s.append(edges_0[i_0])
            x0s.append(edges_0[i_0_upper])
    else:
        #print(i_0, cmf_0[i_0], y_val)
        #print(i_0-1, cmf_0[i_0-1], y_val)
        assert(y_val <= cmf_0[i_0])
        assert(y_val > cmf_0[i_0-1])
        # Use cmf_0[i_0 - 1] for interpolation
        x0 = interp(y_val, cmf_0[i_0-1], cmf_0[i_0], edges_0[i_0-1], edges_0[i_0])
        x0s.append(x0)
    return x0s, i_0
"""
def cmf_interp_helper(cmf_0, edges_0, y_val, i_0):
    i_0 = np.searchsorted(cmf_0[i_0:], y_val, 'left') + i_0
    x0s = []
    i_0 = min(i_0, len(edges_0)-1)
    if cmf_0[i_0] == y_val:
        i_1 = np.searchsorted(cmf_0[i_0:], y_val, 'right') + i_0 - 1
        if i_0 == i_1:
            x0s.append(i_0)
        else:
            x0s.extend([i_0, i_1])
    else:
        x0s.append(interp(y_val, cmf_0[i_0-1], cmf_0[i_0], edges_0[i_0-1], edges_0[i_0]))
    return x0s, i_0
"""

def compute_interp_cmf(cmf_0, cmf_1, edges_0, edges_1, x):
    y_vals = np.unique(np.concatenate([cmf_0, cmf_1]))
    i_0 = 0
    i_1 = 0
    x_pairs = []
    return_y = []
    for y_val in y_vals:
        x0s, i_0 = cmf_interp_helper(cmf_0, edges_0, y_val, i_0)
        x1s, i_1 = cmf_interp_helper(cmf_1, edges_1, y_val, i_1)
        pairs = [(x0, x1) for x0 in x0s for x1 in x1s]
        x_pairs.extend(pairs)
        return_y.extend([y_val]*len(pairs))
    return_x = [x0*(1.0-x) + x1*x for x0, x1 in x_pairs]
    return [np.array(x) for x in zip(*sorted(zip(return_y, return_x), key=lambda x: x[1]))]

def pre_compute_cmf_interp_info(cmf_0, cmf_1, edges_0, edges_1):
    y_vals = np.unique(np.concatenate([cmf_0, cmf_1]))
    i_0 = 0
    i_1 = 0
    x_pairs = []
    return_y = []
    for y_val in y_vals:
        x0s, i_0 = cmf_interp_helper(cmf_0, edges_0, y_val, i_0)
        x1s, i_1 = cmf_interp_helper(cmf_1, edges_1, y_val, i_1)
        pairs = [(x0, x1) for x0 in x0s for x1 in x1s]
        x_pairs.extend(pairs)
        return_y.extend([y_val]*len(pairs))
    x_pairs = np.array(sorted(x_pairs, key=lambda x: x[0]))

    return x_pairs[:,0], x_pairs[:,1], np.array(return_y)

def compute_interp_cmf_from_cache(x, cache):
    x0, x1, return_y = cache
    return_x = x0*(1.0-x) + x1*x
    index = np.argsort(return_x)
    return return_y[index], return_x[index]

def rebin_cmf(cmf_0, edges_0, edges_1):
    r = np.empty(shape=len(edges_1))
    mask_less = edges_1 < np.amin(edges_0)
    mask_more = edges_1 >= np.amax(edges_0)
    mask = ~np.logical_or(mask_less, mask_more)
    r[mask_less] = 0.0
    r[mask_more] = 1.0

    x_pos = np.digitize(edges_1[mask], bins=edges_0) - 1
    y_vals = interp(edges_1[mask],
            edges_0[x_pos], edges_0[x_pos+1],
            cmf_0[x_pos], cmf_0[x_pos])
    r[mask] = y_vals
    return r

def ensure_cmf(cmf_0):
    n = len(np.shape(cmf_0))
    s0 = tuple([slice(None)]*(n-1) + [0])
    s1 = tuple([slice(None)]*(n-1) + [-1])
    cmf_0[s0] = 0.0
    cmf_0[s1] = 1.0

class MuExpect:
    def __init__(self,data,eprim,ebins,posbins, dimension_order=["muon energy", "loss energy", "loss position"]):
        desired_dimension_order = ["muon energy", "loss position", "loss energy"]
        order = [desired_dimension_order.index(s) for s in dimension_order]

        self.pdf_data = np.moveaxis(data, [0,1,2], order) # (muon energy, loss position, loss energy)
        self.total_data = np.cumsum(self.pdf_data, axis=1)
        self.expect_data = np.copy(self.total_data)
        self.expect_2d_data = np.copy(self.pdf_data)

        self.pdf_norm_data = np.sum(self.pdf_data, axis=2)
        self.total_norm_data = np.sum(self.total_data, axis=2)

        self.pdf_data = np.array(self.pdf_data / self.pdf_norm_data[:,:,None])
        self.total_data = np.array(self.total_data / self.total_norm_data[:,:,None])

        self.pdf_cmf = cmf(self.pdf_data, axis=2)
        self.total_cmf = cmf(self.total_data, axis=2)
        ensure_cmf(self.pdf_cmf)
        ensure_cmf(self.total_cmf)
        self.pdf_cmf[:,:,0] = 0.0
        self.pdf_cmf[:,:,-1] = 1.0
        self.total_cmf[:,:,0] = 0.0
        self.total_cmf[:,:,-1] = 1.0

        self.muon_energies = eprim
        self.energy_bins = ebins
        self.position_bins = posbins
        self.total_positions = posbins[1:]
        self.total_interp_cache = dict()

    def total_cache(self, mu_index, pos_index):
        key = (mu_index, pos_index)
        if key not in self.total_interp_cache:
            cmf_0 = self.total_cmf[mu_index, pos_index-1]
            cmf_1 = self.total_cmf[mu_index, pos_index]
            self.total_interp_cache[key] = pre_compute_cmf_interp_info(cmf_0, cmf_1, self.energy_bins, self.energy_bins)
        return self.total_interp_cache[key]

    def interp_pos_total(self, mu_energy_index, pos):
        data = self.total_cmf[mu_energy_index]
        index = np.searchsorted(self.total_positions, pos)

        left = self.total_positions[index-1]
        right = self.total_positions[index]

        if pos == right:
            return data[index], self.total_norm_data[mu_energy_index, index]

        cache = self.total_cache(mu_energy_index, index)

        x = (pos - left) / (right - left)

        new_cmf, new_edges = compute_interp_cmf_from_cache(x, cache)
        new_cmf = rebin_cmf(new_cmf, new_edges, self.energy_bins)

        norm_0 = self.total_norm_data[mu_energy_index, index]
        norm_1 = self.total_norm_data[mu_energy_index, index-1]

        new_norm = interp(pos, right, left, norm_0, norm_1)

        return new_cmf, new_norm

    def interp_energy_total(self, mu_energy, pos):
        index = np.searchsorted(self.muon_energies, mu_energy)
        #print(mu_energy)
        try:
            assert(mu_energy <= np.amax(self.muon_energies) and mu_energy >= np.amin(self.muon_energies))
        except:
            print(self.muon_energies)
            print(mu_energy)
            raise

        left = self.muon_energies[index-1]
        right = self.muon_energies[index]

        cmf_1, norm_1 = self.interp_pos_total(index, pos)
        ensure_cmf(cmf_1)

        if mu_energy == right:
            return cmf_1, norm_1

        x = (np.log10(mu_energy) - np.log10(left)) / (np.log10(right) - np.log10(left))
        assert(index > 0)

        cmf_0, norm_0 = self.interp_pos_total(index-1, pos)
        ensure_cmf(cmf_0)
        cache = pre_compute_cmf_interp_info(cmf_0, cmf_1, self.energy_bins, self.energy_bins)
        new_cmf, new_edges = compute_interp_cmf_from_cache(x, cache)
        new_cmf = rebin_cmf(new_cmf, new_edges, self.energy_bins)

        new_norm = interp(mu_energy, right, left, norm_0, norm_1)

        return new_cmf, new_norm

    def expect(self, mu_energy, pos):
        new_cmf, new_norm = self.interp_energy_total(mu_energy, pos)
        return pmf(new_cmf)*new_norm

    def simple_expect(self, mu_energy, pos):
        e_index = np.searchsorted(self.muon_energies, mu_energy)

        left = self.muon_energies[e_index-1]
        right = self.muon_energies[e_index]

        alpha = (np.log10(mu_energy) - np.log10(left)) / (np.log10(right) - np.log10(left))

        p_index = np.searchsorted(self.total_positions, pos)

        bottom = self.total_positions[p_index-1]
        top = self.total_positions[p_index]

        beta = (pos - bottom) / (top - bottom)

        a00 = (1-alpha)*(1-beta)
        a10 = (alpha)*(1-beta)
        a01 = (1-alpha)*(beta)
        a11 = alpha*beta

        b00 = self.expect_data[e_index-1,p_index-1]
        b10 = self.expect_data[e_index,p_index-1]
        b01 = self.expect_data[e_index-1,p_index]
        b11 = self.expect_data[e_index,p_index]
        return a00*b00 + a10*b10 + a01*b01 + a11*b11

    def simple_2d_expect(self, mu_energy, pos):
        e_index = np.searchsorted(self.muon_energies, mu_energy)

        left = self.muon_energies[e_index-1]
        right = self.muon_energies[e_index]

        alpha = (np.log10(mu_energy) - np.log10(left)) / (np.log10(right) - np.log10(left))

        p_index = np.searchsorted(self.total_positions, pos)

        bottom = self.total_positions[p_index-1]
        top = self.total_positions[p_index]

        beta = (pos - bottom) / (top - bottom)

        a0 = alpha
        a1 = 1-alpha

        b0 = beta
        b1 = 1-beta

        c0 = np.copy(self.expect_2d_data[e_index-1])
        c0[:,p_index:] = 0.0
        c0[:,p_index-1] = c0[:,p_index-1]*a1

        c1 = np.copy(self.expect_2d_data[e_index,p_index])
        c1[:,p_index:] = 0.0
        c1[:,p_index-1] = c1[:,p_index-1]*a1

        return c0*b0 + c1*b1
