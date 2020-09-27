import os
import re
import numpy as np
import scipy
import scipy.stats
import bisect

def get_bins(emin=100, emax=1e7, ewidth=0.111, eedge=100, zmin=-1, zmax=0, zwidth=0.1, zedge=0):
    """
    Get the default analysis bins.
    """
    target_emin = emin
    target_emax = emax

    n_edges = 1
    energy_edge = eedge
    emin = energy_edge
    while emin > target_emin:
        n_edges += 1
        emin /= 10.**ewidth
    emax = energy_edge
    while emax < target_emax:
        n_edges += 1
        emax *= 10.**ewidth

    energy_bins = np.logspace(np.log10(emin), np.log10(emax), n_edges)

    target_zmin = zmin
    target_zmax = zmax

    n_edges = 1
    zenith_edge = zedge
    zmin = zenith_edge
    while zmin > target_zmin:
        n_edges += 1
        zmin -= zwidth
    zmax = zenith_edge
    while zmax < target_zmax:
        n_edges += 1
        zmax += zwidth

    zmax = min(zmax, 1)
    zmin = max(zmin, -1)

    zenith_bins = np.arccos(np.linspace(zmin, zmax, n_edges))[::-1]

    #zenith_bins = np.arccos(np.linspace(-1, 1, nzenith+1))[::-1]

    return energy_bins, zenith_bins


def get_particle_masks(particleType):
    """
    Get a dictionary containing masks by particle type.
    """
    particle_dict = {
        'eminus': 11,
        'eplus': -11,
        'muminus': 13,
        'muplus': -13,
        'tauminus': 15,
        'tauplus': -15,
        'nue': 12,
        'nuebar': -12,
        'numu': 14,
        'numubar': -14,
        'nutau': 16,
        'nutaubar': -16,
    }
    abs_particle_dict = {
        'e': 11,
        'mu': 13,
        'tau': 15,
        '2nue': 12,
        '2numu': 14,
        '2nutau': 16,
    }
    other_particle_dict = {
            'nu': lambda x: (lambda xx: reduce(np.logical_or, [(xx == 12), (xx == 14), (xx == 16)], np.zeros(xx.shape)))(abs(np.array(x))),
            'all': lambda x: np.ones(np.array(x).shape).astype(bool),
    }
    masks = {}
    for name, id in particle_dict.items():
        mask = particleType == id
        if np.any(mask):
            masks[name] = mask
    for name, id in abs_particle_dict.items():
        mask = abs(particleType) == id
        if np.any(mask):
            masks[name] = mask
    for name, id in other_particle_dict.items():
        mask = id(particleType)
        if np.any(mask):
            masks[name] = mask
    return masks

def get_bin_masks(energy, zenith, energy_bins, zenith_bins):
    """
    Get masks for all analysis bins
    Returns 4 sets of masks: cascade bins, track bins, double cascade bins, and all bins
    """

    def make_bin_masks(energies, zeniths, energy_bins, zenith_bins):

        assert(len(energies) == len(zeniths))

        n_energy_bins = len(energy_bins) - 1
        n_zenith_bins = len(zenith_bins) - 1

        energy_mapping = np.digitize(energies, bins=energy_bins) - 1
        zenith_mapping = np.digitize(zeniths, bins=zenith_bins) - 1
        bin_masks = []
        for j in range(n_zenith_bins):
            for k in range(n_energy_bins):
                mask = zenith_mapping == j
                mask = np.logical_and(mask, energy_mapping == k)
                bin_masks.append(mask)
        return bin_masks

    masks = make_bin_masks(energy, zenith, energy_bins, zenith_bins)

    return masks

def mode(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """

    if axis is not None:
        fnc = lambda x: mode(x, dtype=dtype)
        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:
                wMin = data[-1] - data[0]
                N = data.size / 2 + data.size % 2
                j = None
                for i in xrange(0, N):
                    w = data[i+N-1] - data[i]
                    if w < wMin:
                        wMin = w
                        j = i
                if j is None:
                    return data.mean()
                return _hsm(data[j:j+N])

        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)

        # The data need to be sorted for this to work
        data = np.sort(data)

        # Find the mode
        dataMode = _hsm(data)

    return dataMode

def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of
    a given width

    Assumes that x is sorted numpy array.
    """
    n = len(x)
    cred_mass = 1.0 - alpha

    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return hdi_min, hdi_max

def hpd(x, alpha=0.05, transform=lambda x: x):
    """Calculate highest posterior density (HPD) of array for given alpha. The HPD is the
    minimum width Bayesian credible interval (BCI).

    :Arguments:
      x : Numpy array
          An array containing MCMC samples
      alpha : float
          Desired probability of type I error (defaults to 0.05)
      transform : callable
          Function to transform data (defaults to identity)

    """
    # Make a copy of trace
    x = transform(x.copy())

    # For multivariate node
    if x.ndim > 1:

        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:] + [0])
        dims = np.shape(tx)

        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1] + (2,))

        for index in make_indices(dims[:-1]):

            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])

            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)

        # Transpose back before returning
        return np.array(intervals)

    else:
        # Sort univariate node
        sx = np.sort(x)

        return np.array(calc_min_interval(sx, alpha))

def interval(arr, proportion=scipy.special.erf(1.0/np.sqrt(2.0)), min=None, max=None):
    """
    Compute distribution mode and the HPD that contains `proportion` of the mass.
    """
    x = hpd(arr, alpha=(1.0-proportion))
    if x[0] <= np.amin(arr):
        if min is not None:
            x[0] = min
    if x[-1] <= np.amax(arr):
        if max is not None:
            x[1] = max
    return x[0], mode(arr), x[1]

def weighted_median(quantity, weights, alpha=0.5):
    total = np.sum(weights)
    order = np.argsort(quantity)
    sorted_q = quantity[order]
    sorted_w = weights[order]
    cumulative_w = np.cumsum(sorted_w) / total

    i = bisect.bisect_left(cumulative_w, alpha) - 1
    if i < 0 or i >= len(quantity):
        return None
    return (sorted_q[i]*sorted_w[i]*(1.0 - alpha) + sorted_q[i+1]*sorted_w[i+1]*(alpha))/(sorted_w[i]*(1.0 - alpha) + sorted_w[i+1]*(alpha))

def bayes_format(bayes_factor):
    x = bayes_factor
    if x < 0.01 or x >= 100:
        exponent = np.floor(np.log10(x))
        mantissa = x * 10.0**(-exponent)
        mantissa = np.round(mantissa, 2)
        s = r'%.2f\times10^{%d}' % (mantissa, exponent)
    elif x < 1 and x >= 0.1:
        s = '%.3f' % x
    elif x < 0.1 and x >= 0.01:
        s = '%.4f' % x
    elif x < 100 and x >= 10:
        s = '%.0f' % x
    elif x < 10 and x >= 1:
        s = '%.2f' % x
    return s

def to_precision(x, p, scientific_notation=False):
    """
    returns a string representation of x formatted with a precision of p

    Based on the webkit javascript implementation taken from here:
    https://code.google.com/p/webkit-mirror/source/browse/JavaScriptCore/kjs/number_object.cpp
    """

    x = float(x)

    if x == 0.:
        return "0." + "0"*(p-1)

    out = []

    if x < 0:
        out.append("-")
        x = -x

    e = int(np.log10(x))
    tens = np.power(10., e - p + 1)
    n = np.floor(x/tens)

    if n < np.power(10., p - 1):
        e = e -1
        tens = np.power(10., e - p+1)
        n = np.floor(x / tens)

    if abs((n + 1.) * tens - x) <= abs(n * tens -x):
        n = n + 1

    if n >= np.power(10.,p):
        n = n / 10.
        e = e + 1

    m = "%.*g" % (p, n)

    if scientific_notation and (e < -2 or e >= p):
        out.append(m[0])
        if p > 1:
            out.append(".")
            out.extend(m[1:p])
        out.append('e')
        if e > 0:
            out.append("+")
        out.append(str(e))
    elif e == (p -1):
        out.append(m)
    elif e+1>len(m) and '.' not in m:
        out.append(m)
        out.extend(['0']*(e+1-len(m)))
    elif e >= 0:
        out.append(m[:e+1])
        if e+1 < len(m):
            out.append(".")
            out.extend(m[e+1:])
    else:
        out.append("0.")
        out.extend(["0"]*-(e+1))
        out.append(m)

    return "".join(out)

def error_format(nums, errors, error_precision=2, scientific_notation=False, default_num_precision=3):
    error_place = (lambda x: (np.min(x) - (error_precision - 1)) if len(x) > 0 else None)([int(np.floor(np.log10(abs(error)))) for error in errors if abs(error) > 0.0])
    if error_place is None:
        num_precision = [default_num_precision for num in nums]
    else:
        num_precision = [int(np.floor(np.log10(abs(num)))) - error_place + 1 if abs(num) > 0 else default_num_precision for num in nums]
    num_s = [to_precision(num, nprecision, scientific_notation=scientific_notation) for num, nprecision in zip(nums, num_precision)]
    error_s = [to_precision(error, error_precision, scientific_notation=scientific_notation) for error in errors]
    return num_s, error_s

def pretty_interval(flux_values):
    indices = [int(np.floor((0 if np.amax(f) == 0 else np.log10(np.amax(f))) if f[1] == 0 else np.log10(f[1]))) for f in flux_values]
    nums_to_format = [f/10.**i for f,i in zip(flux_values, indices)]
    interval_info = [(error_format([x[1]], [x[1]-x[0], x[2]-x[1]], scientific_notation=False), i) for x,i in zip(nums_to_format, indices)]
    out = []
    for info in interval_info:
        (bf,), (lower, upper) = info[0]
        index = str(info[1])
        out.append(r'${' + bf + r'}^{' + upper + r'}_{-' + lower + r'}\times10^{' + index + r'}$')
    return out

def chunk(items, chunks, chunk_number):
    if chunks > 0:
        a = int(np.floor(float(len(items)) / float(chunks)))
        b = int(np.ceil(float(len(items)) / float(chunks)))
        x = len(items) - a*chunks
        if chunk_number < x:
            n = b
            n0 = n*chunk_number
        else:
            n = a
            n0 = b*x + a*(chunk_number - x)
        n1 = min(n0 + n, len(items))
        if n0 >= len(items):
            return scan_results
        items = items[n0:n1]
    return items
