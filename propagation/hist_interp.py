import numpy as np
import numpy.linalg

import itertools

class simplex_interpolator:
    def __init__(self, parameter_points):
        self.points = parameter_points
        self.delaunay = scipy.spatial.Delaunay(self.points)

    def get_simplex(self, point):
        return self.delaunay.find_simplex(point)

    def get_interpolating_points(self, point):
        simplex = self.get_simplex(point)
        points = self.delaunay.simplices[simplex]
        return pionts

    def get_interpolant_factors(self, point, points):
        points =  np.array([self.points[pi] for pi in points])
        n = len(points)
        m = len(points[0])
        assert(np.all([len(p) == m for p in points]))

        is_simplex = m == n-1
        assert(is_simplex)

        M = np.array(points).T
        T = M[:,:-1] - M[:,-1:-2]
        Tinv = np.linalg.inv(T)
        r = np.array(point) - M[:,-1]
        lam = np.matmul(Tinv,r)
        res = np.empty((n,))
        res[:-1] = lam
        res[-1] = 1-np.sum(lam)
        return res

    def get_interpolation(self, point):
        points = self.get_interpolating_points(point)
        alpha = self.get_interpolant_factors(point, points)
        return point, points, alpha

class k_nearest_interpolator:
    def __init__(self, parameter_points):
        self.points = parameter_points
        self.dim = len(self.points[0])
        assert(np.all([len(p) == self.dim for p in self.points]))
        self.tree = scipy.spatial.cKDTree(self.points)

    def get_k_nearest(self, point):
        d, i = self.tree.query(point, self.dim+1)
        return i, d

    def get_interpolation(self, point):
        d, i = self.get_k_nearest(point)
        alpha = np.array(d)
        alpha /= np.sum(alpha)
        points = i
        return point, points, alpha

def sub_simplex(points):
    new_points = []
    faces = [points]
    while len(faces):
        face = faces.pop()
        new_points.append(np.mean(face, axis=0))
        if len(face) > 2:
            faces.extend(itertools.combinations(face,len(face-1)))
    return new_points

def sub_sample(points, depth=2):
    points = list(points)
    new_points = []
    simplices = [points]
    for d in range(depth):
        for simplex in simplices:
            new_points.extend(sub_simplex(simplex))
        if d < depth-1:
            d = scipy.spatial.Delaunay(points + new_points)
            simplices = [d.points[s] for s in d.simplices]
    return new_points

def sub_sample_bin(low, high, n=1):
    if n == 1:
        return [(low+high)/2.0]
    else:
        delta = (high-low)/(n+1)
        return np.linspace(low+delta, high-delta, n)

def sub_sample_bins(bins, n=1):
    try:
        n = list(n)
    except:
        n = [n]*len(bins)
    new_points = []
    for b,nn in zip(bins,n):
        new_points.append([])
        for i in range(len(b)-1):
            p = sub_sample_bin(b[i], b[i+1], n=nn)
            new_points[-1].extend(p)
    return itertools.product(*new_points)

def ndbin(x, bins):
    x = np.asarray(x)
    digi = [np.digitize(x[:,i], bins=b)-1 for i,b in enumerate(bins)]
    mapping = [d[None] == np.arange(len(b)-1)[:,None] for d,b in zip(digi,bins)]
    bin_shape = [len(b)-1 for b in bins]
    masks = [reduce(np.logical_and, [mapping[j][ii] for j,ii in enumerate(i)]) for i in itertools.product(*[np.arange(len(b)-1) for b in bins])]
    return masks

class hist_interpolator:
    def __init__(self, hists, mu, cov):
        self.hists = hists
        self.mu = np.asarray(mu)
        self.cov = np.asarray(cov)
        self.cov_det = np.array([np.linalg.det(c) for c in self.cov])
        self.std = np.sqrt(self.cov)
        self.inv_cov = np.array([np.linalg.inv(c) for c in cov])
        self.inv_std = np.sqrt(self.inv_cov)

    def get_mu(self, points, alpha):
        points = np.asarray(points)
        alpha = np.asarray(alpha)
        return np.sum(self.mu[points]*alpha[:,None], axis=0)

    def get_cov(self, points, alpha):
        points = np.asarray(points)
        alpha = np.asarray(alpha)
        return np.sum(self.cov[points]*alpha[:,None], axis=0)

    def get_x(self, x, point, mu, std):
        return np.matmul(std, np.matmul(self.std_inv[point], (x-self.mu[point]))) + mu

    def get_jac(self, point, cov_det):
        return np.sqrt(self.cov_det[point]/cov_det)

    def get_hist(self, points, alpha, bins, n=1):
        try:
            n = list(n)
        except:
            n = [n]*len(bins)
        mu = self.get_mu(points, alpha)
        cov = self.get_cov(points, alpha)
        std = np.sqrt(cov)
        jac_factors = []
        n_samples = np.product((len(b) for b in bins))
        bin_samples = list(sub_sample_bins(bins))
        bin_shape = tuple(n)
        for point in points:
            x = [self.get_x(xx, point, mu, std) for xx in bin_samples]
            hist = mask = ndbin(x, bins)
            [np.sum() for mask in masks]
            



class hist_interpolator:
    def __init__(self, hists, parameter_points, means=None, covariances=None, method=None):
        self.hists = hists
        self.points = parameter_points

        if method is None:
            method = "simplex"
        self.method = method

        if self.method == "k-nearest":
            self.interp = k_nearest_interpolator(self.points)
        elif self.method == "simplex":
            self.interp = simplex_interpolator(points)
        else:
            raise ValueError("Method: " + str(self.method) + " not recognized!")

    def get_interpolation(self, point):
        return self.interp.get_interpolation(point)

    def interpolate_moment(self, moments, interpolants):
        return np.sum(np.array(moments)*np.array(interpolants), axis=0)


