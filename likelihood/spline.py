import numpy as np
import photospline

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

def memodict(f, maxsize=1):
    """ Memoization decorator for a function taking a single argument """
    m = memodict_(f, maxsize)
    return m

class flux_repo_helper(type):
    fluxes = memodict_()
    def __getitem__(cls, item):
        if item not in flux_repo_helper.fluxes:
            try:
                spline = photospline.SplineTable(item)
                spline_repo_helper.splines[item] = spline
            except:
                print(item)
                raise ValueError("Spline cannot be opened")
        return spline_repo_helper.splines[item]

class spline_repo(object, metaclass=spline_repo_helper):
    pass

def eval_spline(spline, coords, grad=None):
    if np.shape(coords) == 2:
        coords = coords.T
    if grad is None:
        return spline.evaluate_simple(coords, 0)
    else:
        try:
            grad = int(grad)
            assert(grad < len(coords))
            grad = 0x1 << grad
            return spline.evaluate_simple(coords, grad)
        except:
            try:
                grad = list(grad)
                res = 0x0
                for i in range(np.shape(coords)[0]):
                    if i in grad:
                        res |= (0x1 << i)
                return spline.evaluate_simple(coords, res)
            except:
                try:
                    return spline.evaluate_simple(coords, grad)
                except:
                    raise ValueError("Could not interpret grad!")

