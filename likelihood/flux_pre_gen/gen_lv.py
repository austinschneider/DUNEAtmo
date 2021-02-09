import sys
import os
import os.path
base_path = os.environ['GOLEMSPACE']
core_path = base_path + '/sources/DUNEAtmo/likelihood/core/'
default_flux_path = '/n/holyscratch01/arguelles_delgado_lab/Lab/DUNEAnalysis/store/fluxes/'
sys.path.insert(0, core_path)
import numpy as np
import nuflux
import nuSQuIDS as nsq
import oscillator

import argparse                                                                                                            
parser = argparse.ArgumentParser(description="Calculate fluxes")
parser.add_argument('--flux-path',
        type=str,
        dest='flux_path',
        required=False,
        default=default_flux_path,
        )
parser.add_argument('--chunks',
        type=int,
        dest="chunks",
        default=0,
        required=False,
        )
parser.add_argument('--chunk-number',
        type=int,
        dest="chunk_number",
        default=0,
        required=False,
        )
args = parser.parse_args()


flux_path = args.flux_path

units = nsq.Const()
ebins = np.logspace(1, 6, 100 + 1) * units.GeV
czbins = np.linspace(-1, 1, 100 + 1)

def build_grid(rho, f1, f2):
    one = np.ones((len(rho), len(f1), len(f2)))
    a = one*rho[:,None,None]*f1[None,:,None]
    b = one*rho[:,None,None]*f2[None,None,:]*np.sqrt(1-f1**2)[None,:,None]
    c = one*rho[:,None,None]*(np.sqrt(1-f2**2)[None,None,:])*np.sqrt(1-f1**2)[None,:,None]
    points = np.array([a.flatten(), b.flatten(), c.flatten()]).T
    return np.unique(points, axis=0)

rho_3_grid = np.concatenate([[0.0], np.logspace(-25, -19, 5*6+1)])
f1_3_grid = np.linspace(-1, 1, 50+1)
f2_3_grid = np.linspace(0, 1, 25+1)

rho_4_grid = np.concatenate([[0.0], np.logspace(-29, -23, 5*6+1)])
f1_4_grid = np.linspace(-1, 1, 50+1)
f2_4_grid = np.linspace(0, 1, 25+1)

flux = nuflux.makeFlux("H3a_SIBYLL23C")
osc = oscillator.oscillator(
    "H3a_SIBYLL23C", flux, ebins, czbins, "lv", flux_path, cache_size=10
)

# lv_emu_re, lv_emu_im, lv_mutau_re, lv_mutau_im, lv_etau_re, lv_etau_im, lv_ee, lv_mumu
#osc[(3, 0, 0, 0, 0, 0, 0, 0, 0)]

parameter_points = []

for diag3, re3, im3 in build_grid(rho_3_grid, f1_3_grid, f2_3_grid):
    params = (3, 0, 0, re3, im3, 0, 0, 0, diag3)
    parameter_points.append(params)

for diag4, re4, im4 in build_grid(rho_4_grid, f1_4_grid, f2_4_grid):
    params = (4, 0, 0, re4, im4, 0, 0, 0, diag4)
    parameter_points.append(params)

items = parameter_points

print(len(items))

if args.chunks > 0:
    a = int(np.floor(float(len(items)) / float(args.chunks)))
    b = int(np.ceil(float(len(items)) / float(args.chunks)))
    x = len(items) - a*args.chunks
    if args.chunk_number < x:
        n = b
        n0 = n*args.chunk_number
    else:
        n = a
        n0 = b*x + a*(args.chunk_number - x)
    n1 = min(n0 + n, len(items))
    if n0 >= len(items): 
        exit(0)
    items = items[n0:n1]

parameter_points = items

for params in parameter_points:
    osc[params]
