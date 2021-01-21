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

diag3_grid = np.concatenate([[0.0], np.logspace(-25, -22, 5*3+1), -np.logspace(-25, -22, 5*3+1)])
re3_grid = np.concatenate([[0.0], np.logspace(-25, -22, 5*3+1)])
im3_grid = np.concatenate([[0.0], np.logspace(-25, -22, 5*3+1)])

diag4_grid = np.concatenate([[0.0], np.logspace(-29, -26, 5*3+1), -np.logspace(-29, -26, 5*3+1)])
re4_grid = np.concatenate([[0.0], np.logspace(-29, -26, 5*3+1)])
im4_grid = np.concatenate([[0.0], np.logspace(-29, -26, 5*3+1)])

flux = nuflux.makeFlux("H3a_SIBYLL23C")
osc = oscillator.oscillator(
    "H3a_SIBYLL23C", flux, ebins, czbins, "lv", flux_path, cache_size=10
)

# lv_emu_re, lv_emu_im, lv_mutau_re, lv_mutau_im, lv_etau_re, lv_etau_im, lv_ee, lv_mumu
#osc[(3, 0, 0, 0, 0, 0, 0, 0, 0)]

print(len(re4_grid)*len(im4_grid)*len(diag4_grid))
print(len(re3_grid)*len(im3_grid)*len(diag3_grid))

parameter_points = []

for re3 in re3_grid:
    for im3 in im3_grid:
        for diag3 in diag3_grid:
            params = (3, 0, 0, re3, im3, 0, 0, 0, diag3)
            parameter_points.append(params)

for re4 in re4_grid:
    for im4 in im4_grid:
        for diag4 in diag4_grid:
            params = (4, 0, 0, re4, im4, 0, 0, 0, diag4)
            parameter_points.append(params)

items = parameter_points

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
