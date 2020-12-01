import sys
import os
import os.path
base_path = os.environ['GOLEMSPACE']
core_path = base_path + '/sources/DUNEAtmo/likelihood/core/'
sys.path.insert(0, core_path)
import numpy as np
import nuflux
import nuSQUIDSpy as nsq
import oscillator

units = nsq.Const()
ebins = np.logspace(1, 6, 100 + 1) * units.GeV
czbins = np.linspace(-1, 1, 100 + 1)

dm2_grid = np.logspace(-1, 1, 10*2+1)
th14 = 0.0
s22th24_grid = np.logspace(-2, 0, 10*2+1)
th24_grid = np.arcsin(np.sqrt(s22th24_grid)) / 2.0
th34 = 0.0
cp = 1.0

flux = nuflux.makeFlux("H3a_SIBYLL23C")
osc = oscillator.oscillator(
    "H3a_SIBYLL23C", flux, ebins, czbins, "sterile", "./fluxes/", cache_size=10
)

for dm2 in dm2_grid:
    for th24 in th24_grid:
        params = (4, dm2, th14, th24, th34, cp)
        osc[params]
