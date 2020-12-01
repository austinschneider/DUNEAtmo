import numpy as np
import nuflux
import nuSQUIDSpy as nsq
import oscillator

units = nsq.Const()
ebins = np.logspace(1, 6, 100 + 1) * units.GeV
czbins = np.linspace(-1, 1, 100 + 1)

re3_grid = np.logspace(-25, -22, 5*3+1)
im3_grid = np.logspace(-25, -22, 5*3+1)

re4_grid = np.logspace(-29, -26, 5*3+1)
im4_grid = np.logspace(-29, -26, 5*3+1)

flux = nuflux.makeFlux("H3a_SIBYLL23C")
osc = oscillator.oscillator(
    "H3a_SIBYLL23C", flux, ebins, czbins, "lv", "./fluxes/", cache_size=10
)

osc[(3, 0, 0, 0, 0)]

for re3 in re3_grid:
    params = (3, 0, 0, re3, 0)
    osc[params]

for im3 in im3_grid:
    params = (3, 0, 0, 0, im3)
    osc[params]

for re4 in re4_grid:
    params = (4, 0, 0, re4, 0)
    osc[params]

for im4 in im4_grid:
    params = (4, 0, 0, 0, im4)
    osc[params]

for re3 in re3_grid:
    for im3 in im3_grid:
        params = (3, 0, 0, re3, im3)
        osc[params]

for re4 in re4_grid:
    for im4 in im4_grid:
        params = (4, 0, 0, re4, im4)
        osc[params]
