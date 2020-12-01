import numpy as np
import json
import functools

def load_data(fname='./weighted/weighted.json'):
    data = json.load(open(fname, 'r'))
    energy = np.array(data["energy"])
    zenith = np.pi - np.array(data["zenith"])
    azimuth = 2.*np.pi - np.array(data["azimuth"])
    bjorken_x = np.array(data["bjorken_x"])
    bjorken_y = np.array(data["bjorken_y"])
    final_type_0 = np.array(data["final_type_0"]).astype(int)
    final_type_1 = np.array(data["final_type_1"]).astype(int)
    particle = np.array(data["particle"]).astype(int)
    x = np.array(data["x"])
    y = np.array(data["y"])
    z = np.array(data["z"])
    total_column_depth = np.array(data["total_column_depth"])
    gen_prob = np.array(data["gen_prob"])
    mu_energy = np.array(data["mu_energy"])
    mu_x = np.array(data["mu_x"])
    mu_y = np.array(data["mu_y"])
    mu_z = np.array(data["mu_z"])
    mu_zenith = np.pi - np.array(data["mu_zenith"])
    mu_azimuth = 2.*np.pi - np.array(data["mu_azimuth"])
    entry_energy = np.array(data["entry_energy"])
    entry_x = np.array(data["entry_x"])
    entry_y = np.array(data["entry_y"])
    entry_z = np.array(data["entry_z"])
    entry_zenith = np.pi - np.array(data["entry_zenith"])
    entry_azimuth = 2.*np.pi - np.array(data["entry_azimuth"])
    exit_energy = np.array(data["exit_energy"])
    exit_x = np.array(data["exit_x"])
    exit_y = np.array(data["exit_y"])
    exit_z = np.array(data["exit_z"])
    exit_zenith = np.pi - np.array(data["exit_zenith"])
    exit_azimuth = 2.*np.pi - np.array(data["exit_azimuth"])
    track_length = np.array(data["track_length"])
    morphology = np.array(data["morphology"])
    deposited_energy = np.array(data["deposited_energy"])
    entry_distance = np.sqrt((entry_x - x)**2 + (entry_y - y)**2 + (entry_z - z)**2)
    injector_count = np.array(data["injector_count"]).astype(int)[::-1]
    muon_start_energy = np.array(entry_energy)
    mask = np.isnan(muon_start_energy)
    muon_start_energy[mask] = mu_energy[mask]
    muon_start_zenith = np.array(entry_zenith)
    mask = np.isnan(muon_start_zenith)
    muon_start_zenith[mask] = mu_zenith[mask]
    muon_nx = exit_x - entry_x
    muon_ny = exit_y - entry_y
    muon_nz = exit_z - entry_z
    muon_d = np.sqrt(muon_nx**2 + muon_ny**2 + muon_nz**2)
    muon_nx /= muon_d
    muon_ny /= muon_d
    muon_nz /= muon_d
    muon_zenith = np.pi - np.arccos(muon_nz)
    muon_azimuth = 2.*np.pi - np.arctan2(muon_ny, muon_nx)

    rand = np.random.default_rng(seed=2303)
    mask_0 = morphology == 3
    mask_1 = functools.reduce(np.logical_or, [morphology == 1, morphology == 4, morphology == 5])
    #factor_0 = rand.lognormal(mean=0.0, sigma=np.log10(2.0), size=np.sum(mask_0))
    factor_0 = rand.lognormal(mean=0.0, sigma=(abs(np.log10(1.2)) + abs(np.log10(0.8)))/2., size=np.sum(mask_0))
    factor_1 = rand.lognormal(mean=0.0, sigma=(abs(np.log10(1.1)) + abs(np.log10(0.9)))/2., size=np.sum(mask_1))
    reco_energy = np.empty(energy.shape)
    reco_energy[mask_0] = muon_start_energy[mask_0] * factor_0
    reco_energy[mask_1] = energy[mask_1] * factor_1

    data = np.empty(len(energy), dtype=[
          ("energy", energy.dtype),
          ("zenith", zenith.dtype),
          ("azimuth", azimuth.dtype),
          ("bjorken_x", bjorken_x.dtype),
          ("bjorken_y", bjorken_y.dtype),
          ("final_type_0", final_type_0.dtype),
          ("final_type_1", final_type_1.dtype),
          ("particle", particle.dtype),
          ("x", x.dtype),
          ("y", y.dtype),
          ("z", z.dtype),
          ("total_column_depth", total_column_depth.dtype),
          ("gen_prob", gen_prob.dtype),
          ("entry_energy", entry_energy.dtype),
          ("entry_x", entry_x.dtype),
          ("entry_y", entry_y.dtype),
          ("entry_z", entry_z.dtype),
          ("entry_zenith", entry_zenith.dtype),
          ("entry_azimuth", entry_azimuth.dtype),
          ("exit_energy", exit_energy.dtype),
          ("exit_x", exit_x.dtype),
          ("exit_y", exit_y.dtype),
          ("exit_z", exit_z.dtype),
          ("exit_zenith", exit_zenith.dtype),
          ("exit_azimuth", exit_azimuth.dtype),
          ("track_length", track_length.dtype),
          ("morphology", morphology.dtype),
          ("deposited_energy", deposited_energy.dtype),
          ("entry_distance", entry_distance.dtype),
          ("muon_start_energy", muon_start_energy.dtype),
          ("muon_start_zenith", muon_start_zenith.dtype),
          ("recoEnergy", reco_energy.dtype),
          ("recoZenith", muon_start_zenith.dtype),
          ])

    data["energy"] = energy
    data["zenith"] = zenith
    data["azimuth"] = azimuth
    data["bjorken_x"] = bjorken_x
    data["bjorken_y"] = bjorken_y
    data["final_type_0"] = final_type_0
    data["final_type_1"] = final_type_1
    data["particle"] = particle
    data["x"] = x
    data["y"] = y
    data["z"] = z
    data["total_column_depth"] = total_column_depth
    data["gen_prob"] = gen_prob
    data["entry_energy"] = entry_energy
    data["entry_x"] = entry_x
    data["entry_y"] = entry_y
    data["entry_z"] = entry_z
    data["entry_zenith"] = entry_zenith
    data["entry_azimuth"] = entry_azimuth
    data["exit_energy"] = exit_energy
    data["exit_x"] = exit_x
    data["exit_y"] = exit_y
    data["exit_z"] = exit_z
    data["exit_zenith"] = exit_zenith
    data["exit_azimuth"] = exit_azimuth
    data["track_length"] = track_length
    data["morphology"] = morphology
    data["deposited_energy"] = deposited_energy
    data["entry_distance"] = entry_distance
    data["muon_start_energy"] = muon_start_energy
    data["muon_start_zenith"] = muon_start_zenith
    data["recoEnergy"] = reco_energy
    data["recoZenith"] = muon_start_zenith

    return data

