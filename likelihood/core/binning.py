import numpy as np
import functools

def get_bins(
    tg_emin = 1e2,
    tg_emax = 1e5,
    tg_ebins = 30,
    start_emin = 1e2,
    start_emax = 1e5,
    start_ebins = 30,
    tg_czmin  = -1,
    tg_czmax  = 0,
    tg_czbins = 20,
    start_czmin  = -1,
    start_czmax  = 1,
    start_czbins = 40,
):
    """
    Get the analysis bins.
    Using argument defaults will return the default analysis binning used in the paper
    """
    tg_energy_bins = np.logspace(np.log10(tg_emin), np.log10(tg_emax), tg_ebins+1)
    start_energy_bins = np.logspace(np.log10(start_emin), np.log10(start_emax), start_ebins+1)
    tg_zenith_bins = np.arccos(np.linspace(tg_czmin, tg_czmax, tg_czbins+1))[::-1]
    start_zenith_bins = np.arccos(np.linspace(start_czmin, start_czmax, start_czbins+1))[::-1]
    return tg_energy_bins, start_energy_bins, tg_zenith_bins, start_zenith_bins

def get_bin_masks(
    morphology,
    energy, zenith, through_going_energy_bins, starting_energy_bins, through_going_zenith_bins, starting_zenith_bins,
):
    """
    Get masks for all analysis bins
    Returns 3 sets of masks: cascade bins, track bins, double cascade bins
    """

    # Define the masks for the two morphologies
    # missing = 0
    # starting = 1
    # stopping = 2
    # through_going = 3
    # contained = 4
    # pseudo_contained = 5
    mask_0 = morphology == 3
    mask_1 = functools.reduce(np.logical_or, [morphology == 1, morphology == 4, morphology == 5])

    def make_bin_masks(energies, zeniths, energy_bins, zenith_bins):
        """
        Takes energy/zenith quantities and their bins
        Returns masks correponding to those bins
        """

        assert len(energies) == len(zeniths)

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

    # Cascades are binned in energy and zenith
    through_going_energy_bins = np.array(through_going_energy_bins.flatten())
    masks_0 = np.logical_and(
        make_bin_masks(energy, zenith, through_going_energy_bins, through_going_zenith_bins), mask_0
    )

    # Tracks are binned in energy and zenith
    starting_energy_bins = np.array(starting_energy_bins.flatten())
    masks_1 = np.logical_and(
        make_bin_masks(energy, zenith, starting_energy_bins, starting_zenith_bins), mask_1
    )

    return masks_0, masks_1


def sort_by_bin(data, masks):
    """ Returns the data/MC events sorted by masks, and returns the slices for each mask
    Parameters
    ----------
    data : array_like
        list of data/MC events
    masks : array_like
        list of masks
        masks should be the same length as data

    Return
    ---------
    sorted_data: array_like
        list of 'data', sorted by bin

    bin_slices: array_like
        list of bin slices. Each element in bin_slices, gives the slice that returns the data elements in a particular bin
    """
    no_mask = ~functools.reduce(np.logical_or, masks)
    masks = list(masks) + [no_mask]

    sorted_data = np.empty(data.shape, dtype=data.dtype)
    bin_edge = 0
    bin_slices = []
    for mask in masks:
        n_events = np.sum(mask)
        bin_slices.append(slice(bin_edge, bin_edge + n_events))
        sorted_data[bin_edge : bin_edge + n_events] = data[mask]
        bin_edge += n_events

    return sorted_data, bin_slices[:-1]


def bin_data(data, **kwargs):
    """ Returns the data/MC events sorted by bins, and returns the bin slices for each bin

    Parameters
    ----------
    data : array_like
        list of data/MC events

    Return
    ---------
    sorted_data: array_like
        list of 'data', sorted by bin

    bin_slices: array_like
        list of bin slices. Each element in bin_slices, gives the slice that returns the data elements in a particular bin
    """

    # Get the bin edges
    bins = get_bins(**kwargs)
    tg_energy_bins, start_energy_bins, tg_zenith_bins, start_zenith_bins = bins

    # Get the corresponding masks: data[mask] --> events in the bin corresponding to mask
    tg_masks, start_masks = get_bin_masks(
        data["morphology"],
        data["recoEnergy"],
        data["recoZenith"],
        tg_energy_bins,
        start_energy_bins,
        tg_zenith_bins,
        start_zenith_bins,
    )

    # Combine masks in one array
    masks = np.concatenate([tg_masks, start_masks], axis=0)

    # Sort the data by bin
    sorted_data, bin_slices = sort_by_bin(data, masks)

    return sorted_data, bin_slices
