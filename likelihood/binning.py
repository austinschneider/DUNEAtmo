import numpy as np


def get_bins(
    emin   = 1e-1,
    emax   = 1e5,
    ebins  = 18,
    czmin  = -1,
    czmax  = 0,
    czbins = 5,
):
    """
    Get the analysis bins.
    Using argument defaults will return the default analysis binning used in the paper
    """
    energy_bins = np.logspace(np.log10(emin), np.log10(emax), ebins+1)
    zenith_bins = np.cos(np.linspace(czmin, czmax, czbins+1))[::-1]
    return energy_bins, zenith_bins


def get_bin_masks(
    energy, zenith, energy_bins, zenith_bins
):
    """
    Get masks for all analysis bins
    """

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

    masks = make_bin_masks(energy, zenith, energy_bins, zenith_bins)

    return masks


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

    sorted_data = np.empty(data.shape, dtype=data.dtype)
    bin_edge = 0
    bin_slices = []
    for mask in masks:
        n_events = np.sum(mask)
        bin_slices.append(slice(bin_edge, bin_edge + n_events))
        sorted_data[bin_edge : bin_edge + n_events] = data[mask]
        bin_edge += n_events

    return sorted_data, bin_slices


def bin_data(data):
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
    bins = get_bins()

    # Get the corresponding masks: data[mask] --> events in the bin corresponding to mask
    masks = get_bin_masks(
        data["recoEnergy"],
        data["recoZenith"],
        energy_bins,
        zenith_bins,
    )

    # Sort the data by bin
    sorted_data, bin_slices = sort_by_bin(data, masks)

    return sorted_data, bin_slices
