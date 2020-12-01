import matplotlib
matplotlib.use('agg')
import matplotlib.style
matplotlib.style.use('./paper.mplstyle')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import argparse
import glob
import json

import numpy as np
import h5py as h5
import LWpy
import LeptonInjector
import nuSQUIDSpy as nsq
import json
import functools
import meander

outdir = './plots/lv/'

json_fname = './lv_scan.json'
f = open(json_fname, 'r')
json_data = json.load(f)

val_by_key = dict()
index_by_params = dict()
i = 0
for entry in json_data:
    keys = entry.keys()
    j = 0
    for k in keys:
        if k not in val_by_key:
            val_by_key[k] = []
            j += 1
        val_by_key[k].append(entry[k])
    assert(j == 0 or j == len(entry))
    index_by_params[(int(entry['operator_dimension']), float(entry['lv_mutau_re']), float(entry['lv_mutau_im']))] = 1
    i += 1

keys = val_by_key.keys()
null_index = index_by_params[(3, 0.0, 0.0)]
null_entry = dict([(k, val_by_key[k][null_index]) for k in keys])
for k in keys:
    del val_by_key[k][null_index]
    val_by_key[k] = np.array(val_by_key[k])

for dim in [3,4]:
    mask = val_by_key['operator_dimension'] == dim
    lv_mutau_re = val_by_key['lv_mutau_re']
    lv_mutau_im = val_by_key['lv_mutau_im']
    mask = functools.reduce(np.logical_and, [val_by_key['operator_dimension'] == dim, lv_mutau_re > 0, lv_mutau_im > 0])
    lv_mutau_re = lv_mutau_re[mask]
    lv_mutau_im = lv_mutau_im[mask]
    llh = val_by_key['llh'][mask]

    points = np.array([lv_mutau_re, lv_mutau_im]).T

    a_min = np.amin(points[:,0])
    b_min = np.amin(points[:,1])
    a_max = np.amax(points[:,0])
    b_max = np.amax(points[:,1])
    a_points = np.unique(points[:,0])
    b_points = np.unique(points[:,1])
    a_points.sort()
    b_points.sort()
    delta_a = np.amin(np.diff(a_points))
    delta_b = np.amin(np.diff(b_points))
    top = np.zeros((len(a_points), 2))
    bottom = np.zeros((len(a_points), 2))
    right = np.zeros((len(b_points), 2))
    left = np.zeros((len(b_points), 2))

    top[:,1] = b_max + delta_b
    top[:,0] = a_points
    #top[0,0] = a_min - delta_a
    #top[-1,0] = a_max + delta_a

    bottom[:,1] = b_min - delta_b
    bottom[:,0] = a_points
    #bottom[0,0] = a_min - delta_a
    #bottom[-1,0] = a_max + delta_a

    right[:,0] = a_max + delta_a
    right[:,1] = b_points

    left[:,0] = a_min - delta_a
    left[:,1] = b_points

    sample_points = np.concatenate((points[:,0:2], top, bottom, right, left))

    this_llh = np.array(np.concatenate([llh, [np.finfo(np.float64).max] *(len(a_points)*2 + len(b_points)*2)])).astype(float)
    points_mask = ~np.isnan(this_llh)
    null_llh = null_entry['llh']
    samples = this_llh - null_llh
    samples = samples[points_mask]
    sample_points = sample_points[points_mask,:]

    sigmas = np.array([1, 2])

    # Set the number of degrees of freedom
    dof = 2

    # Convert the sigma value to a percentage
    proportions = scipy.special.erf(sigmas / np.sqrt(2.0))
    #proportions = np.array([0.9, 0.99])
    print("proportions:", proportions)

    # Calculate the critical delta LLH for each confidence level
    levels = scipy.special.gammaincinv(dof / 2.0, np.array(proportions))
    print("levels:", levels)

    contours_by_level = meander.compute_contours(sample_points, samples, levels)

    fig, ax = plt.subplots(figsize=(7,5))

    i = 0
    labels = ["DUNE atmospheric"]

    cm = plt.get_cmap('plasma')
    print(samples[samples < 0])
    color_min = np.amin(np.log10(samples[~np.isnan(samples)]))
    color_max = np.amax(np.log10(samples[~np.isnan(samples)]))
    #print (color_min, color_max)
    for j in range(sample_points.shape[0]):
        color_val = min(max(np.log10((samples[j])-(color_min))/((color_max)-(color_min)), 0.0), 1.0)
        print([sample_points[j,0]], [sample_points[j,1]], color_val)
        ax.plot([sample_points[j,0]], [sample_points[j,1]], color=cm(color_val), marker='o', linestyle='none')

    color = 'orange'
    level_line_styles = ['-','--',':',':','-','--',':',':']
    for j, contours in enumerate(contours_by_level):
        sigma = proportions[j]*100.0
        if np.around(sigma) == sigma:
            s = str(int(sigma))
        else:
            s = '%0.1f' % sigma
        label = labels[i] + r' $' + s + r'\%$'
        label_done = False
        linestyle = level_line_styles[j]
        for contour in contours:
            contour = np.array(contour)
            if label_done:
                ax.plot(contour[:,0],contour[:,1], color=color, linestyle=linestyle, linewidth=2)
            else:
                ax.plot(contour[:,0],contour[:,1], color=color, linestyle=linestyle, label=label, linewidth=2)
                label_done = True

    char = None
    if dim % 2 == 0:
        char = 'c'
    else:
        char = 'a'
    power = (4-dim)
    if power == 0:
        units = ""
    else:
        if power == 1:
            units = r"[\textmd{GeV}]"
        else:
            units = r"[\textmd{GeV}^{" + str(power) + r"}]"

    ax.set_xlabel(r'$\textmd{Re}(\mathring{' + char + r'}^{(' + ('%d' % dim) + r')}_{\mu\tau})' + units + r'$')
    ax.set_ylabel(r'$\textmd{Im}(\mathring{' + char + r'}^{(' + ('%d' % dim) + r')}_{\mu\tau})' + units + r'$')
    #ax.set_xlim((0.001, 1))
    #ax.set_ylim((0.01, 100))
    ax.set_xlim((a_min, a_max))
    ax.set_ylim((b_min, b_max))
    print(a_min, a_max)
    print(b_min, b_max)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir + 'lv_dim' + ('%d' % dim) + '_sensitivity.pdf')
    fig.savefig(outdir + 'lv_dim' + ('%d' % dim) + '_sensitivity.png', dpi=200)
