import matplotlib

matplotlib.use("agg")
import matplotlib.style

#matplotlib.style.use("./paper.mplstyle")
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
import json
import functools
import meander

outdir = "./plots/lv_dim3/"

fnames = glob.glob(
    "/n/holyscratch01/arguelles_delgado_lab/Lab/DUNEAnalysis/store/scans/lv/dim3/*.json"
)

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

def extract_params(entry):
    params = (
            entry["operator_dimension"],
            entry["lv_emu_re"],
            entry["lv_emu_im"],
            entry["lv_mutau_re"],
            entry["lv_mutau_im"],
            entry["lv_etau_re"],
            entry["lv_etau_im"],
            entry["lv_ee"],
            entry["lv_mumu"],
            )
    return params

def closest_point_nd(point, points):
    points = np.array(points)
    point = np.array(point)
    diff = np.abs(points - point[None,:])
    s = (np.abs(points) + np.abs(point)[None,:])/2.0
    s[s == 0] = 1.0
    i = np.argmin(np.amax(diff/s, axis=1))
    return points[i]

def closest_point_1d(point, points):
    points = np.array(points)
    diff = np.abs(points - point)
    i = np.argmin(diff)
    return points[i]

def convert_point(params):
    b = params[3]
    c = params[4]
    a = params[8]
    a2 = a**2
    b2 = b**2
    c2 = c**2
    bc2 = b2 + c2
    bc = np.sqrt(bc2)
    rho = np.sqrt(a2 + bc2)
    if rho == 0:
        f1 = 0
    else:
        f1 = a / rho
    if bc == 0:
        f2 = 0
    else:
        f2 = b / bc
    return rho, f1, f2

def closest_grid_point(point):
    new_point = convert_point(point)
    rho = closest_point_1d(new_point[0], rho_3_grid)
    f1 = closest_point_1d(new_point[1], f1_3_grid)
    f2 = closest_point_1d(new_point[2], f2_3_grid)
    return rho, f1, f2

val_by_key = dict()
index_by_params = dict()
indices_by_grid_point = dict()
i = 0

for json_fname in fnames:
    f = open(json_fname, "r")
    for line in f:
        json_data = json.loads(line)
        if type(json_data) is dict:
            json_data = [json_data]
        for entry in json_data:
            keys = entry.keys()
            j = 0
            for k in keys:
                if k not in val_by_key:
                    val_by_key[k] = []
                    j += 1
                val_by_key[k].append(entry[k])
            assert j == 0 or j == len(entry)
            params = extract_params(entry)
            grid_point = closest_grid_point(params)
            for k,v in zip(["rho", "f1", "f2"], grid_point):
                if k not in val_by_key:
                    val_by_key[k] = []
                val_by_key[k].append(v)
            grid_point = grid_point[:-1]
            if grid_point not in indices_by_grid_point:
                indices_by_grid_point[grid_point] = []
            indices_by_grid_point[grid_point].append(i)

            #index_by_params[tuple(closest_point_nd(extract_params(entry)).tolist())] = i
            i += 1

keys = val_by_key.keys()
null_index = indices_by_grid_point[(0.0, 0.0)][0]
null_entry = dict([(k, val_by_key[k][null_index]) for k in keys])

points = []
llh = []
for k in indices_by_grid_point.keys():
    point = k
    llh.append(min([val_by_key["llh"][i] for i in indices_by_grid_point[k]]))
    points.append(point)


points = np.array(points)
points_mask = points[:,0] > 0
c_points = np.copy(points)[points_mask]
llh = np.array(llh)[points_mask]
c_points[:, 0] = np.log10(c_points[:, 0])

a_min = np.amin(c_points[:, 0])
b_min = np.amin(c_points[:, 1])
a_max = np.amax(c_points[:, 0])
b_max = np.amax(c_points[:, 1])
a_points = np.unique(c_points[:, 0])
b_points = np.unique(c_points[:, 1])
a_points.sort()
b_points.sort()
delta_a = np.amin(np.diff(a_points))
delta_b = np.amin(np.diff(b_points))
top = np.zeros((len(a_points), 2))
bottom = np.zeros((len(a_points), 2))
right = np.zeros((len(b_points), 2))
left = np.zeros((len(b_points), 2))

top[:, 1] = b_max + delta_b
top[:, 0] = a_points
# top[0,0] = a_min - delta_a
# top[-1,0] = a_max + delta_a

bottom[:, 1] = b_min - delta_b
bottom[:, 0] = a_points
# bottom[0,0] = a_min - delta_a
# bottom[-1,0] = a_max + delta_a

right[:, 0] = a_max + delta_a
right[:, 1] = b_points

left[:, 0] = a_min - delta_a
left[:, 1] = b_points

sample_points = np.concatenate((c_points[:, 0:2], top, bottom, right, left))
#sample_points = points[:, 0:2]

this_llh = np.array(
    np.concatenate(
        [llh, [np.finfo(np.float64).max] * (len(a_points) * 2 + len(b_points) * 2)]
    )
).astype(float)
#this_llh = llh

#points_mask = np.logical_and(~np.isnan(this_llh), sample_points[:,0] > 0)
points_mask = ~np.isnan(this_llh)
null_llh = null_entry["llh"]
samples = this_llh - null_llh
samples = samples[points_mask]
samples[samples < 0] = 0
sample_points = sample_points[points_mask, :]

sigmas = np.array([1, 2])

# Set the number of degrees of freedom
dof = 2

# Convert the sigma value to a percentage
proportions = scipy.special.erf(sigmas / np.sqrt(2.0))
# proportions = np.array([0.9, 0.99])
print("proportions:", proportions)

# Calculate the critical delta LLH for each confidence level
levels = scipy.special.gammaincinv(dof / 2.0, np.array(proportions))
print("levels:", levels)

print(np.shape(sample_points))
print(np.shape(samples))
print(samples)
c_sample_points = np.copy(sample_points)
#c_sample_points[:,0] = np.log10(c_sample_points[:,0])
contours_by_level = meander.compute_contours(c_sample_points, samples, levels)

fig, ax = plt.subplots(figsize=(7, 5))

i = 0
labels = ["DUNE atmospheric"]

cm = plt.get_cmap("plasma")
print(samples[samples < 0])
good_samples = functools.reduce(np.logical_and, [samples > 0, ~np.isinf(samples), ~np.isnan(samples), samples < np.finfo(np.float64).max, sample_points[:,0] <= a_max, sample_points[:,0] >= a_min, sample_points[:,1] <= b_max, sample_points[:,1] >= b_min])
color_min = np.amin(np.log10(samples[good_samples]))
color_max = np.amax(np.log10(samples[good_samples]))
# print (color_min, color_max)
for j in range(sample_points.shape[0]):
    color_val = min(
        max(
            np.log10((samples[j]) - (color_min)) / ((color_max) - (color_min)), 0.0
        ),
        1.0,
    )
    print([sample_points[j, 0]], [sample_points[j, 1]], color_val)
    color = cm(color_val)
    if samples[j] <= levels[0]:
        color = "black"
    ax.plot(
        [sample_points[j, 0]],
        [sample_points[j, 1]],
        color=color,
        marker="o",
        linestyle="none",
    )

color = "orange"
level_line_styles = ["-", "--", ":", ":", "-", "--", ":", ":"]
for j, contours in enumerate(contours_by_level):
    sigma = proportions[j] * 100.0
    if np.around(sigma) == sigma:
        s = str(int(sigma))
    else:
        s = "%0.1f" % sigma
    #label = labels[i] + r" $" + s + r"\%$"
    label = labels[i] + r" " + s + r"%"
    label_done = False
    linestyle = level_line_styles[j]
    for contour in contours:
        contour = np.array(contour)
        contour[:, 0] = 10**contour[:,0]
        if label_done:
            ax.plot(
                contour[:, 0],
                contour[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=2,
            )
        else:
            ax.plot(
                contour[:, 0],
                contour[:, 1],
                color=color,
                linestyle=linestyle,
                label=label,
                linewidth=2,
            )
            label_done = True

dim = 3

char = None
if dim % 2 == 0:
    char = "c"
else:
    char = "a"

ax.set_xlabel("rho")
ax.set_ylabel(char + "/rho")
# ax.set_xlim((0.001, 1))
# ax.set_ylim((0.01, 100))
sample_points[:, 0] = 10.0**sample_points[:, 0]
a_min = np.amin(sample_points[good_samples,0])
a_max = np.amax(sample_points[good_samples,0])
b_min = np.amin(sample_points[good_samples,1])
b_max = np.amax(sample_points[good_samples,1])
ax.set_xlim((a_min, a_max))
ax.set_ylim((b_min, b_max))
ax.set_ylim((-1.0, 1.0))
print(a_min, a_max)
print(b_min, b_max)
ax.set_xscale("log")
ax.legend()
#fig.tight_layout()
fig.savefig(outdir + "lv_dim" + ("%d" % dim) + "_sensitivity.pdf")
fig.savefig(outdir + "lv_dim" + ("%d" % dim) + "_sensitivity.png", dpi=200)
