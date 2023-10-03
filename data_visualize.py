from itertools import repeat

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchsampler import ImbalancedDatasetSampler
from tqdm import tqdm

from multiple_sta_dataset import (
    CustomSubset,
    multiple_station_dataset,
)
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl import ticker


# events histogram
data_path = "data preprocess/events_traces_catalog"
origin_catalog = pd.read_csv(f"{data_path}/1999_2019_final_catalog.csv")

test_year = 2016
train_catalog=origin_catalog.query(f"year!={test_year}")
test_catalog=origin_catalog.query(f"year=={test_year}")
fig, ax = plt.subplots()
ax.hist(train_catalog["depth"], bins=30, ec="black", label="train")
ax.hist(test_catalog["depth"],bins=30, ec='black',label="test",alpha=0.8)
ax.set_yscale("log")
ax.set_xlabel("Depth",fontsize=15)
ax.set_ylabel("Number of events",fontsize=15)
ax.legend()
# fig.savefig(f"paper image/event depth distribution.png",dpi=300)
# fig.savefig(f"paper image/event depth distribution.pdf",dpi=300)

#event distribution in map
src_crs = ccrs.PlateCarree()
fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))
ax_map.coastlines("10m")
ax_map.scatter(
    train_catalog["lon"]+train_catalog["lon_minute"]/60,
    train_catalog["lat"]+train_catalog["lat_minute"]/60,
    edgecolors="k",
    linewidth=1,
    marker="o",
    c="grey",
    s=2**train_catalog["magnitude"],
    zorder=3,
    alpha=0.5,
    label="train"
)
ax_map.scatter(
    test_catalog["lon"]+test_catalog["lon_minute"]/60,
    test_catalog["lat"]+test_catalog["lat_minute"]/60,
    edgecolors="k",
    linewidth=1,
    marker="o",
    c="orange",
    s=2**test_catalog["magnitude"],
    zorder=3,
    alpha=0.5,
    label="test"
)
ax_map.add_feature(
    cartopy.feature.OCEAN, edgecolor="k"
)

xmin, xmax = ax_map.get_xlim()
ymin, ymax = ax_map.get_ylim()
xticks = ticker.LongitudeLocator(nbins=5)._raw_ticks(xmin, xmax)
yticks = ticker.LatitudeLocator(nbins=5)._raw_ticks(ymin, ymax)

ax_map.set_xticks(xticks, crs=ccrs.PlateCarree())
ax_map.set_yticks(yticks, crs=ccrs.PlateCarree())

ax_map.xaxis.set_major_formatter(
    ticker.LongitudeFormatter(zero_direction_label=True)
)
ax_map.yaxis.set_major_formatter(ticker.LatitudeFormatter())

ax_map.xaxis.set_ticks_position("both")
ax_map.yaxis.set_ticks_position("both")
ax_map.legend()
fig.savefig(f"paper image/event distribution map.png",dpi=300)
fig.savefig(f"paper image/event distribution map.pdf",dpi=300)

# traces pga histogram
traces_catalog = pd.read_csv(
    f"{data_path}/1999_2019_final_traces_Vs30.csv"
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(traces_catalog.query(f"year!={test_year}")["pga"], bins=30, ec="black", label="train")
ax.hist(traces_catalog.query(f"year=={test_year}")["pga"], bins=30,alpha=0.8, ec="black",label="test")
pga_threshold = np.log10(
    [1e-5, 0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0, 10]
)
label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
ax.vlines(pga_threshold[1:-1], 0, 17700, linestyles="dotted", color="k")
for i in range(len(label)):
    if label[i]=="0":
        continue
    ax.text(((pga_threshold[i] + pga_threshold[i + 1]) / 2)-0.05, 15000, label[i])
ax.set_yscale("log")
ax.set_xlabel(r"PGA log(${m/s^2}$)",fontsize=15)
ax.set_ylabel("Number of traces",fontsize=15)
fig.legend(fontsize=13)
fig.savefig(f"paper image/trace pga distribution.png",dpi=300)
fig.savefig(f"paper image/trace pga distribution.pdf",dpi=300)
# ax.set_title(f"2012-2020 TSIMP data: validate on {validation_year}")
# ax.legend(loc="upper left")


# training data (add oversampling)
origin_data = multiple_station_dataset(
    "D:/TEAM_TSMIP/data/TSMIP_1999_2019_Vs30.hdf5",
    mode="train",
    mask_waveform_sec=3,
    weight_label=False,
    oversample=1,
    oversample_mag=4,
    test_year=2016,
    mask_waveform_random=True,
    mag_threshold=0,
    label_key="pga",
    input_type="acc",
    data_length_sec=15,
    station_blind=True,
)
oversampled_data = multiple_station_dataset(
    "D:/TEAM_TSMIP/data/TSMIP_1999_2019_Vs30.hdf5",
    mode="train",
    mask_waveform_sec=3,
    weight_label=False,
    oversample=1.5,
    oversample_mag=4,
    test_year=2016,
    mask_waveform_random=True,
    mag_threshold=0,
    label_key="pga",
    input_type="acc",
    data_length_sec=15,
    station_blind=True,
)
bias_closed_sta_data = multiple_station_dataset(
    "D:/TEAM_TSMIP/data/TSMIP_1999_2019_Vs30.hdf5",
    mode="train",
    mask_waveform_sec=3,
    weight_label=False,
    oversample=1.5,
    oversample_mag=4,
    test_year=2016,
    mask_waveform_random=True,
    mag_threshold=0,
    label_key="pga",
    input_type="acc",
    data_length_sec=15,
    station_blind=True,
    bias_to_closer_station=True,
)

origin_loader = DataLoader(origin_data, batch_size=16)
oversampled_loader = DataLoader(dataset=oversampled_data, batch_size=16)
bias_closed_sta_loader = DataLoader(dataset=bias_closed_sta_data, batch_size=16)


origin_PGA = []
for sample in tqdm(origin_loader):
    tmp_pga = torch.index_select(
        sample["label"].flatten(), 0, sample["label"].flatten().nonzero().flatten()
    ).tolist()
    origin_PGA.extend(tmp_pga)
origin_PGA_array = np.array(origin_PGA)
high_intensity_rate = np.sum(origin_PGA_array > np.log10(0.250)) / len(origin_PGA_array)
print(f"origin rate:{high_intensity_rate}")

oversampled_PGA = []
for sample in tqdm(oversampled_loader):
    tmp_pga = torch.index_select(
        sample["label"].flatten(), 0, sample["label"].flatten().nonzero().flatten()
    ).tolist()
    oversampled_PGA.extend(tmp_pga)
oversampled_PGA_array = np.array(oversampled_PGA)
oversampled_high_intensity_rate = np.sum(oversampled_PGA_array > np.log10(0.250)) / len(
    oversampled_PGA_array
)
print(f"oversampled rate:{oversampled_high_intensity_rate}")

bias_closed_sta_PGA = []
for sample in tqdm(bias_closed_sta_loader):
    tmp_pga = torch.index_select(
        sample["label"].flatten(), 0, sample["label"].flatten().nonzero().flatten()
    ).tolist()
    bias_closed_sta_PGA.extend(tmp_pga)
bias_closed_sta_PGA_array = np.array(bias_closed_sta_PGA)
bias_closed_sta_high_intensity_rate = np.sum(
    bias_closed_sta_PGA_array > np.log10(0.250)
) / len(bias_closed_sta_PGA_array)
print(f"bias_closed_sta rate:{bias_closed_sta_high_intensity_rate}")

label = ["2", "3", "4", "5-", "5+", "6-", "6+", "7"]
pga_threshold = np.log10([0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0, 10])
# label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
# pgv_threshold = np.log10(
#     [0.00001, 0.002, 0.007, 0.019, 0.057, 0.15, 0.3, 0.5, 0.8, 1.4, 20]
# )

fig, ax = plt.subplots(figsize=(7, 7))
ax.hist(bias_closed_sta_PGA, bins=32, edgecolor="k", label="bias_closed_sta")
ax.hist(oversampled_PGA, bins=32, edgecolor="k", label="oversampled", alpha=0.6)
ax.hist(origin_PGA, bins=32, edgecolor="k", label="origin", alpha=0.6)
ax.vlines(pga_threshold[1:-1], 0, 40000, linestyles="dotted", color="k")
for i in range(len(pga_threshold) - 1):
    ax.text((pga_threshold[i] + pga_threshold[i + 1]) / 2, 50000, label[i])
ax.text(
    0.01,
    0.8,
    f"high intensity rate:\norigin: {np.round(high_intensity_rate,2)}\noversampled: {np.round(oversampled_high_intensity_rate,2)}\nbias to station: {np.round(bias_closed_sta_high_intensity_rate,2)}",
    transform=ax.transAxes,
    fontsize=12,
)
ax.set_xlim(-2.75, 1.25)
ax.set_ylabel("Number of traces", size=14)
ax.set_xlabel(r"log(PGA (${m/s^2}$))", size=14)
ax.set_title("TSMIP PGA distribution in training", size=14)
ax.set_yscale("log")
fig.legend(loc="upper right")
fig.savefig("PGA distribution.png", dpi=300,bbox_inches='tight')

# test label ditribution
pre1 = pd.read_csv("predict/model1 2 3 sec 1 triggered station prediction.csv")

fig, ax = plt.subplots(figsize=(7, 7))
ax.hist(pre1["answer"], bins=32, edgecolor="k", label="origin data")
ax.vlines(pgv_threshold[1:-1], 0, 17700, linestyles="dotted", color="k")
for i in range(len(pgv_threshold) - 1):
    ax.text((pgv_threshold[i] + pgv_threshold[i + 1]) / 2, 22500, label[i])
ax.set_ylabel("number of trace")
ax.set_xlabel("log(PGA (m/s2))")
ax.set_title("TSMIP Test data PGA distribution")
ax.set_yscale("log")
fig.legend(loc="upper right")
