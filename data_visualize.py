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

# from wordcloud import WordCloud


# events histogram
Afile_path = "data/Afile"
origin_catalog = pd.read_csv(f"{Afile_path}/final catalog.csv")
# catalog=pd.read_csv(f"{Afile_path}/final catalog (station exist).csv")

validation_year = 2018
fig, ax = plt.subplots()
ax.hist(origin_catalog["magnitude"], bins=30, ec="black", label="train")
# year_filter=(origin_catalog["year"]==validation_year)
# ax.hist(origin_catalog[year_filter]["magnitude"],bins=30,ec='black',label="validation")
ax.set_yscale("log")
ax.set_xlabel("Magnitude")
ax.set_ylabel("number of events")
ax.set_title(f"1991-2020 TSIMP data: validate on {validation_year}")
ax.set_title(f"1991-2020 TSIMP event catalog")
ax.legend()

# traces pga histogram
# traces_catalog=pd.read_csv(f"{Afile_path}/2012-2020 traces with picking and label_new (sta location exist).csv")
traces_catalog = pd.read_csv(
    f"{Afile_path}/1991-2020 traces with picking and label_new.csv"
)
merged_catalog = pd.merge(traces_catalog, origin_catalog, how="left", on="EQ_ID")
oldtime_catalog = merged_catalog[
    (merged_catalog["year_y"] >= 1999)
    & (merged_catalog["year_y"] <= 2020)
    & (merged_catalog["magnitude"] >= 5.5)
]
newtime_catalog = merged_catalog[
    (merged_catalog["year_y"] >= 2018)
    & (merged_catalog["year_y"] <= 2020)
    & (merged_catalog["magnitude"] >= 4.5)
    & (merged_catalog["magnitude"] < 5.5)
]
now_catalog = merged_catalog[
    (merged_catalog["year_y"] >= 2012) & (merged_catalog["year_y"] <= 2020)
]
concat_catalog = pd.concat([oldtime_catalog, newtime_catalog])
# validation_year=2018
PGA = np.sqrt(
    now_catalog["pga_z"] ** 2 + now_catalog["pga_ns"] ** 2 + now_catalog["pga_ew"] ** 2
)
PGA_filtered = np.sqrt(
    concat_catalog["pga_z"] ** 2
    + concat_catalog["pga_ns"] ** 2
    + concat_catalog["pga_ew"] ** 2
)
fig, ax = plt.subplots()
ax.hist(np.log10(PGA / 100), bins=80, ec="black")
ax.hist(np.log10(PGA_filtered / 100), bins=80, ec="black", alpha=0.5)
# ax.hist(traces_catalog["pga"],bins=30,ec='black',label="train")
# year_filter=(traces_catalog["year"]==validation_year)
# ax.hist(traces_catalog[year_filter]["pga"],bins=30,ec='black',label="validation")
pga_threshold = np.log10(
    [1e-5, 0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0, 10]
)
# pga_threshold = [100*1e-5, 100*0.008, 100*0.025, 100*0.080, 100*0.250,
#                 100*0.80, 100*1.4, 100*2.5, 100*4.4, 100*8.0,100*10]
label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
ax.vlines(pga_threshold[1:-1], 0, 17700, linestyles="dotted", color="k")
for i in range(len(label)):
    ax.text((pga_threshold[i] + pga_threshold[i + 1]) / 2, 15000, label[i])
ax.set_yscale("log")
# ax.set_xlabel("log (PGA m/s2)")
ax.set_xlabel("PGA (gal)")
ax.set_ylabel("number of traces")
ax.set_title(f"TSMIP traces")
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
