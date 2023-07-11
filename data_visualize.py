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


# station don't have location
sta_path = "data/station information"
Afile_path = "data/Afile"
traces = pd.read_csv(f"{Afile_path}/2012-2020 traces with picking and label_new.csv")
station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv")
sta_filter = traces["station_name"].isin(station_info["location_code"])
tmp_traces = traces[~sta_filter]
plt.bar(
    tmp_traces["station_name"].value_counts()[:5].index,
    tmp_traces["station_name"].value_counts().values[:5],
)

wordcloud = WordCloud(width=1600, height=800).generate(
    tmp_traces["station_name"].to_string()
)
plt.figure(figsize=(20, 10), facecolor="k")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

validataion_year = 2018
fig, ax = plt.subplots()
ax.hist(tmp_traces["pga"], bins=30, ec="black", label="train")
year_filter = tmp_traces["year"] == validataion_year
ax.hist(tmp_traces[year_filter]["pga"], bins=30, ec="black", label="validation")
pga_threshold = np.log10([1e-5, 0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0])
label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
ax.vlines(pga_threshold[1:-1], 0, 17700, linestyles="dotted", color="k")
for i in range(len(pga_threshold) - 1):
    ax.text((pga_threshold[i] + pga_threshold[i + 1]) / 2, 15000, label[i])
ax.set_yscale("log")
ax.set_xlabel("log (PGA m/s2)")
ax.set_ylabel("number of traces")
ax.set_title(f"2012-2020 TSIMP no station location: validate on {validataion_year}")
ax.legend(loc="center right")

# training data (add oversampling)
# new_data=multiple_station_dataset_new("D:/TEAM_TSMIP/data/TSMIP_new.hdf5",mode="train",mask_waveform_sec=3,
#                                                 oversample_by_labels=True,dowmsampling=True,oversample=1,label_key="pgv",test_year=2016)
origin_data = multiple_station_dataset(
    "D:/TEAM_TSMIP/data/TSMIP_1999_2019.hdf5",
    mode="train",
    mask_waveform_sec=3,
    weight_label=False,
    oversample=1,
    oversample_mag=5,
    test_year=2016,
    mask_waveform_random=True,
    mag_threshold=5,
    label_key="pga",
    input_type="acc",
    data_length_sec=15,
    part_small_event=True
)
new_data = multiple_station_dataset(
    "D:/TEAM_TSMIP/data/TSMIP_1999_2019.hdf5",
    mode="train",
    mask_waveform_sec=3,
    weight_label=False,
    oversample=1.5,
    oversample_mag=5,
    test_year=2016,
    mask_waveform_random=True,
    mag_threshold=5,
    label_key="pga",
    input_type="acc",
    data_length_sec=15,
    part_small_event=True
)
# oversample_data=multiple_station_dataset_new("D:/TEAM_TSMIP/data/TSMIP.hdf5",train_mode=True,mask_waveform_sec=5,oversample=1.5,oversample_mag=5)
# pre1 = pd.read_csv(
#     f"./predict/model 37 3 sec prediction.csv"
# )

# train_set_size = int(len(new_data) * 0.8)
# valid_set_size = len(new_data) - train_set_size
# indice = np.arange(len(new_data))
# np.random.seed(0)
# np.random.shuffle(indice)
# train_indice, test_indice = np.array_split(indice, [train_set_size])
# train_dataset = CustomSubset(new_data, train_indice)
# val_dataset = CustomSubset(new_data, test_indice)

# train_sampler = WeightedRandomSampler(
#     weights=train_dataset.weight, num_samples=len(train_dataset), replacement=True
# )
# train_loader = DataLoader(
#     dataset=train_dataset,
#     batch_size=16,
#     sampler=train_sampler,
#     shuffle=False,
#     drop_last=True,
# )
# val_loader=DataLoader(dataset=val_dataset,batch_size=16,
#                                 shuffle=False,drop_last=True)

origin_loader = DataLoader(origin_data, batch_size=16, shuffle=False, drop_last=True)
new_loader=DataLoader(dataset=new_data,batch_size=16, shuffle=False, drop_last=True)
# new_loader1=DataLoader(dataset=new_data1,batch_size=16)


origin_PGA = []
for sample in tqdm(origin_loader):
    tmp_pga = torch.index_select(
        sample["label"].flatten(), 0, sample["label"].flatten().nonzero().flatten()
    ).tolist()
    origin_PGA.extend(tmp_pga)
origin_PGA_array = np.array(origin_PGA)
high_intensity_rate = np.sum(origin_PGA_array > np.log10(0.250)) / len(origin_PGA_array)
print(f"origin rate:{high_intensity_rate}")

new_PGA = []
for sample in tqdm(new_loader):
    tmp_pga = torch.index_select(
        sample["label"].flatten(), 0, sample["label"].flatten().nonzero().flatten()
    ).tolist()
    new_PGA.extend(tmp_pga)
new_PGA_array = np.array(new_PGA)
oversampled_high_intensity_rate = np.sum(new_PGA_array > np.log10(0.250)) / len(
    new_PGA_array
)
print(f"oversampled rate:{oversampled_high_intensity_rate}")

# new_PGA1=[]
# for sample in tqdm(new_loader1):
#     tmp_pga=torch.index_select(sample[3].flatten(),
#                                 0,
#                                 sample[3].flatten().nonzero().flatten()).tolist()
#     new_PGA1.extend(tmp_pga)
# new_PGA1_array=np.array(new_PGA1)
# magnitude_threshold_high_intensity_rate=np.sum(new_PGA1_array>np.log10(0.25))/len(new_PGA1_array)
# print(f"magnitude threshold rate:{magnitude_threshold_high_intensity_rate}")

label = ["2", "3", "4", "5-", "5+", "6-", "6+", "7"]
pga_threshold = np.log10(
    [0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0,10])
# label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
# pgv_threshold = np.log10(
#     [0.00001, 0.002, 0.007, 0.019, 0.057, 0.15, 0.3, 0.5, 0.8, 1.4, 20]
# )

fig, ax = plt.subplots(figsize=(7, 7))
ax.hist(new_PGA, bins=32, edgecolor="k", label="oversampled train data", alpha=0.6)
ax.hist(origin_PGA, bins=32, edgecolor="k", label="train data", alpha=0.6)
# ax.hist(new_PGA1,bins=32,edgecolor="k",label="mag>=4.5")
# ax.hist(pre1["answer"], bins=28, edgecolor="k", label="2016 data")
ax.vlines(pga_threshold[1:-1], 0, 40000, linestyles="dotted", color="k")
for i in range(len(pga_threshold) - 1):
    ax.text((pga_threshold[i] + pga_threshold[i + 1]) / 2, 50000, label[i])
ax.set_ylabel("Number of traces",size=14)
ax.set_xlabel(r"log(PGA (${m/s^2}$))",size=14)
ax.set_title("TSMIP data PGA distribution",size=16)
ax.set_yscale("log")
fig.legend(loc="upper right")
# fig.savefig("PGA distribution.pdf", dpi=300,bbox_inches='tight')

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

# try oversample function

label = ["2", "3", "4", "5-", "5+", "6-", "6+", "7"]
pgv_threshold = np.log10([0.007, 0.019, 0.057, 0.15, 0.3, 0.5, 0.8, 1.4, 20])
origin_PGA_array = np.array(origin_PGA)
x = origin_PGA_array[origin_PGA_array > np.log10(0.019)]
y = 4.5 ** (1.5**x) + 1
y = np.round(y, 0)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(x, y)
ax.vlines(pgv_threshold[1:-1], 0, 10, linestyles="dotted", color="k")
for i in range(len(pgv_threshold) - 1):
    ax.text((pgv_threshold[i] + pgv_threshold[i + 1]) / 2, 10, label[i])
# ax.set_ylim(0,10)

label = origin_data.labels.flatten()


class pgv_intensity_classifier:
    def __init__(self):
        self.threshold = np.log10([0.002, 0.007, 0.019, 0.057, 0.15, 0.5, 1.4, 20])
        self.label = [0, 1, 2, 3, 4, 5, 6, 7]

    def classify(self, input_array):
        output_array = np.zeros_like(input_array)
        for i in range(1, len(input_array)):
            if input_array[i] < self.threshold[0]:
                output_array[i] = self.label[0]
            elif input_array[i] < self.threshold[1]:
                output_array[i] = self.label[1]
            elif input_array[i] < self.threshold[2]:
                output_array[i] = self.label[2]
            elif input_array[i] < self.threshold[3]:
                output_array[i] = self.label[3]
            elif input_array[i] < self.threshold[4]:
                output_array[i] = self.label[4]
            elif input_array[i] < self.threshold[5]:
                output_array[i] = self.label[5]
            elif input_array[i] < self.threshold[6]:
                output_array[i] = self.label[6]
            elif input_array[i] < self.threshold[7]:
                output_array[i] = self.label[7]
        return output_array


pgv_classifier = pgv_intensity_classifier()

output_array = pgv_classifier.classify(label)

label_class, counts = np.unique(output_array, return_counts=True)


samples_weight = torch.as_tensor(
    [1 / counts[int(i)] for i in output_array], dtype=torch.double
)

from torch.utils.data import WeightedRandomSampler

sampler = WeightedRandomSampler(
    weights=samples_weight, num_samples=len(output_array), replacement=True
)
