import matplotlib.pyplot as plt

fig,ax=plt.subplots()
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from multiple_sta_dataset import (multiple_station_dataset,
                                  multiple_station_dataset_new)

# from wordcloud import WordCloud


#events histogram
Afile_path="data/Afile"
origin_catalog=pd.read_csv(f"{Afile_path}/final catalog.csv")
# catalog=pd.read_csv(f"{Afile_path}/final catalog (station exist).csv")

validation_year=2018
fig,ax=plt.subplots()
ax.hist(origin_catalog["magnitude"],bins=30,ec='black',label="train")
# year_filter=(origin_catalog["year"]==validation_year)
# ax.hist(origin_catalog[year_filter]["magnitude"],bins=30,ec='black',label="validation")
ax.set_yscale("log")
ax.set_xlabel("Magnitude")
ax.set_ylabel("number of events")
ax.set_title(f"1991-2020 TSIMP data: validate on {validation_year}")
ax.set_title(f"1991-2020 TSIMP event catalog")
ax.legend()

#traces pga histogram
# traces_catalog=pd.read_csv(f"{Afile_path}/2012-2020 traces with picking and label_new (sta location exist).csv")
traces_catalog=pd.read_csv(f"{Afile_path}/1991-2020 traces with picking and label_new.csv")
merged_catalog=pd.merge(traces_catalog,origin_catalog,how="left",on="EQ_ID")
oldtime_catalog=merged_catalog[(merged_catalog["year_y"]>=1999) & (merged_catalog["year_y"]<=2020) & 
                                (merged_catalog["magnitude"]>=5.5)]
newtime_catalog=merged_catalog[(merged_catalog["year_y"]>=2018) & (merged_catalog["year_y"]<=2020) & 
                                (merged_catalog["magnitude"]>=4.5) & (merged_catalog["magnitude"]<5.5)]
now_catalog=merged_catalog[(merged_catalog["year_y"]>=2012) & (merged_catalog["year_y"]<=2020)]
concat_catalog=pd.concat([oldtime_catalog,newtime_catalog])
# validation_year=2018
PGA=np.sqrt(now_catalog["pga_z"]**2+
            now_catalog["pga_ns"]**2+
            now_catalog["pga_ew"]**2)
PGA_filtered=np.sqrt(concat_catalog["pga_z"]**2+
            concat_catalog["pga_ns"]**2+
            concat_catalog["pga_ew"]**2)
fig,ax=plt.subplots()
ax.hist(np.log10(PGA/100),bins=80,ec='black')
ax.hist(np.log10(PGA_filtered/100),bins=80,ec='black',alpha=0.5)
# ax.hist(traces_catalog["pga"],bins=30,ec='black',label="train")
# year_filter=(traces_catalog["year"]==validation_year)
# ax.hist(traces_catalog[year_filter]["pga"],bins=30,ec='black',label="validation")
pga_threshold = np.log10(
    [1e-5, 0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0,10])
# pga_threshold = [100*1e-5, 100*0.008, 100*0.025, 100*0.080, 100*0.250,
#                 100*0.80, 100*1.4, 100*2.5, 100*4.4, 100*8.0,100*10]
label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
ax.vlines(pga_threshold[1:-1],0,17700,linestyles='dotted',color="k")
for i in range(len(label)):
    ax.text((pga_threshold[i]+pga_threshold[i+1])/2,15000,label[i])
ax.set_yscale("log")
# ax.set_xlabel("log (PGA m/s2)")
ax.set_xlabel("PGA (gal)")
ax.set_ylabel("number of traces")
ax.set_title(f"TSMIP traces")
# ax.set_title(f"2012-2020 TSIMP data: validate on {validation_year}")
# ax.legend(loc="upper left")                


#station don't have location
sta_path="data/station information"
Afile_path="data/Afile"
traces=pd.read_csv(f"{Afile_path}/2012-2020 traces with picking and label_new.csv")
station_info=pd.read_csv(f"{sta_path}/TSMIPstations_new.csv")
sta_filter=traces["station_name"].isin(station_info["location_code"])
tmp_traces=traces[~sta_filter]
plt.bar(tmp_traces["station_name"].value_counts()[:5].index,tmp_traces["station_name"].value_counts().values[:5])

wordcloud = WordCloud(width=1600, height=800).generate(tmp_traces["station_name"].to_string())
plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

validataion_year=2018
fig,ax=plt.subplots()
ax.hist(tmp_traces["pga"],bins=30,ec='black',label="train")
year_filter=(tmp_traces["year"]==validataion_year)
ax.hist(tmp_traces[year_filter]["pga"],bins=30,ec='black',label="validation")
pga_threshold = np.log10(
    [1e-5, 0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0])
label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
ax.vlines(pga_threshold[1:-1],0,17700,linestyles='dotted',color="k")
for i in range(len(pga_threshold)-1):
    ax.text((pga_threshold[i]+pga_threshold[i+1])/2,15000,label[i])
ax.set_yscale("log")
ax.set_xlabel("log (PGA m/s2)")
ax.set_ylabel("number of traces")
ax.set_title(f"2012-2020 TSIMP no station location: validate on {validataion_year}")
ax.legend(loc="center right")

#training data (add oversampling)
origin_data=multiple_station_dataset_new("D:/TEAM_TSMIP/data/TSMIP_new.hdf5",mode="train",mask_waveform_sec=3,
                                                trigger_station_threshold=1,oversample=1,label_key="pgv")  
new_data=multiple_station_dataset_new("D:/TEAM_TSMIP/data/TSMIP_new.hdf5",mode="train",mask_waveform_sec=3,
                                                trigger_station_threshold=1,oversample=1.5,oversample_mag=4,label_key="pgv") 
# new_data1=multiple_station_dataset_new("D:/TEAM _TSMIP/data/TSMIP_new.hdf5",mode="train",mask_waveform_sec=3,
#                                         mag_threshold=4.5,oversample=1.5,oversample_mag=6)
# oversample_data=multiple_station_dataset_new("D:/TEAM_TSMIP/data/TSMIP.hdf5",train_mode=True,mask_waveform_sec=5,oversample=1.5,oversample_mag=5)
# pre1=pd.read_csv(f"./predict/model7 5 sec prediction.csv")


origin_loader=DataLoader(dataset=origin_data,batch_size=16)
new_loader=DataLoader(dataset=new_data,batch_size=16)
# new_loader1=DataLoader(dataset=new_data1,batch_size=16)


origin_PGA=[]
for sample in tqdm(origin_loader):
    tmp_pga=torch.index_select(sample[3].flatten(), 
                                0, 
                                sample[3].flatten().nonzero().flatten()).tolist()
    origin_PGA.extend(tmp_pga)
origin_PGA_array=np.array(origin_PGA)
high_intensity_rate=np.sum(origin_PGA_array>np.log10(0.25))/len(origin_PGA_array)
print(f"origin rate:{high_intensity_rate}")

new_PGA=[]
for sample in tqdm(new_loader):
    tmp_pga=torch.index_select(sample[3].flatten(), 
                                0, 
                                sample[3].flatten().nonzero().flatten()).tolist()
    new_PGA.extend(tmp_pga)
new_PGA_array=np.array(new_PGA)
oversampled_high_intensity_rate=np.sum(new_PGA_array>np.log10(0.25))/len(new_PGA_array)
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

# label = ["2", "3", "4", "5-", "5+", "6-", "6+", "7"]
# pga_threshold = np.log10(
#     [0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0,10])
label = ["2", "3", "4", "5-", "5+", "6-", "6+", "7"]
pgv_threshold = np.log10(
    [0.007,0.019,0.057,0.15,0.3,0.5,0.8,1.4,20])

fig,ax=plt.subplots(figsize=(7,7))
ax.hist(new_PGA,bins=32,edgecolor="k",color = "lightblue",label="oversampled train data")
ax.hist(origin_PGA,bins=32,edgecolor="k",label="origin train data")
# ax.hist(new_PGA1,bins=32,edgecolor="k",label="mag>=4.5")
ax.hist(pre1["answer"],bins=28,edgecolor="k",label="test data")
ax.vlines(pgv_threshold[1:-1],0,40000,linestyles='dotted',color="k")
for i in range(len(pgv_threshold)-1):
    ax.text((pgv_threshold[i]+pgv_threshold[i+1])/2,50000,label[i])
ax.set_ylabel("number of traces")
ax.set_xlabel("log(PGV (m/s))")
ax.set_title("TSMIP data PGV distribution")
ax.set_yscale("log")
fig.legend(loc='upper right')

# test label ditribution
pre1=pd.read_csv("predict/model1 2 3 sec 1 triggered station prediction.csv")

fig,ax=plt.subplots(figsize=(7,7))
ax.hist(pre1["answer"],bins=32,edgecolor="k",label="origin data")
ax.vlines(pgv_threshold[1:-1],0,17700,linestyles='dotted',color="k")
for i in range(len(pgv_threshold)-1):
    ax.text((pgv_threshold[i]+pgv_threshold[i+1])/2,22500,label[i])
ax.set_ylabel("number of trace")
ax.set_xlabel("log(PGA (m/s2))")
ax.set_title("TSMIP Test data PGA distribution")
ax.set_yscale("log")
fig.legend(loc='upper right')
