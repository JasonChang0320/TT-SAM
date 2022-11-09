import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wordcloud import WordCloud

#events histogram
Afile_path="data/Afile"
origin_catalog=pd.read_csv(f"{Afile_path}/2012-2020 catalog (no 2020_7-9).csv")
catalog=pd.read_csv(f"{Afile_path}/final catalog (station exist).csv")

validation_year=2018
fig,ax=plt.subplots()
ax.hist(catalog["magnitude"],bins=30,ec='black',label="train")
year_filter=(catalog["year"]==validation_year)
ax.hist(catalog[year_filter]["magnitude"],bins=30,ec='black',label="validation")
ax.set_yscale("log")
ax.set_xlabel("Magnitude")
ax.set_ylabel("number of events")
ax.set_title(f"2012-2020 TSIMP data: validate on {validation_year}")
ax.legend()

#traces pga histogram
traces_catalog=pd.read_csv(f"{Afile_path}/2012-2020 traces with picking and label_new (sta location exist).csv")
validation_year=2018
fig,ax=plt.subplots()
ax.hist(traces_catalog["pga"],bins=30,ec='black',label="train")
year_filter=(traces_catalog["year"]==validation_year)
ax.hist(traces_catalog[year_filter]["pga"],bins=30,ec='black',label="validation")
pga_threshold = np.log10(
    [1e-5, 0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0])
label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
ax.vlines(pga_threshold[1:-1],0,17700,linestyles='dotted',color="k")
for i in range(len(pga_threshold)-1):
    ax.text((pga_threshold[i]+pga_threshold[i+1])/2,15000,label[i])
ax.set_yscale("log")
ax.set_xlabel("log (PGA m/s2)")
ax.set_ylabel("number of traces")
ax.set_title(f"2012-2020 TSIMP data: validate on {validation_year}")
ax.legend(loc="upper left")


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

