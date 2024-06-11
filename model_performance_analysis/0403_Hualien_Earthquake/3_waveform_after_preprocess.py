import pandas as pd
import os
import obspy
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append("../..")
from data_preprocess.read_tsmip import get_peak_value

data_path = "./0403asc_by_Joey"

files = os.listdir(f"{data_path}")
asc_files = [file for file in files if file.endswith(".asc")]
output_df = {"station_code": [], "PGA": []}
for i in range(len(asc_files)):
    data = pd.read_csv(
        f"{data_path}/{asc_files[i]}", sep="\s+", skiprows=[0], header=None
    ).to_numpy()

    stream = obspy.core.stream.Stream()
    channel = ["HLZ", "HLN", "HLE"]

    for j, chan in enumerate(channel):
        trace = obspy.core.trace.Trace(data[:, j + 1])
        trace.stats.sampling_rate = 100
        # trace.stats.starttime = obspy.UTCDateTime(asc_files[0][:17])
        stream.append(trace)
    stream.filter("lowpass", freq=10)
    # plot
    # fig,ax=plt.subplots(3,1)
    # for k in range(3):
    #     ax[k].plot(stream[k].data)
    # ax[0].set_title(asc_files[i][26:30])
    # ax[2].set_xlabel("time sample (100Hz)")
    # ax[1].set_ylabel("amplitude (gal)")
    # plt.close()
    # fig.savefig(f"{data_path}/image/{asc_files[i][26:30]}.png",dpi=300)

    pga, _ = get_peak_value(stream)
    output_df["station_code"].append(asc_files[i][26:30])
    output_df["PGA"].append(pga)

output_df = pd.DataFrame(output_df)

station_info = pd.read_csv("../../data/station_information/TSMIPstations_new.csv")

output_df = pd.merge(
    output_df,
    station_info[["station_code", "location_code"]],
    left_on="station_code",
    right_on="station_code",
    how="left",
)

# output_df.to_csv(f"true_answer.csv", index=False)
