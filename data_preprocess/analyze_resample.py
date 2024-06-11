import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate,PchipInterpolator
from read_tsmip import read_tsmip, get_peak_value

"""
In our dataset, we have different sampling rate waveforms, most of the data is 200Hz.
In this script, we checked the residual of PGA after resampling all of waveforms to 200Hz.
"""

target_sampling_rate = 200
waveform_path = "../data/waveform"
output_path = "./traces_sampling_rate"
traces = pd.read_csv(f"events_traces_catalog/1999_2019_final_traces_Vs30.csv")
traces["p_arrival_abs_time"] = pd.to_datetime(
    traces["p_arrival_abs_time"], format="%Y-%m-%d %H:%M:%S"
)

dict = {"station_name":[],"sta_latitude":[],"sta_longitude":[],"sampling_rate": [], "origin_PGA": [], "resampled_PGA": []}
for i in range(len(traces)):
    print(f"{i}/{len(traces)}")
    year = str(traces["year"][i])
    month = str(traces["month"][i])
    if len(month) < 2:
        month = "0" + month
    filename = traces["file_name"][i].strip()
    waveform = read_tsmip(f"{waveform_path}/{year}/{month}/{filename}.txt")
    sampling_rate = waveform[0].stats.sampling_rate
    if sampling_rate != target_sampling_rate:
        dict["station_name"].append(traces["station_name"][i])
        dict["sta_latitude"].append(traces["latitude"][i])
        dict["sta_longitude"].append(traces["longitude"][i])
        dict["sampling_rate"].append(sampling_rate)
        

        pick_point = int(np.round(traces["p_pick_sec"][i] * sampling_rate, 0))
        waveform.detrend(type="demean")
        waveform.filter("lowpass", freq=10)
        origin_pga = 10 ** get_peak_value(waveform, pick_point=pick_point)[0] * 100
        dict["origin_PGA"].append(origin_pga)


        for channel in range(len(waveform)):
            print(max(waveform[channel].data))
            duration=len(waveform[channel].data)/sampling_rate
            origin_x=np.linspace(0,duration,int(len(waveform[channel].data)))
            resample_x=np.linspace(0,duration,int(target_sampling_rate*duration))
            interpolater= PchipInterpolator(origin_x, waveform[channel].data)
            resample_waveform = interpolater(resample_x)
            
            # fig,ax=plt.subplots(2,1)
            # ax[0].plot(origin_x, waveform[channel].data)
            # ax[0].axvline(traces["p_pick_sec"][i],c="r")
            # ax[1].plot(resample_x, resample_waveform)
            # ax[1].axvline(traces["p_pick_sec"][i],c="r")
            waveform[channel].data=resample_waveform
            waveform[channel].stats.sampling_rate=target_sampling_rate
            print(max(waveform[channel].data))

        pick_point = int(np.round(traces["p_pick_sec"][i] * target_sampling_rate, 0))
        resample_pga = 10 ** get_peak_value(waveform, pick_point=pick_point)[0] * 100
        dict["resampled_PGA"].append(resample_pga)
    break


output = pd.DataFrame(dict)
output["residual"] = output["origin_PGA"] - output["resampled_PGA"]
# output.to_csv(f"{output_path}/statistic_sampling_rate_new.csv", index=False)



fig, ax = plt.subplots()
ax.hist(output["residual"], bins=20, edgecolor="gray")
ax.set_yscale("log")
ax.set_xlabel("Residual (pga-resampled pga, unit: gal)")
ax.set_ylabel("Number of traces")
# fig.savefig(f"{output_path}/pga residual after resampling.png",dpi=300)
