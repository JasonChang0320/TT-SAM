import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from read_tsmip import read_tsmip, get_peak_value


target_sampling_rate = 200
waveform_path = "../data/waveform"
output_path = "./traces_sampling_rate"
traces = pd.read_csv(f"events_traces_catalog/1999_2019_final_traces_Vs30.csv")
traces["p_arrival_abs_time"] = pd.to_datetime(
    traces["p_arrival_abs_time"], format="%Y-%m-%d %H:%M:%S"
)

dict = {"sampling_rate": [], "origin_PGA": [], "resampled_PGA": []}
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
        dict["sampling_rate"].append(sampling_rate)
        dict["resampled_PGA"].append(10 ** traces["pga"][i] * 100)

        pick_point = int(np.round(traces["p_pick_sec"][i] * sampling_rate, 0))
        waveform.detrend(type="demean")
        waveform.filter("lowpass", freq=10)
        origin_pga = 10 ** get_peak_value(waveform, pick_point=pick_point)[0] * 100
        dict["origin_PGA"].append(origin_pga)


output = pd.DataFrame(dict)
output.to_csv(f"{output_path}/statistic_sampling_rate.csv", index=False)


output["residual"] = output["origin_PGA"] - output["resampled_PGA"]
fig, ax = plt.subplots()
ax.hist(output["residual"], bins=20, edgecolor="gray")
ax.set_yscale("log")
ax.set_xlabel("Residual (pga-resampled pga, unit: gal)")
ax.set_ylabel("Number of traces")
# fig.savefig(f"{output_path}/pga residual after resampling.png",dpi=300)
