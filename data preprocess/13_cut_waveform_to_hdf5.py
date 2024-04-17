import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from read_tsmip import cut_traces
start_year = 1999
end_year = 2019
sta_path = "../data/station_information"
waveform_path = "../data/waveform"
catalog = pd.read_csv(
    f"./events_traces_catalog/{start_year}_{end_year}_final_catalog.csv"
)
traces = pd.read_csv(
    f"./events_traces_catalog/{start_year}_{end_year}_final_traces_Vs30.csv"
)
station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv") 
traces.loc[traces.index, "p_pick_sec"] = pd.to_timedelta(
    traces["p_pick_sec"], unit="sec"
)
traces.loc[traces.index, "p_arrival_abs_time"] = pd.to_datetime(
    traces["p_arrival_abs_time"], format="%Y-%m-%d %H:%M:%S"
)

# into hdf5
output = f"../data/TSMIP_{start_year}_{end_year}_Vs30.hdf5"
error_event = {"EQ_ID": [], "reason": []}
with h5py.File(output, "w") as file:
    data = file.create_group("data")
    meta = file.create_group("metadata")
    for eq_id in tqdm(catalog["EQ_ID"]):
        # for eq_id in [247]:
        try:
            tmp_traces, traces_info = cut_traces(
                traces, eq_id, waveform_path, waveform_type="acc"
            )
            _, vel_info = cut_traces(traces, eq_id, waveform_path, waveform_type="vel")
            _, dis_info = cut_traces(traces, eq_id, waveform_path, waveform_type="dis")
            traces_info["vel"] = vel_info["traces"]
            traces_info["dis"] = dis_info["traces"]
            # fig=plot_cutting_event(tmp_traces,traces_info)
            start_time_str_arr = np.array(traces_info["start_time"], dtype="S30")
            station_name_str_arr = np.array(tmp_traces["station_name"], dtype="S30")
            tmp_station_info = pd.merge(
                tmp_traces[["station_name","Vs30"]],
                station_info[
                    ["location_code", "latitude", "longitude", "elevation (m)"]
                ],
                how="left",
                left_on="station_name",
                right_on="location_code",
            )
            location_array = np.array(
                tmp_station_info[["latitude", "longitude", "elevation (m)"]]
            )
            Vs30_array=np.array(tmp_traces["Vs30"])
            if np.isnan(location_array).any():
                print("The location array contain NaN values")
                continue
            event = data.create_group(f"{eq_id}")
            event.create_dataset(
                "acc_traces", data=traces_info["traces"], dtype=np.float64
            )
            event.create_dataset(
                "vel_traces", data=traces_info["vel"], dtype=np.float64
            )
            event.create_dataset(
                "dis_traces", data=traces_info["dis"], dtype=np.float64
            )
            event.create_dataset("p_picks", data=traces_info["p_picks"], dtype=np.int64)
            event.create_dataset("pga", data=traces_info["pga"], dtype=np.float64)
            event.create_dataset("pgv", data=traces_info["pgv"], dtype=np.float64)
            event.create_dataset(
                "start_time", data=start_time_str_arr, maxshape=(None), chunks=True
            )
            event.create_dataset(
                "pga_time", data=traces_info["pga_time"], dtype=np.int64
            )
            event.create_dataset(
                "pgv_time", data=traces_info["pgv_time"], dtype=np.int64
            )
            event.create_dataset(
                "station_name", data=station_name_str_arr, maxshape=(None), chunks=True
            )
            event.create_dataset(
                "station_location", data=location_array, dtype=np.float64
            )
            event.create_dataset(
                "Vs30", data=Vs30_array, dtype=np.float64
            )
        except Exception as reason:
            print(f"EQ_ID:{eq_id}, {reason}")
            error_event["EQ_ID"].append(eq_id)
            error_event["reason"].append(reason)
            continue
        # fig.savefig(f"data/cutting waveform image/{eq_id}.png")
    error_event_df = pd.DataFrame(error_event)
    error_event_df.to_csv(
        "./events_traces_catalog/load into hdf5 error event.csv", index=False
    )

catalog.to_hdf(output, key="metadata/event_metadata", mode="a", format="table")
traces.to_hdf(output, key="metadata/traces_metadata", mode="a", format="table")
