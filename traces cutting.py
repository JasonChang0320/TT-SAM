from read_tsmip import *
import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

Afile_path="data/Afile"
sta_path="data/station information"
catalog=pd.read_csv(f"{Afile_path}/final catalog (station exist).csv")
traces=pd.read_csv(f"{Afile_path}/2012-2020 traces with picking and label_new (sta location exist).csv")
station_info=pd.read_csv(f"{sta_path}/TSMIPstations_new.csv")
traces.loc[traces.index,"p_picks (sec)"]=pd.to_timedelta(traces["p_picks (sec)"],unit="sec")
traces.loc[traces.index,"start_time"]=pd.to_datetime(traces['start_time'], format='%Y%m%d%H%M%S')

eq_id=2561
eq_id=7613
eq_id=4682
eq_id=423

output="data/TSMIP.hdf5"
error_event={"EQ_ID":[],"reason":[]}
with h5py.File(output, 'w') as file:
        data = file.create_group('data')
        meta = file.create_group('metadata')
        for eq_id in tqdm(catalog["EQ_ID"]):
        # for eq_id in [247]:
            try:
                tmp_traces,traces_info=cut_traces(traces,eq_id)
                # fig=plot_cutting_event(tmp_traces,traces_info)
                start_time_str_arr = np.array(traces_info["start_time"],dtype='S30')
                station_name_str_arr=np.array(tmp_traces["station_name"],dtype='S30')
                tmp_station_info=pd.merge(tmp_traces["station_name"],
                                            station_info[["location_code","latitude","longitude","elevation (m)"]],
                                            how="left",left_on="station_name",right_on="location_code")
                location_array=np.array(tmp_station_info[["latitude","longitude","elevation (m)"]])

                event = data.create_group(f"{eq_id}")
                event.create_dataset("traces", data=traces_info["traces"],dtype=np.float64)
                event.create_dataset("p_picks", data=traces_info["p_picks"],dtype=np.int64)
                event.create_dataset("pga",data=traces_info["pga"],dtype=np.float64)
                event.create_dataset("pgv",data=traces_info["pgv"],dtype=np.float64)
                event.create_dataset("start_time", data=start_time_str_arr,maxshape=(None),chunks=True)
                event.create_dataset("pga_time", data=traces_info["pga_time"],dtype=np.int64)
                event.create_dataset("pgv_time", data=traces_info["pgv_time"],dtype=np.int64)
                event.create_dataset("station_name",data=station_name_str_arr,maxshape=(None),chunks=True)
                event.create_dataset("station_location", data=location_array,dtype=np.float64)
            except Exception as reason:
                print(f"EQ_ID:{eq_id}, {reason}")
                error_event["EQ_ID"].append(eq_id)
                error_event["reason"].append(reason)
                continue
            # fig.savefig(f"data/cutting waveform image/{eq_id}.png")
        error_event_df=pd.DataFrame(error_event)
        error_event_df.to_csv("data/load into hdf5 error event.csv",index=False)

catalog.to_hdf(output, key='metadata/event_metadata', mode='a', format='table')
traces.to_hdf(output, key='metadata/traces_metadata', mode='a', format='table')
