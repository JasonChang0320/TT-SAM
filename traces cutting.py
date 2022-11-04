from read_tsmip import *
import os
import pandas as pd
import numpy as np
import h5py

Afile_path="data/Afile"
catalog=pd.read_csv(f"{Afile_path}/final catalog.csv")
traces=pd.read_csv(f"{Afile_path}/2012-2020 traces with picking and label_new.csv")
station_info=pd.read_csv(f"data/TSMIPstations.csv")
station_code=station_info["station"].str.extract(r'(.*?)[(]')
location_code=station_info["station"].str.extract(r'[(](.*?)[)]')
station_info.insert(1,"station_code",station_code.values)
station_info.insert(2,"location_code",location_code.values)
station_info.drop(["station"],axis=1,inplace=True)
station_info.to_csv(f"data/TSMIPstations_new.csv",index=False)

traces.loc[traces.index,"p_picks (sec)"]=pd.to_timedelta(traces["p_picks (sec)"],unit="sec")
traces.loc[traces.index,"start_time"]=pd.to_datetime(traces['start_time'], format='%Y%m%d%H%M%S')

eq_id=2561
eq_id=7613
eq_id=4682
eq_id=423

output="data/TSMIP.hdf5"
with h5py.File(output, 'w') as file:
        data = file.create_group('data')
        meta = file.create_group('metadata')
        for eq_id in catalog["EQ_ID"][:20]:
            tmp_traces,traces_info=cut_traces(traces,eq_id)
            fig=plot_cutting_event(tmp_traces,traces_info,close_fig=True)
            start_time_str_arr = np.array(traces_info["start_time"],dtype='S30')
            event = data.create_group(f"{eq_id}")
            event.create_dataset("traces", data=traces_info["traces"],dtype=np.float64)
            event.create_dataset("p_picks", data=traces_info["p_picks"],dtype=np.int64)
            event.create_dataset("pga",data=traces_info["pga"],dtype=np.float64)
            event.create_dataset("pgv",data=traces_info["pgv"],dtype=np.float64)
            event.create_dataset("start_time", data=start_time_str_arr,maxshape=(None),chunks=True)
            event.create_dataset("pga_time", data=traces_info["pga_time"],dtype=np.int64)
            event.create_dataset("pgv_time", data=traces_info["pgv_time"],dtype=np.int64)
            # fig.savefig(f"data/cutting waveform image/{eq_id}.png")


catalog.to_hdf(output, key='metadata/event_metadata', mode='a', format='table')
traces.to_hdf(output, key='metadata/traces_metadata', mode='a', format='table')

