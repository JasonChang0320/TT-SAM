import h5py
import numpy as np
import pandas as pd

from plot_predict_map import plot_pga_map, warning_map

mask_after_sec=3
EQ_ID=27305
trigger_station_threshold=1
path="./predict/random sec updated dataset and new data generator/ok model prediction"
Afile_path="data/Afile"

catalog=pd.read_csv(f"{Afile_path}/final catalog (station exist)_1.csv")
traces_info=pd.read_csv(f"{Afile_path}/1991-2020 traces with picking and label_new (sta location exist)_1.csv")
prediction=pd.read_csv(f"{path}/model2 7 9 {mask_after_sec} sec {trigger_station_threshold} triggered station prediction.csv")
station_info=pd.read_csv("data/station information/TSMIPstations_new.csv")

data_path="D:/TEAM_TSMIP/data/TSMIP_new.hdf5"
dataset = h5py.File(data_path, 'r')
station_name=dataset['data'][str(EQ_ID)]["station_name"][:].tolist()
station_df= pd.DataFrame(data={'station_name' : station_name})
station_name_list = station_df['station_name'].str.decode("utf-8").to_list()


# trace_info_with_sta_locat=pd.merge(traces_info[traces_info["EQ_ID"]==EQ_ID],station_info,
#                         how="left",left_on=["station_name"],right_on=["location_code"])
event_prediction=prediction[prediction["EQ_ID"]==EQ_ID]
event_prediction.insert(4,"station_name",station_name_list)
event_prediction.insert(9,"epdis (km)",traces_info[traces_info["EQ_ID"]==EQ_ID]["epdis (km)"].tolist())

event=catalog[catalog["EQ_ID"]==EQ_ID]
event=event.assign(latitude=event["lat"]+event["lat_minute"]/60,
                    longitude=event["lon"]+event["lon_minute"]/60)
dataset.close()
plot_pga_map(trace_info=event_prediction,eventmeta=event,
            true_pga=event_prediction["answer"],pred_pga=event_prediction["predict"],
            sec=mask_after_sec,EQ_ID=EQ_ID,grid_method="linear",pad=100)

pga_threshold=np.log10(9.8*0.025)
warning_map(trace_info=event_prediction,eventmeta=event,EQ_ID=EQ_ID,sec=mask_after_sec,
                        pga_threshold=pga_threshold)
