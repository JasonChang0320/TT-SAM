import h5py
import matplotlib.pyplot as plt
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
p_picks=dataset['data'][str(EQ_ID)]["p_picks"][:].tolist()
pga_time=dataset['data'][str(EQ_ID)]["pga_time"][:].tolist()
station_name=dataset['data'][str(EQ_ID)]["station_name"][:].tolist()
latitude=dataset['data'][str(EQ_ID)]["station_location"][:,0].tolist()
longitude=dataset['data'][str(EQ_ID)]["station_location"][:,1].tolist()
elevation=dataset['data'][str(EQ_ID)]["station_location"][:,2].tolist()
pga=dataset['data'][str(EQ_ID)]["pga"][:].tolist()

station_df=pd.DataFrame({"p_picks":p_picks,"pga":pga,"pga_time":pga_time,"station_name":station_name,
    "latitude":latitude,"longitude":longitude,"elevation":elevation})
station_df['station_name']=station_df['station_name'].str.decode("utf-8")
station_df=station_df.drop_duplicates(subset=["p_picks","pga_time","station_name"],keep="last")

event_prediction=prediction[prediction["EQ_ID"]==EQ_ID]
event_prediction=event_prediction.drop_duplicates(subset=["p_picks","pga_time","predict"],keep="last")

event_prediction=pd.merge(event_prediction,station_df[["p_picks","pga_time","station_name"]],how="left",on=["p_picks","pga_time"])
single_event_traces_info=traces_info[traces_info["EQ_ID"]==EQ_ID].drop_duplicates(subset=["station_name"])
event_prediction=pd.merge(event_prediction,single_event_traces_info[["station_name","epdis (km)"]],
                        how="left",on=["station_name"])

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

fig,ax= plt.subplots()

true_warning_prediction=event_prediction.query(f"predict > {pga_threshold} and answer > {pga_threshold}")
pga_time=true_warning_prediction["pga_time"]/200-5
pick_time=true_warning_prediction["p_picks"]/200-5
ax.scatter(true_warning_prediction["epdis (km)"],pga_time,label="pga_time")
ax.scatter(true_warning_prediction["epdis (km)"],pick_time,label="P arrival")
ax.axhline(y=mask_after_sec,xmax=true_warning_prediction["epdis (km)"].max()+10,linestyle="dashed",c="r",label="warning")
ax.legend()
for index in true_warning_prediction["epdis (km)"].index:
    distance=[true_warning_prediction["epdis (km)"][index],
                true_warning_prediction["epdis (km)"][index]]
    time=[pga_time[index],pick_time[index]]
    ax.plot(distance,time,c="grey")
ax.set_title(f"EQ ID: {EQ_ID} Warning time")
ax.set_xlabel("epicentral distance (km)")
ax.set_ylabel("time (sec)")