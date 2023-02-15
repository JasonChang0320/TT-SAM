import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig,ax=plt.subplots()

from multiple_sta_dataset import multiple_station_dataset_new
from plot_predict_map import true_predicted

mask_after_sec=3
trigger_station_threshold=1
warning_magnitude_threshold=4
pga_threshold=np.log10(9.8*0.025)
sampling_rate=200
path="./predict/random sec updated dataset and new data generator/ok model prediction"

data=multiple_station_dataset_new("D:/TEAM_TSMIP/data/TSMIP_new.hdf5",mode="test",
                                    mask_waveform_sec=mask_after_sec,test_year=2018,
                                    trigger_station_threshold=trigger_station_threshold,mag_threshold=0)

catalog=data.event_metadata
prediction=pd.read_csv(f"{path}/model2 7 9 {mask_after_sec} sec {trigger_station_threshold} triggered station prediction.csv")
prediction=pd.merge(prediction,catalog,how="left",on="EQ_ID")

# true_predict_filter=((prediction["predict"]>pga_threshold) & ((prediction["answer"]>pga_threshold)))
warning_time_filter=(prediction["pga_time"]>1000+sampling_rate*mask_after_sec)
magnitude_filter=(prediction["magnitude"]>=warning_magnitude_threshold)

prediction_for_warning=prediction[warning_time_filter & magnitude_filter]

warning_time=(prediction_for_warning["pga_time"]-(1000+sampling_rate*mask_after_sec))/sampling_rate
prediction_for_warning.insert(5, "warning_time (sec)", warning_time)

true_predict_filter=((prediction_for_warning["predict"]>pga_threshold) & ((prediction_for_warning["answer"]>pga_threshold)))
positive_filter=(prediction_for_warning["predict"]>pga_threshold)
true_filter=(prediction_for_warning["answer"]>pga_threshold)

EQ_ID=27558
# 27305 28404 28437 28507 29418
eq_id_filter=(prediction_for_warning["EQ_ID"]==EQ_ID)

fig,ax=plt.subplots(figsize=(7,7))
ax.hist(prediction_for_warning[eq_id_filter & true_predict_filter]["warning_time (sec)"],bins=20,ec='black')
describe=prediction_for_warning[eq_id_filter & true_predict_filter]["warning_time (sec)"].describe()
count=int(describe["count"])
mean=np.round(describe["mean"],2)
std=np.round(describe["std"],2)
median=np.round(describe["50%"],2)
max=np.round(describe["max"],2)
precision=np.round(len(prediction_for_warning[eq_id_filter & true_predict_filter])/len(prediction_for_warning[eq_id_filter & positive_filter]),2)
recall=np.round(len(prediction_for_warning[eq_id_filter & true_predict_filter])/len(prediction_for_warning[eq_id_filter & true_filter]),2)
ax.set_title(f"Warning time in EQ ID: {EQ_ID}, \n after first triggered station {mask_after_sec} sec",fontsize=18)
ax.set_xlabel("Warning time (sec)",fontsize=15)
ax.set_ylabel("Number of stations",fontsize=15)
ax.text(0.45, .7,
        f"mean: {mean} s\nstd: {std} s\nmedian: {median}s\nmax: {max} s\neffective warning stations: {count}\nprecision: {precision}\nrecall: {recall}", 
        transform=ax.transAxes,fontsize=14)
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
# fig.savefig(f"{path}/warning time/warning time {mask_after_sec} sec mag_threshold 4.5.png",dpi=150)



# fig,ax=true_predicted(y_true=prediction_for_warning[eq_id_filter & true_predict_filter]["answer"],y_pred=prediction_for_warning[eq_id_filter & true_predict_filter]["predict"],
#                     time=mask_after_sec,quantile=False,agg="point", point_size=70)

#plot effective warning station prediciton
fig,ax=true_predicted(y_true=prediction[prediction["EQ_ID"]==EQ_ID]["answer"],y_pred=prediction[prediction["EQ_ID"]==EQ_ID]["predict"],
                    time=mask_after_sec,quantile=False,agg="point", point_size=70)

ax.scatter(prediction_for_warning[eq_id_filter & true_predict_filter]["answer"],
        prediction_for_warning[eq_id_filter & true_predict_filter]["predict"],s=70,c="red")