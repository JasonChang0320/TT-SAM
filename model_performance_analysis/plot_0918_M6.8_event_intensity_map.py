import pandas as pd
from analysis import Intensity_Plotter

predict_path = "../predict/station_blind_Vs30_bias2closed_station_2016/0918_M6.8_1319_1330"
catalog_path="../data_preprocess/0918_M6.8_1319_1330"
mask_sec = 10
catalog = pd.read_csv(f"{catalog_path}/event_catalog.csv")
prediction = pd.read_csv(
    f"{predict_path}/{mask_sec} sec model11 eqid_30792 prediction with all info.csv"
)
catalog["longitude"]=catalog["lon"]+catalog["lon_minute"]/60
catalog["latitude"]=catalog["lat"]+catalog["lat_minute"]/60
fig, ax = Intensity_Plotter.plot_intensity_map(
    trace_info=prediction,
    eventmeta=catalog,
    label_type="pga",
    true_label=prediction["answer"],
    pred_label=prediction["predict"],
    sec=mask_sec,
    EQ_ID=None,
    grid_method="linear",
    pad=100,
    title=f"{mask_sec} sec intensity Map",
)
# fig.savefig(f"{predict_path}/{mask_sec} sec intensity map.png",dpi=300)
