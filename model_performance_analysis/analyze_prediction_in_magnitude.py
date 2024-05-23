import pandas as pd
import sys

from analysis import Intensity_Plotter

path="../predict/station_blind_Vs30_bias2closed_station_2016"
mask_after_sec=7
prediction_with_info=pd.read_csv(f"{path}/{mask_after_sec} sec model11 with all info.csv")
# ===========plot mag>=5.5===========
mag5_5_prediction = prediction_with_info.query("magnitude>=5.5")
label_type = "pga"
fig, ax = Intensity_Plotter.plot_true_predicted(
    y_true=mag5_5_prediction["answer"],
    y_pred=mag5_5_prediction["predict"],
    quantile=False,
    agg="point",
    point_size=70,
    target=label_type,
    title=f"Magnitude>=5.5 event {mask_after_sec} sec",
)

# ===========check prediction in magnitude===========

label = "pga"
fig, ax = Intensity_Plotter.plot_true_predicted(
    y_true=prediction_with_info["answer"][prediction_with_info["magnitude"] >= 5],
    y_pred=prediction_with_info["predict"][prediction_with_info["magnitude"] >= 5],
    quantile=False,
    agg="point",
    point_size=20,
    target=label,
)

ax.scatter(
    prediction_with_info["answer"][prediction_with_info["magnitude"] < 5],
    prediction_with_info["predict"][prediction_with_info["magnitude"] < 5],
    c="r",
    label="magnitude < 5",
)