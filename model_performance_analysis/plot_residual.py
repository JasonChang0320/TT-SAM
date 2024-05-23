import pandas as pd
import numpy as np
import os
from analysis import Residual_Plotter

mask_after_sec = 7
test_year = 2016
path = f"../predict/station_blind_Vs30_bias2closed_station_{test_year}"
output_path = f"{path}/{mask_after_sec} sec residual plots"
prediction_with_info = pd.read_csv(
    f"{path}/{mask_after_sec} sec model11 with all info.csv"
)

miss_alarm = (prediction_with_info["predict"] < np.log10(0.25)) & (
    prediction_with_info["answer"] >= np.log10(0.25)
)
false_alarm = (prediction_with_info["predict"] >= np.log10(0.25)) & (
    prediction_with_info["answer"] < np.log10(0.25)
)
wrong_predict = prediction_with_info[miss_alarm | false_alarm]

for column in prediction_with_info.columns:
    fig, ax = Residual_Plotter.residual_with_attribute(
        prediction_with_info=prediction_with_info,
        column=column,
        single_case_check=24784.0,
        wrong_predict=wrong_predict,
        test_year=test_year,
    )
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # fig.savefig(
    #     f"{output_path}/{column}.png",
    #     dpi=300,
    # )

# plot event residual on map
fig, ax = Residual_Plotter.single_event_residual_map(
    prediction_with_info=prediction_with_info,
    eq_id=24784.0,
    title=f"{mask_after_sec} sec 2016 Meinong earthquake residual in prediction",
)
# fig.savefig(
#     f"{output_path}/{mask_after_sec} sec 2016 Meinong earthquake residual map.png",
#     dpi=300,
# )

# plot all prediction residual on map
prediction_with_info["predict_residual"] = (
    prediction_with_info["predict"] - prediction_with_info["answer"]
)
grouby_sta = prediction_with_info.groupby("station_name").agg(
    {"longitude": "first", "latitude": "first", "predict_residual": ["mean", "std"]}
)
# 當station樣本只有1時，std 會有NaN之情形發生，需剃除
grouby_sta = grouby_sta[~grouby_sta["predict_residual", f"std"].isna()]
# grouby_sta.to_csv(f"{mask_after_sec}_sec_station_correction.csv")

max_abs_difference = abs(grouby_sta["predict_residual", "mean"]).max()
negative_max_difference = -max_abs_difference

fig, ax = Residual_Plotter.events_station_map(
    grouby_sta=grouby_sta,
    column="mean",
    cmap="seismic",
    title=f"{mask_after_sec} sec residual mean in 2016 prediction",
)
fig, ax = Residual_Plotter.events_station_map(
    grouby_sta=grouby_sta,
    column="std",
    cmap="Reds",
    title=f"{mask_after_sec} sec residual std in 2016 prediction",
)
# fig.savefig(
#     f"{output_path}/{mask_after_sec} sec residual {column} map.png",
#     dpi=300,
# )
