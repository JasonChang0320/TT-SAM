import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import os

mask_after_sec = 10
test_year = 2016
output_path = f"predict/station_blind_Vs30_bias2closed_station_{test_year}/{mask_after_sec} sec residual plots"
prediction_with_info = pd.read_csv(
    f"predict/station_blind_Vs30_bias2closed_station_{test_year}/{mask_after_sec} sec model11 with all info.csv"
)

miss_alarm = (prediction_with_info["predict"] < np.log10(0.25)) & (
    prediction_with_info["answer"] >= np.log10(0.25)
)
false_alarm = (prediction_with_info["predict"] >= np.log10(0.25)) & (
    prediction_with_info["answer"] < np.log10(0.25)
)

wrong_predict = prediction_with_info[miss_alarm | false_alarm]

meinong_earthquake = prediction_with_info.query("EQ_ID==24784.0")

no_meinong_predict = prediction_with_info.query("EQ_ID!=24784.0")

# plot x: feature, y: residul (predict-answer)
for column in prediction_with_info.columns:
    fig, ax = plt.subplots()
    ax.scatter(
        prediction_with_info[f"{column}"],
        prediction_with_info["predict"] - prediction_with_info["answer"],
        s=10,
        alpha=0.3,
        label="others",
    )
    ax.scatter(
        prediction_with_info.query("EQ_ID==24784.0")[f"{column}"],
        prediction_with_info.query("EQ_ID==24784.0")["predict"]
        - prediction_with_info.query("EQ_ID==24784.0")["answer"],
        s=10,
        alpha=0.3,
        c="r",
        label="meinong eq",
    )
    residual_mean = np.round(
        (prediction_with_info["predict"] - prediction_with_info["answer"]).mean(), 3
    )
    residual_std = np.round(
        (prediction_with_info["predict"] - prediction_with_info["answer"]).std(), 3
    )
    wrong_predict_rate = np.round(len(wrong_predict) / len(prediction_with_info), 3)
    wrong_predict_count = len(wrong_predict)
    ax.legend()
    ax.set_xlabel(f"{column}")
    ax.set_ylabel("predict-answer")
    ax.set_title(
        f"Predicted residual in {test_year} \n mean: {residual_mean}, std: {residual_std}, wrong rate: {wrong_predict_rate}"
    )
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # fig.savefig(
    #     f"{output_path}/{column}.png",
    #     dpi=300,
    # )

# plot event residual on map
eq_id = 24784.0
earthquake = prediction_with_info.query(f"EQ_ID =={eq_id}")
residual = earthquake["predict"] - earthquake["answer"]

max_abs_difference = abs(residual).max()

src_crs = ccrs.PlateCarree()
fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))
ax_map.coastlines("10m")
scatter = ax_map.scatter(
    earthquake["longitude"],
    earthquake["latitude"],
    edgecolors="k",
    linewidth=1,
    marker="o",
    s=15,
    zorder=3,
    c=residual,
    cmap="seismic",
    alpha=0.5,
    vmin=-max_abs_difference,
    vmax=max_abs_difference,
)

cbar = plt.colorbar(scatter)

cbar.set_label(r"predict-answer log(PGA ${m/s^2}$)")

ax_map.set_title(f"{mask_after_sec} sec 2016 Meinong earthquake residual in prediction")

# plot all prediction residual on map
prediction_with_info["predict_residual"] = (
    prediction_with_info["predict"] - prediction_with_info["answer"]
)


grouby_sta = prediction_with_info.groupby("station_name").agg(
    {"longitude": "first", "latitude": "first", "predict_residual": ["mean", "std"]}
)
max_abs_difference = abs(grouby_sta["predict_residual", "mean"]).max()
negative_max_difference = -max_abs_difference
for column, cmap in zip(["mean", "std"], ["seismic", "Reds"]):
    rc_crs = ccrs.PlateCarree()
    fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))
    ax_map.coastlines("10m")
    if column == "std":
        max_abs_difference = None
        negative_max_difference = None
    scatter = ax_map.scatter(
        grouby_sta["longitude", "first"],
        grouby_sta["latitude", "first"],
        edgecolors="k",
        linewidth=1,
        marker="o",
        s=15,
        zorder=3,
        c=grouby_sta["predict_residual", f"{column}"],
        cmap=cmap,
        alpha=0.5,
        vmin=negative_max_difference,
        vmax=max_abs_difference,
    )
    cbar = plt.colorbar(scatter)
    if column == "std":
        cbar.set_label(r"standard deviation")
    if column == "mean":
        cbar.set_label(r"predict-answer log(PGA ${m/s^2}$)")

    ax_map.set_title(f"{mask_after_sec} sec residual {column} in 2016 prediction")
    # fig.savefig(
    #     f"{output_path}/{mask_after_sec} sec residual {column} map.png",
    #     dpi=300,
    # )
