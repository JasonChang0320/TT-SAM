import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

mask_after_sec = 5
test_year = 2016
prediction_with_info = pd.read_csv(
    f"predict/station_blind_noVs30_bias2closed_station_{test_year}/{mask_after_sec} sec ensemble 510 with all info.csv"
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

    # fig.savefig(
    #     f"predict/station_blind_noVs30_bias2closed_station_{test_year}/{mask_sec} sec residual plots/{column}.png",
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

cbar.set_label("predict-answer log10(PGA (m/s^2))")

ax_map.set_title(f"{mask_after_sec} sec 2016 Meinong earthquake residual in prediction")
