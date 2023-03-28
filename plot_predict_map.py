import bisect
import os

import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry as sgeom
import sklearn.metrics as metrics
from cartopy.geodesic import Geodesic
from cartopy.mpl import ticker
from scipy.interpolate import griddata


class TaiwanIntensity:
    label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
    pga = np.log10(
        [1e-5, 0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0]
    )  # log10(m/s^2)
    pgv = np.log10(
        [1e-5, 0.002, 0.007, 0.019, 0.057, 0.15, 0.3, 0.5, 0.8, 1.4]
    )  # log10(m/s)

    def __init__(self):
        self.pga_ticks = self.get_ticks(self.pga)
        self.pgv_ticks = self.get_ticks(self.pgv)

    def calculate(self, pga, pgv=None, label=False):
        pga_intensity = bisect.bisect(self.pga, pga) - 1
        intensity = pga_intensity

        if pga > self.pga[5] and pgv is not None:
            pgv_intensity = bisect.bisect(self.pgv, pgv) - 1
            if pgv_intensity > pga_intensity:
                intensity = pgv_intensity

        if label:
            return self.label[intensity]
        else:
            return intensity

    @staticmethod
    def get_ticks(array):
        ticks = np.cumsum(array, dtype=float)
        ticks[2:] = ticks[2:] - ticks[:-2]
        ticks = ticks[1:] / 2
        ticks = np.append(ticks, (ticks[-1] * 2 - ticks[-2]))
        return ticks


def plot_intensity_map(
    trace_info=None,
    eventmeta=None,
    label_type="pga",
    true_label=None,
    pred_label=None,
    center=None,
    pad=None,
    sec=None,
    title=None,
    output_dir=None,
    EQ_ID=None,
    grid_method="linear",
    Pwave_vel=6.5,
    Swave_vel=3.5,
):
    intensity = TaiwanIntensity()
    src_crs = ccrs.PlateCarree()
    fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))

    ax_map.coastlines("10m")

    cmap = mpl.colors.ListedColormap(
        [
            "#ffffff",
            "#e0ffe3",
            "#34ff32",
            "#fefd32",
            "#fe8532",
            "#fd5233",
            "#c43f3b",
            "#9d4646",
            "#9a4c86",
            "#b51fea",
        ]
    )
    if label_type == "pga":
        norm = mpl.colors.BoundaryNorm(intensity.pga, cmap.N)
        intensity_ticks = intensity.pga_ticks
    if label_type == "pgv":
        norm = mpl.colors.BoundaryNorm(intensity.pgv, cmap.N)
        intensity_ticks = intensity.pgv_ticks

    numcols, numrows = 100, 200
    xi = np.linspace(
        min(trace_info["longitude"]), max(trace_info["longitude"]), numcols
    )
    yi = np.linspace(min(trace_info["latitude"]), max(trace_info["latitude"]), numrows)
    xi, yi = np.meshgrid(xi, yi)

    grid_pred = griddata(
        (trace_info["longitude"], trace_info["latitude"]),
        pred_label,
        (xi, yi),
        method=grid_method,
    )
    ax_map.add_feature(
        cartopy.feature.OCEAN, zorder=2, edgecolor="k"
    )  # zorder越大的圖層 越上面
    ax_map.contourf(xi, yi, grid_pred, cmap=cmap, norm=norm, zorder=1)

    sta = ax_map.scatter(
        trace_info["longitude"],
        trace_info["latitude"],
        c=true_label,
        cmap=cmap,
        norm=norm,
        edgecolors="k",
        linewidth=1,
        marker="o",
        s=20,
        zorder=3,
        label="True Intensity",
    )
    event_lon = eventmeta["longitude"]
    event_lat = eventmeta["latitude"]
    ax_map.scatter(
        event_lon,
        event_lat,
        color="red",
        edgecolors="k",
        linewidth=1,
        marker="*",
        s=500,
        zorder=10,
        label="Epicenter",
    )
    gd = Geodesic()
    geoms = []
    for wave_velocity in [Pwave_vel, Swave_vel]:
        radius = (
            trace_info["epdis (km)"][
                trace_info["p_picks"] == trace_info["p_picks"].min()
            ].values[0]
            + sec * wave_velocity
        ) * 1000
        cp = gd.circle(lon=event_lon, lat=event_lat, radius=radius)
        geoms.append(sgeom.Polygon(cp))
    ax_map.add_geometries(
        geoms,
        crs=src_crs,
        edgecolor=["k", "r"],
        color=["grey", "red"],
        alpha=0.2,
        zorder=2.5,
    )
    ax_map.text(
        event_lon + 0.15,
        event_lat,
        f"M{eventmeta['magnitude'].values[0]}",
        va="center",
        zorder=11,
    )
    xmin, xmax = ax_map.get_xlim()
    ymin, ymax = ax_map.get_ylim()

    if xmax - xmin > ymax - ymin:  # check if square
        ymin = (ymax + ymin) / 2 - (xmax - xmin) / 2
        ymax = (ymax + ymin) / 2 + (xmax - xmin) / 2
    else:
        xmin = (xmax + xmin) / 2 - (ymax - ymin) / 2
        xmax = (xmax + xmin) / 2 + (ymax - ymin) / 2

    if center:
        xmin, xmax, ymin, ymax = [
            center[0] - pad,
            center[0] + pad,
            center[1] - pad,
            center[1] + pad,
        ]

    xticks = ticker.LongitudeLocator(nbins=5)._raw_ticks(xmin, xmax)
    yticks = ticker.LatitudeLocator(nbins=5)._raw_ticks(ymin, ymax)

    ax_map.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax_map.set_yticks(yticks, crs=ccrs.PlateCarree())

    ax_map.xaxis.set_major_formatter(
        ticker.LongitudeFormatter(zero_direction_label=True)
    )
    ax_map.yaxis.set_major_formatter(ticker.LatitudeFormatter())

    ax_map.xaxis.set_ticks_position("both")
    ax_map.yaxis.set_ticks_position("both")

    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    if title:
        ax_map.set_title(title)
    else:
        ax_map.set_title(
            f"EQ ID: {EQ_ID} {sec} sec Predicted {label_type} Intensity Map"
        )
    cbar = plt.colorbar(sta, extend="both")
    cbar.set_ticks(intensity_ticks)
    cbar.set_ticklabels(intensity.label)
    cbar.set_label("Seismic Intensity")
    plt.legend()
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"intensity_{sec}s.png"), format="png")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax_map


def warning_map(
    trace_info=None,
    eventmeta=None,
    label_type="pga",
    intensity="IV",
    EQ_ID=None,
    sec=None,
    label_threshold=None,
    title=None,
    Pwave_vel=6.5,
    Swave_vel=3.5,
):
    true_warn_filter = (trace_info["predict"] > label_threshold) & (
        (trace_info["answer"] > label_threshold)
    )
    true_not_warn_filter = (trace_info["predict"] <= label_threshold) & (
        (trace_info["answer"] <= label_threshold)
    )
    loss_warn_filter = (trace_info["predict"] <= label_threshold) & (
        (trace_info["answer"] > label_threshold)
    )
    wrong_warn_filter = (trace_info["predict"] > label_threshold) & (
        (trace_info["answer"] <= label_threshold)
    )
    predict_filter = [true_not_warn_filter, loss_warn_filter, wrong_warn_filter]

    title = f"EQ_ID: {EQ_ID}, {sec} sec performance, warning threshold: {intensity}"
    src_crs = ccrs.PlateCarree()
    fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))

    ax_map.coastlines("10m")

    numcols, numrows = 100, 200
    xi = np.linspace(
        min(trace_info["longitude"]), max(trace_info["longitude"]), numcols
    )
    yi = np.linspace(min(trace_info["latitude"]), max(trace_info["latitude"]), numrows)
    xi, yi = np.meshgrid(xi, yi)

    ax_map.add_feature(
        cartopy.feature.OCEAN, zorder=2, edgecolor="k"
    )  # zorder越大的圖層 越上面

    sta_num = len(trace_info[true_warn_filter])
    warning_time = trace_info[true_warn_filter][f"{label_type}_time"] / 200 - (sec + 5)

    cvals = [0, warning_time.mean(), warning_time.max()]
    colors = ["white", "orange", "red"]

    if warning_time.min() < 0:
        cvals = [warning_time.min(), 0, warning_time.max()]
        colors = ["purple", "white", "red"]

    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    warn_sta = ax_map.scatter(
        trace_info[true_warn_filter]["longitude"],
        trace_info[true_warn_filter]["latitude"],
        c=warning_time,
        norm=norm,
        cmap=cmap,
        edgecolors="k",
        linewidth=1,
        marker="o",
        s=30,
        zorder=3,
        label=f"TP: {sta_num}",
        alpha=0.7,
    )

    for filter, color, label, marker in zip(
        predict_filter,
        ["#009ACD", "green", "purple"],
        ["TN", "FN", "FP"],
        ["o", "^", "^"],
    ):
        sta_num = len(trace_info[filter])
        sta = ax_map.scatter(
            trace_info[filter]["longitude"],
            trace_info[filter]["latitude"],
            c=color,
            edgecolors="k",
            linewidth=1,
            marker=marker,
            s=30,
            zorder=3,
            label=f"{label}: {sta_num}",
            alpha=0.7,
        )

    event_lon = eventmeta["longitude"]
    event_lat = eventmeta["latitude"]
    ax_map.scatter(
        event_lon,
        event_lat,
        color="red",
        edgecolors="k",
        linewidth=1,
        marker="*",
        s=500,
        zorder=10,
        label="Epicenter",
    )
    gd = Geodesic()
    geoms = []
    for wave_velocity in [Pwave_vel, Swave_vel]:
        radius = (
            trace_info["epdis (km)"][
                trace_info["p_picks"] == trace_info["p_picks"].min()
            ].values[0]
            + sec * wave_velocity
        ) * 1000
        cp = gd.circle(lon=event_lon, lat=event_lat, radius=radius)
        geoms.append(sgeom.Polygon(cp))
    ax_map.add_geometries(
        geoms,
        crs=src_crs,
        edgecolor=["k", "r"],
        color=["grey", "red"],
        alpha=0.2,
        zorder=2.5,
    )
    ax_map.text(
        event_lon + 0.15,
        event_lat,
        f"M{eventmeta['magnitude'].values[0]}",
        va="center",
        zorder=11,
    )
    xmin, xmax = ax_map.get_xlim()
    ymin, ymax = ax_map.get_ylim()

    if xmax - xmin > ymax - ymin:  # check if square
        ymin = (ymax + ymin) / 2 - (xmax - xmin) / 2
        ymax = (ymax + ymin) / 2 + (xmax - xmin) / 2
    else:
        xmin = (xmax + xmin) / 2 - (ymax - ymin) / 2
        xmax = (xmax + xmin) / 2 + (ymax - ymin) / 2

    xticks = ticker.LongitudeLocator(nbins=5)._raw_ticks(xmin, xmax)
    yticks = ticker.LatitudeLocator(nbins=5)._raw_ticks(ymin, ymax)

    ax_map.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax_map.set_yticks(yticks, crs=ccrs.PlateCarree())

    ax_map.xaxis.set_major_formatter(
        ticker.LongitudeFormatter(zero_direction_label=True)
    )
    ax_map.yaxis.set_major_formatter(ticker.LatitudeFormatter())

    ax_map.xaxis.set_ticks_position("both")
    ax_map.yaxis.set_ticks_position("both")

    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    ax_map.legend()
    if title:
        ax_map.set_title(title)
    cbar = plt.colorbar(warn_sta, extend="both")
    cbar.set_label("Warning time (sec)")

    return fig, ax_map


def true_predicted(
    y_true,
    y_pred,
    time,
    agg="mean",
    quantile=True,
    ms=None,
    ax=None,
    axis_fontsize=20,
    point_size=2,
    target="pga",
):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
    ax.set_aspect("equal")

    if quantile:
        c_quantile = np.sum(
            y_pred[:, :, 0]
            * (
                1
                - norm.cdf((y_true.reshape(-1, 1) - y_pred[:, :, 1]) / y_pred[:, :, 2])
            ),
            axis=-1,
            keepdims=False,
        )
    else:
        c_quantile = None

    if agg == "mean":
        y_pred_point = np.sum(y_pred[:, :, 0] * y_pred[:, :, 1], axis=1)
    elif agg == "point":
        y_pred_point = y_pred
    else:
        raise ValueError(f'Aggregation type "{agg}" unknown')

    limits = (np.min(y_true) - 0.5, np.max(y_true) + 0.5)
    ax.plot(limits, limits, "k-", zorder=1)
    if ms is None:
        cbar = ax.scatter(
            y_true,
            y_pred_point,
            c=c_quantile,
            cmap="coolwarm",
            zorder=2,
            alpha=0.3,
            s=point_size,
        )
    else:
        cbar = ax.scatter(
            y_true, y_pred_point, s=ms, c=c_quantile, cmap="coolwarm", zorder=2
        )

    intensity = TaiwanIntensity()
    if target == "pga":
        intensity_threshold = intensity.pga
        ticks = intensity.pga_ticks
    elif target == "pgv":
        intensity_threshold = intensity.pgv
        ticks = intensity.pgv_ticks
    ax.hlines(
        intensity_threshold[3:-1],
        limits[0],
        intensity_threshold[3:-1],
        linestyles="dotted",
    )
    ax.vlines(
        intensity_threshold[3:-1],
        limits[0],
        intensity_threshold[3:-1],
        linestyles="dotted",
    )
    for i, label in zip(ticks[2:-2], intensity.label[2:-2]):
        ax.text(i, limits[0], label, va="bottom", fontsize=axis_fontsize - 7)

    ax.set_xlabel(f"ture_{target} log(m/s^2)", fontsize=axis_fontsize)
    ax.set_ylabel(f"predicted_{target} log(m/s^2)", fontsize=axis_fontsize)
    ax.set_title(f"{time}s True Predict Plot", fontsize=axis_fontsize + 5)
    ax.tick_params(axis="x", labelsize=axis_fontsize - 5)
    ax.tick_params(axis="y", labelsize=axis_fontsize - 5)
    # ax.set_ylim(-3.5,1.5)
    # ax.set_xlim(-3.5,1.5)

    r2 = metrics.r2_score(y_true, y_pred_point)
    ax.text(
        min(np.min(y_true), limits[0]),
        max(np.max(y_pred_point), limits[1]),
        f"$R^2={r2:.2f}$",
        va="top",
        fontsize=axis_fontsize - 5,
    )

    # return ax, cbar
    return fig, ax


def warning_time_hist(
    prediction=None,
    catalog=None,
    mask_after_sec=None,
    EQ_ID=None,
    warning_mag_threshold=None,
    bins=20,
    label_threshold=None,
    label_type="pga",
    sampling_rate=200,
    first_pick_sec=5,
):
    prediction = pd.merge(prediction, catalog, how="left", on="EQ_ID")
    warning_time_filter = (
        prediction[f"{label_type}_time"]
        > (first_pick_sec + mask_after_sec) * sampling_rate
    )
    magnitude_filter = prediction["magnitude"] >= warning_mag_threshold

    prediction_for_warning = prediction[warning_time_filter & magnitude_filter]

    warning_time = (
        prediction_for_warning[f"{label_type}_time"]
        - ((first_pick_sec + mask_after_sec) * sampling_rate)
    ) / sampling_rate
    prediction_for_warning.insert(5, "warning_time (sec)", warning_time)

    if EQ_ID:
        eq_id_filter = prediction_for_warning["EQ_ID"] == EQ_ID
        prediction_for_warning = prediction_for_warning[eq_id_filter]

    true_predict_filter = (prediction_for_warning["predict"] > label_threshold) & (
        (prediction_for_warning["answer"] > label_threshold)
    )
    positive_filter = prediction_for_warning["predict"] > label_threshold
    true_filter = prediction_for_warning["answer"] > label_threshold

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hist(
        prediction_for_warning[true_predict_filter]["warning_time (sec)"],
        bins=bins,
        ec="black",
    )
    describe = prediction_for_warning[true_predict_filter][
        "warning_time (sec)"
    ].describe()
    count = int(describe["count"])
    mean = np.round(describe["mean"], 2)
    std = np.round(describe["std"], 2)
    median = np.round(describe["50%"], 2)
    max = np.round(describe["max"], 2)
    precision = np.round(
        len(prediction_for_warning[true_predict_filter])
        / len(prediction_for_warning[positive_filter]),
        2,
    )
    recall = np.round(
        len(prediction_for_warning[true_predict_filter])
        / len(prediction_for_warning[true_filter]),
        2,
    )
    if EQ_ID:
        ax.set_title(
            f"Warning time in EQ ID: {EQ_ID}, \n after first triggered station {mask_after_sec} sec",
            fontsize=18,
        )
    else:
        ax.set_title(
            f"Warning time\n after first triggered station {mask_after_sec} sec",
            fontsize=18,
        )
    ax.set_xlabel("Warning time (sec)", fontsize=15)
    ax.set_ylabel("Number of stations", fontsize=15)
    ax.text(
        0.45,
        0.7,
        f"mean: {mean} s\nstd: {std} s\nmedian: {median}s\nmax: {max} s\neffective warning stations: {count}\nprecision: {precision}\nrecall: {recall}",
        transform=ax.transAxes,
        fontsize=14,
    )
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    return fig, ax


def correct_warning_with_epidist(
    event_prediction=None,
    mask_after_sec=None,
    label_type="pga",
    label_threshold=None,
    sampling_rate=200,
    first_pick_sec=5,
):
    EQ_ID = int(event_prediction["EQ_ID"][0])
    fig, ax = plt.subplots()
    true_warning_prediction = event_prediction.query(
        f"predict > {label_threshold} and answer > {label_threshold}"
    )
    pga_time = (
        true_warning_prediction[f"{label_type}_time"] / sampling_rate - first_pick_sec
    )
    pick_time = true_warning_prediction["p_picks"] / sampling_rate - first_pick_sec
    ax.scatter(
        true_warning_prediction["epdis (km)"], pga_time, label=f"{label_type}_time"
    )
    ax.scatter(true_warning_prediction["epdis (km)"], pick_time, label="P arrival")
    ax.axhline(
        y=mask_after_sec,
        xmax=true_warning_prediction["epdis (km)"].max() + 10,
        linestyle="dashed",
        c="r",
        label="warning",
    )
    ax.legend()
    for index in true_warning_prediction["epdis (km)"].index:
        distance = [
            true_warning_prediction["epdis (km)"][index],
            true_warning_prediction["epdis (km)"][index],
        ]
        time = [pga_time[index], pick_time[index]]
        ax.plot(distance, time, c="grey")
    ax.set_title(f"EQ ID: {EQ_ID} Warning time")
    ax.set_xlabel("epicentral distance (km)")
    ax.set_ylabel("time (sec)")
    return fig, ax
