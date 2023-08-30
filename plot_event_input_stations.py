import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy
from cartopy.geodesic import Geodesic
import shapely.geometry as sgeom
from cartopy.mpl import ticker
import os
import matplotlib.pyplot as plt

# plot input station map
mask_after_sec = 3
eq_id = 24784
prediction_with_info = pd.read_csv(
    f"predict/station_blind_noVs30_bias2closed_station_2016/{mask_after_sec} sec ensemble 510 with all info.csv"
)
record_prediction = prediction_with_info.query(f"EQ_ID=={eq_id}")
first_trigger_time = min(record_prediction["p_picks"])
input_station = record_prediction[
    record_prediction["p_picks"] < first_trigger_time + (mask_after_sec * 200)
]


def plot_station_map(
    trace_info=None,
    center=None,
    pad=None,
    sec=None,
    title=None,
    output_dir=None,
    EQ_ID=None,
    Pwave_vel=6.5,
    Swave_vel=3.5,
):
    src_crs = ccrs.PlateCarree()
    fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))

    ax_map.coastlines("10m")

    ax_map.add_feature(
        cartopy.feature.OCEAN, zorder=2, edgecolor="k"
    )  # zorder越大的圖層 越上面

    sta = ax_map.scatter(
        trace_info["longitude"],
        trace_info["latitude"],
        edgecolors="k",
        linewidth=1,
        marker="^",
        s=20,
        zorder=3,
        label="Station",
    )
    event_lon = trace_info["event_lon"].values[0]
    event_lat = trace_info["event_lat"].values[0]
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
        event_lon + 0.1,
        event_lat,
        f"M{trace_info['magnitude'].values[0]}",
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
    xmin = trace_info["longitude"].min() - 0.6
    xmax = trace_info["longitude"].max() + 0.6
    ymin = trace_info["latitude"].min() - 0.6
    ymax = trace_info["latitude"].max() + 0.6
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
        ax_map.set_title(f"EQ ID: {EQ_ID}, {sec} sec Input Stations")
    plt.legend()
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"intensity_{sec}s.png"), format="png")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax_map


if len(input_station) >= 25:
    input_station = input_station[:25]

fig, ax = plot_station_map(
    trace_info=input_station,
    sec=mask_after_sec,
    EQ_ID=eq_id,
    pad=100,
)

# if len(input_station) >= 25:
#     input_station = input_station[:25]
# # 標示測站
#     for i in range(13, len(input_station)):
#         station_name = input_station["station_name"][i]
#         lon=input_station["longitude"][i]
#         lat=input_station["latitude"][i]
#         adjust_x=0
#         adjust_y=0
#         # if i==16: #KAU026
#         #     adjust_x=-0.2
#         #     adjust_y=0.05
#         # if i==17: #CHY118
#         #     adjust_x=-0.3
#         #     adjust_y=-0.3
#         # if i==18: #KAU025
#         #     adjust_x=0.25
#         #     adjust_y=-0.01
#         # if i==19: #CHY089
#         #     adjust_x=-0.1
#         #     adjust_y=-0.3
#         # if i==21: #KAU023
#         #     adjust_x=-0.1
#         #     adjust_y=-0.1
#         # if i==24: #KAU024
#         #     adjust_x=-0.4
#         #     adjust_y=-0.1
#         ann=ax.annotate(
#             f"{station_name}",
#             xy=(lon, lat),
#             xytext=(lon-0.2-adjust_x,lat-0.2-adjust_y),
#             fontsize=10,
#             arrowprops=dict(arrowstyle='-', color='gray'),
#             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round')
#         )

# fig.savefig(
#     f"{path}/eqid{eq_id}_{mask_after_sec}_sec_station_input.png"
# )
