import matplotlib.pyplot as plt

plt.subplots()  # without this line will cause kernel crashed: when matplotlib and torch import simultaneously
import cartopy.crs as ccrs
from cartopy.mpl import ticker
import cartopy
import numpy as np
from multiple_sta_dataset import multiple_station_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


class Plot_Train_Test_Data:
    def event_histogram(
        train_catalog=None, test_catalog=None, key=None, xlabel=None, title=None
    ):
        fig, ax = plt.subplots()
        ax.hist(train_catalog[f"{key}"], bins=30, ec="black", label="train")
        ax.hist(test_catalog[f"{key}"], bins=30, ec="black", label="test", alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(f"{xlabel}", fontsize=15)
        ax.set_ylabel("Number of events", fontsize=15)
        ax.legend()
        if title:
            ax.set_title(f"{title}")
        return fig, ax

    def event_map(train_catalog=None, test_catalog=None, title=None):
        src_crs = ccrs.PlateCarree()
        fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))
        ax_map.coastlines("10m")
        ax_map.scatter(
            train_catalog["lon"] + train_catalog["lon_minute"] / 60,
            train_catalog["lat"] + train_catalog["lat_minute"] / 60,
            edgecolors="k",
            linewidth=1,
            marker="o",
            c="grey",
            s=2 ** train_catalog["magnitude"],
            zorder=3,
            alpha=0.5,
            label="train",
        )
        ax_map.scatter(
            test_catalog["lon"] + test_catalog["lon_minute"] / 60,
            test_catalog["lat"] + test_catalog["lat_minute"] / 60,
            edgecolors="k",
            linewidth=1,
            marker="o",
            c="orange",
            s=2 ** test_catalog["magnitude"],
            zorder=3,
            alpha=0.5,
            label="test",
        )
        ax_map.add_feature(cartopy.feature.OCEAN, edgecolor="k")

        xmin, xmax = ax_map.get_xlim()
        ymin, ymax = ax_map.get_ylim()
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
        if title:
            ax_map.set_title(f"{title}")
        ax_map.legend()
        return fig, ax_map

    def pga_histogram(traces_catalog=None, test_year=None, title=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(
            traces_catalog.query(f"year!={test_year}")["pga"],
            bins=30,
            ec="black",
            label="train",
        )
        ax.hist(
            traces_catalog.query(f"year=={test_year}")["pga"],
            bins=30,
            alpha=0.8,
            ec="black",
            label="test",
        )
        pga_threshold = np.log10(
            [1e-5, 0.008, 0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0, 10]
        )
        label = ["0", "1", "2", "3", "4", "5-", "5+", "6-", "6+", "7"]
        ax.vlines(pga_threshold[1:-1], 0, 17700, linestyles="dotted", color="k")
        for i in range(len(label)):
            if label[i] == "0":
                continue
            ax.text(
                ((pga_threshold[i] + pga_threshold[i + 1]) / 2) - 0.05, 15000, label[i]
            )
        ax.set_yscale("log")
        ax.set_xlabel(r"PGA log(${m/s^2}$)", fontsize=15)
        ax.set_ylabel("Number of traces", fontsize=15)
        fig.legend(fontsize=13)
        if title:
            ax.set_title(f"{title}")
        return fig, ax


class Increase_High_Data_Test:
    def load_dataset_into_list(
        data_path, oversample_rate=1, bias_to_close_station=False
    ):
        dataset = multiple_station_dataset(
            data_path,
            mode="train",
            mask_waveform_sec=3,
            weight_label=False,
            oversample=oversample_rate,
            oversample_mag=4,
            test_year=2016,
            mask_waveform_random=True,
            mag_threshold=0,
            label_key="pga",
            input_type="acc",
            data_length_sec=15,
            station_blind=True,
            bias_to_closer_station=bias_to_close_station,
        )
        origin_loader = DataLoader(dataset, batch_size=16)
        origin_PGA = []
        for sample in tqdm(origin_loader):
            tmp_pga = torch.index_select(
                sample["label"].flatten(),
                0,
                sample["label"].flatten().nonzero().flatten(),
            ).tolist()
            origin_PGA.extend(tmp_pga)
        return origin_PGA

    def plot_pga_histogram(
        bias_closed_sta_PGA=None,
        oversampled_PGA=None,
        origin_PGA=None,
        origin_high_intensity_rate=None,
        oversampled_high_intensity_rate=None,
        bias_closed_sta_high_intensity_rate=None,
    ):
        label = ["2", "3", "4", "5-", "5+", "6-", "6+", "7"]
        pga_threshold = np.log10([0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0, 10])

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.hist(bias_closed_sta_PGA, bins=32, edgecolor="k", label="bias_closed_sta")
        ax.hist(oversampled_PGA, bins=32, edgecolor="k", label="oversampled", alpha=0.6)
        ax.hist(origin_PGA, bins=32, edgecolor="k", label="origin", alpha=0.6)
        ax.vlines(pga_threshold[1:-1], 0, 40000, linestyles="dotted", color="k")
        for i in range(len(pga_threshold) - 1):
            ax.text((pga_threshold[i] + pga_threshold[i + 1]) / 2, 50000, label[i])
        ax.text(
            0.01,
            0.8,
            f"high intensity rate:\norigin: {np.round(origin_high_intensity_rate,2)}\noversampled: {np.round(oversampled_high_intensity_rate,2)}\nbias to station: {np.round(bias_closed_sta_high_intensity_rate,2)}",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_xlim(-2.75, 1.25)
        ax.set_ylabel("Number of traces", size=14)
        ax.set_xlabel(r"log(PGA (${m/s^2}$))", size=14)
        ax.set_title("TSMIP PGA distribution in training", size=14)
        ax.set_yscale("log")
        fig.legend(loc="upper right")
        return fig, ax


def plot_station_distribution(stations=None, title=None):
    src_crs = ccrs.PlateCarree()
    fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))

    ax_map.coastlines("10m")

    ax_map.add_feature(
        cartopy.feature.OCEAN, zorder=2, edgecolor="k"
    )  # zorder越大的圖層 越上面

    sta = ax_map.scatter(
        stations["longitude"],
        stations["latitude"],
        edgecolors="gray",
        color="red",
        linewidth=0.5,
        marker="^",
        s=20,
        zorder=3,
        label="Station",
    )
    xmin = stations["longitude"].min() - 0.1
    xmax = stations["longitude"].max() + 0.1
    ymin = stations["latitude"].min() - 0.1
    ymax = stations["latitude"].max() + 0.1
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
    ax_map.legend()
    if title:
        ax_map.set_title(title)
    return fig, ax_map


def plot_received_traces_station_map(
    total_station_value_counts, title="Received traces map", output_path=None
):
    src_crs = ccrs.PlateCarree()
    fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))
    ax_map.coastlines("10m")
    ax_map.scatter(
        total_station_value_counts["longitude"],
        total_station_value_counts["latitude"],
        edgecolors="k",
        linewidth=1,
        marker="o",
        s=total_station_value_counts["counts"] * 1.5,
        zorder=3,
        alpha=0.5,
    )
    ax_map.set_title(f"{title}")
    if output_path:
        fig.savefig(f"{output_path}/{title}.png", dpi=300)
    return fig, ax_map
