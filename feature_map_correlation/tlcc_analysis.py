import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr


class Calculator:
        
    def first_occurrences_indices(b):
        first_indices = {}  # 用字典来存储不同数字的第一次出现的索引

        for i, item in enumerate(b):
            if item not in first_indices:
                first_indices[item] = i  # 记录不同数字的第一次出现的索引

        return first_indices


    def normalize_to_zero_one(arr):
        # 找到数组的最小值和最大值
        min_val = arr.min()
        max_val = arr.max()

        # 将数组线性缩放到0到1之间
        normalized_arr = (arr - min_val) / (max_val - min_val)

        return normalized_arr


    def calculate_tlcc(time_series1, time_series2, max_delay):
        """
        計算TLCC（時滯交叉相關性）以及相應的時間延遲和TLCC值。

        參數：
        - time_series1: 第一個時間序列
        - time_series2: 第二個時間序列
        - max_delay: 最大時滯的範圍

        返回值：
        - delay: 時間延遲的數組
        - tlcc_values: 對應的TLCC（皮爾森相關性）值的數組
        """
        delay = np.arange(-max_delay, max_delay + 1)
        tlcc_values = []
        for i, d in enumerate(delay):
            if d < 0:
                x1_lagged = time_series1[: len(time_series1) + d]
                x2_lagged = time_series2[-d:]
            else:
                x1_lagged = time_series1[d:]
                x2_lagged = time_series2[: len(time_series2) - d]
            # if d % 5 == 0:
            #     fig,ax=plt.subplots()
            #     ax.plot(x1_lagged,c="k")
            #     ax.plot(x2_lagged,c="r")
            #     ax.set_title(f"delay:{d}")
            #     plt.grid(True)

            # 計算皮爾森相關性
            pearson_corr, _ = pearsonr(x1_lagged, x2_lagged)
            tlcc_values.append(pearson_corr)

        return delay, tlcc_values

class Plotter:

    def plot_waveform(waveform, eq_id, input_station,index, output_path=None):
        fig, ax = plt.subplots(3, 1, figsize=(14, 7))
        for j in range(len(ax)):
            ax[j].plot(waveform[:, j])
        ax[0].set_title(f"EQ_ID: {eq_id} input waveform{index+1},{input_station}")
        if output_path:
            fig.savefig(f"{output_path}/3 channel input waveform{index+1}.png", dpi=300)
        return fig, ax


    def plot_correlation_curve_with_shift_time(
        delay_values, tlcc_values, eq_id, attribute, index, mask_after_sec, output_path=None
    ):
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(delay_values, tlcc_values)
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.set_xlabel("Shift Time Sample", fontsize=15)
        ax.set_ylabel("TLCC (Pearson Correlation) Value", fontsize=15)
        ax.set_title(f"EQ_ID: {eq_id} {attribute}{index+1} TLCC Analysis", fontsize=15)
        ax.grid(True)
        if output_path:
            fig.savefig(
                f"{output_path}/{mask_after_sec} sec {attribute}{index+1} TLCC Analysis.png",
                dpi=300,
            )
        return fig, ax


    def plot_attribute_with_feature_map(
        attribute_arr,
        resized_feature_map,
        key,
        attribute,
        correlation_starttime,
        correlation_endtime,
        correlation,
        tlcc_values,
        input_station,
        output_path=None,
    ):
        x_pos = 0.05
        y_pos = 0.6
        fig, ax = plt.subplots(3, 1, figsize=(14, 7))
        ax[0].plot(attribute_arr, alpha=0.7)
        ax[1].plot(resized_feature_map, c="red")
        ax[2].plot(
            attribute_arr,
            alpha=0.7,
            label=f"{attribute}",
        )
        ax[2].plot(
            resized_feature_map,
            c="red",
            label="feature map",
        )
        for j in range(len(ax)):
            ax[j].axvline(x=correlation_starttime, color="grey", linestyle="--")
            ax[j].axvline(x=correlation_endtime, color="grey", linestyle="--")
        ax[2].text(
            x_pos,
            y_pos,
            f"correlation: {np.round(correlation, 2)}\nTLCC max correlation: {np.round(np.array(tlcc_values).max(),2)}",
            transform=ax[j].transAxes,
            fontsize=15,
            horizontalalignment="left",
            verticalalignment="top",
        )
        ax[0].set_title(
            f"EQ_ID: {key} {attribute}, station_name:{input_station}",
            fontsize=15,
        )
        ax[1].set_ylabel("normalized acc", fontsize=15)
        ax[-1].set_xlabel("time sample", fontsize=15)
        ax[-1].xaxis.set_tick_params(labelsize=15)
        ax[-1].yaxis.set_tick_params(labelsize=15)
        ax[2].legend()
        if output_path:
            fig.savefig(
                f"{output_path}/{attribute}_{input_station} with feature map.png",
                dpi=300,
            )
        return fig, ax


    def plot_correlation_hist(
        attribute_dict, attribute, TLCC_mean, TLCC_std, mask_after_sec, output_path=None
    ):
        # hist
        fig, ax = plt.subplots()
        ax.hist(
            np.array(attribute_dict[attribute]["tlcc_max_correlation"]),
            bins=15,
            edgecolor="k",
        )
        ax.set_xlabel("correlation", fontsize=12)
        ax.set_ylabel("number of traces", fontsize=12)
        ax.set_title(
            f"Correlation (TLCC) of \n{mask_after_sec} sec {attribute}",
            fontsize=15,
        )
        ax.text(
            0.8,
            0.8,
            f"mean:{TLCC_mean}\nstd:{TLCC_std}",
            transform=ax.transAxes,
            fontsize=12,
        )
        if output_path:
            fig.savefig(
                f"{output_path}/correlation (TLCC) with {attribute} histogram.png",
                dpi=300,
            )
        return fig, ax


    def plot_time_shifted_with_correlation(
        attribute_dict, attribute, TLCC_mean, TLCC_std, mask_after_sec, output_path=None
    ):
        fig, ax = plt.subplots()
        ax.scatter(
            attribute_dict[attribute]["max_delay"],
            attribute_dict[attribute]["tlcc_max_correlation"],
            alpha=0.5,
            s=15,
        )

        ax.set_xlabel("shifted time sample")
        ax.set_ylabel("max Pearson correlation")
        ax.set_title(
            f"Correlation (TLCC) with delay time{mask_after_sec} sec \n{attribute}, mean :{TLCC_mean}, std: {TLCC_std}",
            fontsize=15,
        )
        if output_path:
            fig.savefig(
                f"{output_path}/{mask_after_sec} sec {attribute} TLCC max correlation delay time.png",
                dpi=300,
            )
        return fig, ax


    def plot_time_shifted_with_hist(
        attribute_dict, attribute, delay_mean, delay_std, mask_after_sec, output_path=None
    ):
        fig, ax = plt.subplots()
        ax.hist(attribute_dict[attribute]["max_delay"], bins=15, edgecolor="k")
        ax.text(
            0.75,
            0.8,
            f"mean:{delay_mean}\nstd:{delay_std}",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_xlabel("shifted time sample", fontsize=12)
        ax.set_ylabel("number of traces", fontsize=12)
        ax.set_title(
            f"{mask_after_sec} sec {attribute}\ndistribution of time delay with max correlation (TLCC)",
            fontsize=15,
        )
        if output_path:
            fig.savefig(
                f"{output_path}/{mask_after_sec} sec {attribute} distribution of time delay with max correlation (TLCC).png",
                dpi=300,
            )
        return fig, ax


    def correlation_with_attributes_heat_map(data, attributes=None, output_path=None):
        fig, ax = plt.subplots()
        sns.heatmap(data, annot=True, cmap="Reds")

        ax.set_xticks([x + 0.5 for x in range(data.shape[1])])
        ax.set_xticklabels(["3", "5", "7", "10"], fontsize=12)

        ax.set_yticks([x + 0.5 for x in range(data.shape[0])])
        plt.yticks(rotation=0)
        ax.set_yticklabels(attributes, fontsize=12)

        ax.set_xlabel("second", fontsize=13)

        cbar = ax.collections[0].colorbar

        # 设置颜色条标签的字体大小
        cbar.set_label("Correlation", fontsize=12)
        plt.tight_layout()
        if output_path:
            fig.savefig(f"{output_path}/correlation_heat_map.png", dpi=300)
        return fig, ax
