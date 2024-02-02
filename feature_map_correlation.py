import matplotlib.pyplot as plt

plt.subplots()
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import os
import seaborn as sns
from scipy.ndimage import zoom
from CNN_Transformer_Mixtureoutput_TEAM import CNN_feature_map
from multiple_sta_dataset import multiple_station_dataset
import os
from scipy.stats import pearsonr
from scipy.signal import hilbert


# ==========functions=================
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


def plot_waveform(waveform, eq_id, input_station_list, output_path=None):
    fig, ax = plt.subplots(3, 1, figsize=(14, 7))
    for j in range(len(ax)):
        ax[j].plot(waveform[i, :, j])
    ax[0].set_title(f"EQ_ID: {eq_id} input waveform{i+1},{input_station_list[i]}")
    if output_path:
        fig.savefig(f"{output_path}/3 channel input waveform{i+1}.png", dpi=300)
    return fig, ax


def plot_correlation_curve_with_shift_time(
    delay_values, tlcc_values, eq_id, attribute, output_path=None
):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(delay_values, tlcc_values)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_xlabel("Shift Time Sample", fontsize=15)
    ax.set_ylabel("TLCC (Pearson Correlation) Value", fontsize=15)
    ax.set_title(f"EQ_ID: {eq_id} {attribute}{i+1} TLCC Analysis", fontsize=15)
    ax.grid(True)
    if output_path:
        fig.savefig(
            f"{output_path}/{mask_after_sec} sec {attribute}{i+1} TLCC Analysis.png",
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
    ax[0].plot(normalize_to_zero_one(attribute_arr), alpha=0.7)
    ax[1].plot(normalize_to_zero_one(resized_feature_map), c="red")
    ax[2].plot(
        normalize_to_zero_one(attribute_arr),
        alpha=0.7,
        label=f"{attribute}",
    )
    ax[2].plot(
        normalize_to_zero_one(resized_feature_map),
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


# ===================================

mask_after_sec = 10
sample_rate = 200
label = "pga"
data = multiple_station_dataset(
    "D:/TEAM_TSMIP/data/TSMIP_1999_2019_Vs30.hdf5",
    mode="test",
    mask_waveform_sec=mask_after_sec,
    test_year=2016,
    label_key=label,
    mag_threshold=0,
    input_type="acc",
    data_length_sec=15,
)
# need station name
output_path = "predict/station_blind_Vs30_bias2closed_station_2016"
predict = pd.read_csv(f"{output_path}/{mask_after_sec} sec model11 with all info.csv")

# ===========prepare model==============
device = torch.device("cuda")
num = 11
path = f"./model/model{num}.pt"
emb_dim = 150
mlp_dims = (150, 100, 50, 30, 10)
CNN_model = CNN_feature_map(mlp_input=5665).cuda()

full_model_parameter = torch.load(f"./model/model{num}.pt")
# ===========load CNN parameter==============
CNN_parameter = {}
for name, param in full_model_parameter.items():
    if (
        "model_CNN" in name
    ):  # model_CNN.conv2d1.0.weight : conv2d1.0.weight didn't match
        name = name.replace("model_CNN.", "")
        CNN_parameter[name] = param
CNN_model.load_state_dict(CNN_parameter)

event_index_list = []
for eq_id in data.events_index:
    event_index_list.append(eq_id[0][0, 0])

eq_first_index = first_occurrences_indices(event_index_list)

# plot feature map and calculate correlation
attribute_dict = {
    "euclidean_norm": {"correlation": [], "tlcc_max_correlation": [], "max_delay": []},
    "vertical_envelope": {
        "correlation": [],
        "tlcc_max_correlation": [],
        "max_delay": [],
    },
    "NS_envelope": {"correlation": [], "tlcc_max_correlation": [], "max_delay": []},
    "EW_envelope": {"correlation": [], "tlcc_max_correlation": [], "max_delay": []},
    "vertical_instantaneous_phase": {
        "correlation": [],
        "tlcc_max_correlation": [],
        "max_delay": [],
    },
    "NS_instantaneous_phase": {
        "correlation": [],
        "tlcc_max_correlation": [],
        "max_delay": [],
    },
    "EW_instantaneous_phase": {
        "correlation": [],
        "tlcc_max_correlation": [],
        "max_delay": [],
    },
    "vertical_instantaneous_freq": {
        "correlation": [],
        "tlcc_max_correlation": [],
        "max_delay": [],
    },
    "NS_instantaneous_freq": {
        "correlation": [],
        "tlcc_max_correlation": [],
        "max_delay": [],
    },
    "EW_instantaneous_freq": {
        "correlation": [],
        "tlcc_max_correlation": [],
        "max_delay": [],
    },
}
print(len(eq_first_index.keys()))
for key, index in tqdm(zip(eq_first_index.keys(), eq_first_index.values())):
    event_output_path = (
        f"{output_path}/{mask_after_sec} sec cnn feature map/each event/{str(key)}"
    )
    if not os.path.isdir(f"{event_output_path}"):
        os.makedirs(f"{event_output_path}")
    sample = data[index]
    waveform = sample["waveform"]

    not_padding_station_number = (
        (torch.from_numpy(sample["sta"]) != 0).all(dim=1).sum().item()
    )
    single_event_prediction = predict.query(f"EQ_ID=={key}")
    input_station_list = single_event_prediction["station_name"][
        :not_padding_station_number
    ].tolist()
    if len(input_station_list) < 25:
        input_station_list += [np.nan] * (25 - len(input_station_list))

    p_picks = sample["p_picks"].flatten().tolist()
    # plot 24784 input waveform
    if key == 24784:
        for i in range(not_padding_station_number):
            fig, ax = plot_waveform(waveform, key, input_station_list)

    cnn_input = torch.DoubleTensor(waveform).float().cuda()
    cnn_output, layer_output = CNN_model(cnn_input)
    numeric_array = np.array(layer_output[-1].detach().cpu(), dtype=np.float32)
    feature_map = np.mean(numeric_array, axis=1)
    scale_factor_h = waveform.shape[0] / feature_map.shape[0]
    scale_factor_w = waveform.shape[1] / feature_map.shape[1]

    # zoom out feature map
    resized_feature_map = zoom(feature_map, (scale_factor_h, scale_factor_w), order=3)
    component_dict = {}
    euclidean_waveform = np.linalg.norm(waveform, axis=2) / np.sqrt(3)
    component_dict[f"euclidean_norm"] = euclidean_waveform
    for com, component in enumerate(["vertical", "NS", "EW"]):
        analytic_signal = hilbert(waveform[:, :, com])
        envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.abs(
            (np.diff(instantaneous_phase) / (2.0 * np.pi) * sample_rate)
        )
        component_dict[f"{component}_envelope"] = envelope
        component_dict[f"{component}_instantaneous_phase"] = instantaneous_phase
        component_dict[f"{component}_instantaneous_freq"] = instantaneous_frequency

    for attribute in component_dict:
        for i in range(not_padding_station_number):
            correlation_starttime = p_picks[i] - sample_rate
            correlation_endtime = p_picks[0] + (mask_after_sec + 1) * sample_rate
            if mask_after_sec == 10:
                correlation_endtime = p_picks[0] + (mask_after_sec) * sample_rate
            try:
                correlation = np.corrcoef(
                    component_dict[attribute][
                        i, correlation_starttime:correlation_endtime
                    ],
                    resized_feature_map[i, correlation_starttime:correlation_endtime],
                )[0, 1]
                delay_values, tlcc_values = calculate_tlcc(
                    component_dict[attribute][
                        i, correlation_starttime:correlation_endtime
                    ],
                    resized_feature_map[i, correlation_starttime:correlation_endtime],
                    max_delay=100,
                )
            except:  # for second=10
                correlation = np.corrcoef(
                    component_dict[attribute][
                        i, correlation_starttime:correlation_endtime
                    ],
                    resized_feature_map[
                        i, correlation_starttime + 1 : correlation_endtime
                    ],
                )[0, 1]
                delay_values, tlcc_values = calculate_tlcc(
                    component_dict[attribute][
                        i, correlation_starttime:correlation_endtime
                    ],
                    resized_feature_map[
                        i, correlation_starttime + 1 : correlation_endtime
                    ],
                    max_delay=100,
                )
            attribute_dict[attribute]["correlation"].append(correlation)
            max_index = np.argmax(tlcc_values)
            max_correlation = tlcc_values[max_index]
            max_delay = delay_values[max_index]
            attribute_dict[attribute]["tlcc_max_correlation"].append(max_correlation)
            attribute_dict[attribute]["max_delay"].append(max_delay)

            if key == 24784:  # plot
                fig, ax = plot_correlation_curve_with_shift_time(
                    delay_values, tlcc_values, key, attribute, output_path=None
                )
                fig, ax = plot_attribute_with_feature_map(
                    component_dict[attribute][i],
                    resized_feature_map[i],
                    key,
                    attribute,
                    correlation_starttime,
                    correlation_endtime,
                    correlation,
                    tlcc_values,
                    input_station_list[i],
                )

output_path = f"{output_path}/{mask_after_sec} sec cnn feature map"

for attribute in attribute_dict:
    TLCC_mean = np.round(
        np.array(attribute_dict[attribute]["tlcc_max_correlation"]).mean(), 2
    )
    TLCC_std = np.round(
        np.array(attribute_dict[attribute]["tlcc_max_correlation"]).std(), 2
    )
    fig, ax = plot_correlation_hist(
        attribute_dict, attribute, TLCC_mean, TLCC_std, mask_after_sec, output_path=None
    )
    # x: time sample lag, y: max correlation (TLCC)
    fig, ax = plot_time_shifted_with_correlation(
        attribute_dict, attribute, TLCC_mean, TLCC_std, mask_after_sec, output_path=None
    )
    # max correlation time delay hist
    delay_mean = np.round(np.array(attribute_dict[attribute]["max_delay"]).mean(), 2)
    delay_std = np.round(np.array(attribute_dict[attribute]["max_delay"]).std(), 2)
    fig, ax = plot_time_shifted_with_hist(
        attribute_dict,
        attribute,
        delay_mean,
        delay_std,
        mask_after_sec,
        output_path=None,
    )

# belowed data is correlation with attributes in different seconds
data = np.array(
    [
        [0.61, 0.53, 0.49, 0.46],
        [0.68, 0.58, 0.52, 0.5],
        [0.59, 0.51, 0.47, 0.46],
        [0.58, 0.5, 0.47, 0.45],
        [0.29, 0.23, 0.18, 0.12],
        [0.29, 0.23, 0.18, 0.12],
        [0.29, 0.23, 0.18, 0.12],
        [0.3, 0.22, 0.16, 0.11],
        [0.29, 0.21, 0.15, 0.1],
        [0.3, 0.21, 0.15, 0.1],
    ]
)
attributes = [
    "Euclidean norm",
    "Vertical envelope",
    "NS envelope",
    "EW envelope",
    "Vertical phase",
    "NS phase",
    "EW phase",
    "Vertical frequency",
    "NS frequency",
    "EW frequency",
]
output_path = "./predict/station_blind_Vs30_bias2closed_station_2016"
fig, ax = correlation_with_attributes_heat_map(data, attributes, output_path=None)
