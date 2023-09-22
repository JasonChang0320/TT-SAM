import matplotlib.pyplot as plt

plt.subplots()
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from scipy.ndimage import zoom

from CNN_Transformer_Mixtureoutput_TEAM import (
    CNN,
    MDN,
    MLP,
    PositionEmbedding,
    PositionEmbedding_Vs30,
    TransformerEncoder,
    full_model,
)
from multiple_sta_dataset import multiple_station_dataset
from plot_predict_map import true_predicted

mask_after_sec = 10
sample_rate = 200
eq_id = 24784
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
single_event_prediction = predict.query(f"EQ_ID=={eq_id}")
# ===========prepare model==============
device = torch.device("cuda")
num = 11
path = f"./model/model{num}.pt"
emb_dim = 150
mlp_dims = (150, 100, 50, 30, 10)
CNN_model = CNN(mlp_input=5665).cuda()
pos_emb_model = PositionEmbedding_Vs30(emb_dim=emb_dim).cuda()
transformer_model = TransformerEncoder()
mlp_model = MLP(input_shape=(emb_dim,), dims=mlp_dims).cuda()
mdn_model = MDN(input_shape=(mlp_dims[-1],)).cuda()
full_Model = full_model(
    CNN_model,
    pos_emb_model,
    transformer_model,
    mlp_model,
    mdn_model,
    pga_targets=25,
    data_length=3000,
).to(device)
full_Model.load_state_dict(torch.load(path))


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

# find specific eq_id
loader = DataLoader(dataset=data, batch_size=1)
for j, sample in tqdm(enumerate(loader)):
    if sample["EQ_ID"][:, :, 0].flatten().numpy().tolist()[0] == eq_id:
        break

# waveform average
waveform = sample["waveform"].numpy().reshape(25, 3000, 3)
average_waveform = np.mean(waveform, axis=2)
# station_name_list
not_padding_station_number = (sample["sta"].reshape(25, 4) != 0).all(dim=1).sum().item()
input_station_list = single_event_prediction["station_name"][
    :not_padding_station_number
].tolist()
if len(input_station_list) < 25:
    input_station_list += [np.nan] * (25 - len(input_station_list))
# input trace trigger time
p_picks = sample["p_picks"].flatten().tolist()

cnn_input = torch.DoubleTensor(sample["waveform"].reshape(-1, 3000, 3)).float().cuda()
cnn_output, layer_output = CNN_model(cnn_input)

# plot event
output_path = f"predict/station_blind_Vs30_bias2closed_station_2016/{mask_after_sec} sec cnn feature map"
if not os.path.isdir(output_path):
    os.mkdir(output_path)
fig, ax = plt.subplots(25, 1, figsize=(14, 7))
for i in range(len(ax)):
    for j in range(3):
        ax[i].plot(waveform[i, :, j])
        ax[i].set_yticklabels("")
ax[0].set_title(f"EQ_ID: {eq_id} input waveform")
# fig.savefig(
#     f"{output_path}/eqid_{eq_id} {mask_after_sec} sec waveform input.png", dpi=300
# )
# plot single trace
for i in range(waveform.shape[0]):
    fig, ax = plt.subplots(3, 1, figsize=(14, 7))
    for j in range(len(ax)):
        ax[j].plot(waveform[i, :, j])
    ax[0].set_title(f"EQ_ID: {eq_id} input waveform{i+1},{input_station_list[i]}")
    # fig.savefig(
    #     f"{output_path}/eqid_{eq_id} {mask_after_sec} sec 3 channel waveform{i+1} input.png",
    #     dpi=300,
    # )

# plot convolution layer feature map (each layer each filter)
# for layer_num, tensor in enumerate(layer_output):  # convolution layer number
#     output_path = f"predict/station_blind_Vs30_bias2closed_station_2016/{mask_after_sec} sec cnn feature map//layer {layer_num+1}"
#     if not os.path.isdir(output_path):
#         os.mkdir(output_path)
#     print("layer_number", layer_num)
#     numeric_array = np.array(tensor.detach().cpu(), dtype=np.float32)
#     print(numeric_array.shape)
#     for i in range(numeric_array.shape[1]):  # filter number
#         fig, ax = plt.subplots(figsize=(10, 10))

#         feature_map = numeric_array[:, i, :]
#         if len(feature_map.shape) == 3:
#             feature_map = np.mean(feature_map, axis=2)
#         image = ax.imshow(feature_map, cmap="gray", aspect="auto")
#         ax.set_yticks(np.arange(0 - 0.5, feature_map.shape[0] + 0.5, 1), minor=True)
#         ax.grid(axis="y", linestyle="--", c="red", which="minor")
#         colorbar = plt.colorbar(image, ax=ax)
#         ax.set_title(f"Conv layer {layer_num+1}, filter {i}")
#         fig.savefig(f"{output_path}/Conv layer {layer_num+1}, filter {i}.png", dpi=300)
#         plt.close()

# plot cnn model output
output_path = f"predict/station_blind_Vs30_bias2closed_station_2016/{mask_after_sec} sec cnn feature map"
numeric_array = np.array(cnn_output.detach().cpu(), dtype=np.float32)
fig, ax = plt.subplots(figsize=(14, 7))
image = ax.imshow(numeric_array, cmap="gray", aspect="auto")
colorbar = plt.colorbar(image, ax=ax)
# fig.savefig(f"{output_path}/cnn output.png", dpi=300)


# plot convolution layer feature map (each layer)
for layer_num, tensor in enumerate(layer_output):  # convolution layer number
    output_path = f"predict/station_blind_Vs30_bias2closed_station_2016/{mask_after_sec} sec cnn feature map//layer {layer_num+1}"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    print("layer_number", layer_num)
    numeric_array = np.array(tensor.detach().cpu(), dtype=np.float32)
    feature_map = np.mean(numeric_array, axis=1)
    if len(feature_map.shape) == 3:
        feature_map = np.mean(feature_map, axis=2)
    print(feature_map.shape)
    fig, ax = plt.subplots(figsize=(10, 10))
    image = ax.imshow(feature_map, cmap="gray", aspect="auto")
    ax.set_yticks(np.arange(0 - 0.5, feature_map.shape[0] + 0.5, 1), minor=True)
    ax.grid(axis="y", linestyle="--", c="red", which="minor")
    colorbar = plt.colorbar(image, ax=ax)
    ax.set_title(f"Conv layer {layer_num+1}")
    # fig.savefig(
    #     f"{output_path}/Conv layer {layer_num+1}, average feature map.png", dpi=300
    # )

scale_factor_h = average_waveform.shape[0] / feature_map.shape[0]
scale_factor_w = average_waveform.shape[1] / feature_map.shape[1]

# zoom out feature map
resized_feature_map = zoom(feature_map, (scale_factor_h, scale_factor_w), order=3)
# order=3, Cubic Spline Interpolation
fig, ax = plt.subplots(figsize=(10, 10))
image = ax.imshow(resized_feature_map, cmap="Reds", aspect="auto")
# order=3, Cubic Spline Interpolation
ax.set_yticks(np.arange(0 - 0.5, feature_map.shape[0] + 0.5, 1), minor=True)
ax.grid(axis="y", linestyle="--", c="red", which="minor")
colorbar = plt.colorbar(image, ax=ax)
ax.set_title(f"Conv layer {layer_num+1}")
output_path = f"predict/station_blind_Vs30_bias2closed_station_2016/{mask_after_sec} sec cnn feature map/"
# fig.savefig(f"{output_path}/Conv layer {layer_num+1}, average feature map.png", dpi=300)

# plot event with feature
fig, ax = plt.subplots(25, 1, figsize=(14, 7))
for i in range(len(ax)):
    ax[i].plot(average_waveform[i])
    ax[i].imshow(
        np.expand_dims(resized_feature_map[i], axis=0),
        cmap="Reds",
        aspect="auto",
        vmin=resized_feature_map.min(),
        vmax=resized_feature_map.max(),
    )
    ax[i].set_yticklabels("")
ax[0].set_title(f"EQ_ID: {eq_id} input waveform with last conv layer feature")
ax[-1].set_xlabel("time sample (200Hz)")
# fig.savefig(
#     f"{output_path}/eqid_{eq_id} event input with last conv layer feature.png", dpi=300
# )

# each average waveform plot and feature
for i in range(average_waveform.shape[0]):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(average_waveform[i])
    im = ax.imshow(
        np.expand_dims(resized_feature_map[i], axis=0),
        cmap="Reds",
        aspect="auto",
        extent=[
            0,
            average_waveform.shape[1],
            average_waveform[i].min(),
            average_waveform[i].max(),
        ],
        vmin=resized_feature_map.min(),
        vmax=resized_feature_map.max(),
    )
    ax.set_title(
        f"EQ_ID: {eq_id} input waveform{i+1}, station_name:{input_station_list[i]}"
    )
    cbar = plt.colorbar(im, ax=ax, pad=0.01, orientation="vertical")
    cbar.set_label("conv output value")
    # fig.savefig(
    #     f"{output_path}/eqid_{eq_id} waveform input{i+1} with last conv layer feature.png",
    #     dpi=300,
    # )

# each channel waveform with feature
for i in range(waveform.shape[0]):
    fig, ax = plt.subplots(3, 1, figsize=(14, 7))
    for j in range(len(ax)):
        ax[j].imshow(
            np.expand_dims(resized_feature_map[i], axis=0),
            cmap="Reds",
            aspect="auto",
            vmin=resized_feature_map.min(),
            vmax=resized_feature_map.max(),
        )
        ax[j].plot(waveform[i, :, j])
        ax[j].set_ylim(waveform[i].min() - 0.001, waveform[i].max() + 0.001)
    ax[0].set_title(
        f"EQ_ID: {eq_id} input waveform{i+1}, station_name:{input_station_list[i]}"
    )
    cbar = plt.colorbar(im, ax=ax, pad=0.01, orientation="vertical")
    cbar.set_label("conv output value")
    # fig.savefig(
    #     f"{output_path}/eqid_{eq_id} {mask_after_sec} sec 3 channel waveform{i+1} with feature.png",
    #     dpi=300,
    # )


# each channel waveform with feature
def normalize_to_minus_one_to_one(arr):
    # 找到数组的最大值和最小值
    min_val = arr.min()
    max_val = arr.max()

    # 将数组线性缩放到-1到1之间
    normalized_arr = -1 + 2 * (arr - min_val) / (max_val - min_val)

    return normalized_arr


def normalize_to_zero_one(arr):
    # 找到数组的最小值和最大值
    min_val = arr.min()
    max_val = arr.max()

    # 将数组线性缩放到0到1之间
    normalized_arr = (arr - min_val) / (max_val - min_val)

    return normalized_arr


x_pos = 0.1
y_pos = 0.8
for i in range(not_padding_station_number):
    fig, ax = plt.subplots(3, 1, figsize=(14, 7))
    for j in range(len(ax)):
        correlation_starttime = p_picks[i] - sample_rate
        correlation_endtime = p_picks[0] + (mask_after_sec + 1) * sample_rate
        if mask_after_sec == 10:
            correlation_endtime = p_picks[0] + (mask_after_sec) * sample_rate
        correlation = np.corrcoef(
            abs(waveform)[i, correlation_starttime:correlation_endtime, j],
            resized_feature_map[i, correlation_starttime:correlation_endtime],
        )[0, 1]
        ax[j].plot(
            normalize_to_zero_one(abs(waveform[i, :, j])),
            alpha=0.7,
            label="input waveform",
        )
        ax[j].plot(
            normalize_to_zero_one(resized_feature_map[i]), c="red", label="feature map"
        )
        ax[j].axvline(x=correlation_starttime, color="grey", linestyle="--")
        ax[j].axvline(x=correlation_endtime, color="grey", linestyle="--")
        ax[j].text(
            x_pos,
            y_pos,
            f"correlation: {np.round(correlation, 2)}",
            transform=ax[j].transAxes,
            fontsize=15,
            horizontalalignment="left",
            verticalalignment="top",
        )
    ax[0].legend()
    # ax[j].set_ylim(waveform[i].min()-0.001,waveform[i].max()+0.001)
    ax[0].set_title(
        f"EQ_ID: {eq_id} input waveform{i+1}, station_name:{input_station_list[i]}",
        fontsize=20,
    )
    ax[1].set_ylabel("normalized acc")
    ax[-1].set_xlabel("time sample")
    # fig.savefig(
    #     f"{output_path}/eqid_{eq_id} {mask_after_sec} sec 3 channel abs(waveform){i+1} with feature.png",
    #     dpi=300,
    # )

from scipy.stats import pearsonr


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


# 歐基里德平均
euclidean_correlation_list = []
max_correlation_list=[]
max_delay_list=[]
euclidean_waveform = np.linalg.norm(waveform, axis=2) / np.sqrt(3)
for i in range(not_padding_station_number):
    fig, ax = plt.subplots(3, 1, figsize=(14, 7))
    correlation_starttime = p_picks[i] - sample_rate
    correlation_endtime = p_picks[0] + (mask_after_sec + 1) * sample_rate
    if mask_after_sec == 10:
        correlation_endtime = p_picks[0] + (mask_after_sec) * sample_rate
    ax[0].plot(normalize_to_zero_one(euclidean_waveform[i]), alpha=0.7)
    ax[1].plot(normalize_to_zero_one(resized_feature_map[i]), c="red")
    ax[2].plot(
        normalize_to_zero_one(euclidean_waveform[i]),
        alpha=0.7,
        label="euclidean_waveform",
    )
    ax[2].plot(
        normalize_to_zero_one(resized_feature_map[i]), c="red", label="feature map"
    )
    correlation = np.corrcoef(
        euclidean_waveform[i, correlation_starttime:correlation_endtime],
        resized_feature_map[i, correlation_starttime:correlation_endtime],
    )[0, 1]
    euclidean_correlation_list.append(correlation)
    for j in range(len(ax)):
        ax[j].axvline(x=correlation_starttime, color="grey", linestyle="--")
        ax[j].axvline(x=correlation_endtime, color="grey", linestyle="--")
    ax[2].text(
        x_pos,
        y_pos,
        f"correlation: {np.round(correlation, 2)}",
        transform=ax[j].transAxes,
        fontsize=15,
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax[0].set_title(
        f"EQ_ID: {eq_id} euclidean_waveform{i+1}, station_name:{input_station_list[i]}"
    )
    ax[1].set_ylabel("normalized acc")
    ax[-1].set_xlabel("time sample")
    ax[2].legend()

    # fig.savefig(
    #     f"{output_path}/eqid_{eq_id} {mask_after_sec} sec 3 channel euclidean_waveform{i+1} with feature.png",
    #     dpi=300,
    # )
    #============TLCC======================
    delay_values, tlcc_values = calculate_tlcc(
        euclidean_waveform[i, correlation_starttime:correlation_endtime],
        resized_feature_map[i, correlation_starttime:correlation_endtime],
        max_delay=100,
    )
    max_index = np.argmax(tlcc_values)
    max_correlation=tlcc_values[max_index]
    max_delay=delay_values[max_index]
    max_correlation_list.append(max_correlation)
    max_delay_list.append(max_delay)
    #plot TLCC
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(delay_values, tlcc_values)

    ax.set_xlabel("Time Lag")
    ax.set_ylabel("TLCC (Pearson Correlation) Value")
    ax.set_title(f"euclidean_waveform{i+1} Time Lagged Cross-Correlation (TLCC) Analysis")
    ax.grid(True)
    # fig.savefig(
    #     f"{output_path}/eqid_{eq_id} {mask_after_sec} sec euclidean_waveform{i+1} Time Lagged Cross-Correlation (TLCC) Analysis.png",
    #     dpi=300,
    # )
#plot max correlation with time delay
mean=np.round(np.array(max_correlation_list).mean(),2)
std=np.round(np.array(max_correlation_list).std(),2)
fig, ax = plt.subplots()
ax.scatter(max_delay_list,max_correlation_list)

ax.set_xlabel("time sample lag")
ax.set_ylabel("max Pearson correlation")
ax.set_title(f"{mask_after_sec} sec EQID_{eq_id} max correlation (TLCC) with delay time\nmean :{mean}, std: {std}")
ax.grid(True)
# fig.savefig(
#     f"{output_path}/{mask_after_sec} sec EQID{eq_id} TLCC max correlation delay time.png",
#     dpi=300,
# )
np.array(euclidean_correlation_list).mean()

# envelope_euclidean waveform
envelope_euclidean_correlation_list = []
from scipy.signal import hilbert, detrend

for i in range(not_padding_station_number):
    detrended_data = detrend(euclidean_waveform)
    analytic_signal = hilbert(detrended_data)
    amplitude = np.abs(analytic_signal)
    fig, ax = plt.subplots(3, 1, figsize=(14, 7))
    correlation_starttime = p_picks[i] - sample_rate
    correlation_endtime = p_picks[0] + (mask_after_sec + 1) * sample_rate
    if mask_after_sec == 10:
        correlation_endtime = p_picks[0] + (mask_after_sec) * sample_rate
    ax[0].plot(normalize_to_zero_one(amplitude[i]), alpha=0.7)
    ax[1].plot(normalize_to_zero_one(resized_feature_map[i]), c="red")
    ax[2].plot(
        normalize_to_zero_one(amplitude[i]),
        alpha=0.7,
        label="envelope_euclidean_waveform",
    )
    ax[2].plot(
        normalize_to_zero_one(resized_feature_map[i]), c="red", label="feature map"
    )
    correlation = np.corrcoef(
        amplitude[i, correlation_starttime:correlation_endtime],
        resized_feature_map[i, correlation_starttime:correlation_endtime],
    )[0, 1]
    envelope_euclidean_correlation_list.append(correlation)
    for j in range(len(ax)):
        ax[j].axvline(x=correlation_starttime, color="grey", linestyle="--")
        ax[j].axvline(x=correlation_endtime, color="grey", linestyle="--")
    ax[2].text(
        x_pos,
        y_pos,
        f"correlation: {np.round(correlation, 2)}",
        transform=ax[j].transAxes,
        fontsize=15,
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax[0].set_title(
        f"EQ_ID: {eq_id} envelope_euclidean_waveform{i+1}, station_name:{input_station_list[i]}"
    )
    ax[1].set_ylabel("normalized acc")
    ax[-1].set_xlabel("time sample")
    ax[2].legend()
    # fig.savefig(
    #     f"{output_path}/eqid_{eq_id} {mask_after_sec} sec 3 channel envelope_euclidean_waveform{i+1} with feature.png",
    #     dpi=300,
    # )
np.array(envelope_euclidean_correlation_list).mean()

# ===========load transformer parameter==============
transformer_parameter = {}
for name, param in full_model_parameter.items():
    if (
        "model_Transformer" in name
    ):  # model_CNN.conv2d1.0.weight : conv2d1.0.weight didn't match
        name = name.replace("model_Transformer.", "")
        transformer_parameter[name] = param
transformer_model.load_state_dict(transformer_parameter)
