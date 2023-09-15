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

mask_after_sec = 3
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

cnn_input = torch.DoubleTensor(sample["waveform"].reshape(-1, 3000, 3)).float().cuda()
cnn_output, layer_output = CNN_model(cnn_input)

waveform = sample["waveform"].numpy().reshape(25, 3000, 3)
output_path = f"predict/station_blind_Vs30_bias2closed_station_2016/{mask_after_sec} sec cnn feature map"
if not os.path.isdir(output_path):
    os.mkdir(output_path)
fig, ax = plt.subplots(25, 1, figsize=(14, 7))
for i in range(len(ax)):
    for j in range(3):
        ax[i].plot(waveform[i, :, j])
        ax[i].set_yticklabels("")
ax[0].set_title(f"EQ_ID: {eq_id} input waveform")
fig.savefig(
    f"{output_path}/eqid_{eq_id} {mask_after_sec} sec waveform input.png", dpi=300
)

# plot convolution layer feature map (each layer each filter)
for layer_num, tensor in enumerate(layer_output):  # convolution layer number
    output_path = f"predict/station_blind_Vs30_bias2closed_station_2016/{mask_after_sec} sec cnn feature map//layer {layer_num+1}"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    print("layer_number", layer_num)
    numeric_array = np.array(tensor.detach().cpu(), dtype=np.float32)
    print(numeric_array.shape)
    for i in range(numeric_array.shape[1]):  # filter number
        fig, ax = plt.subplots(figsize=(10, 10))

        feature_map = numeric_array[:, i, :]
        if len(feature_map.shape) == 3:
            feature_map = np.mean(feature_map, axis=2)
        image = ax.imshow(feature_map, cmap="gray", aspect="auto")
        ax.set_yticks(np.arange(0 - 0.5, feature_map.shape[0] + 0.5, 1), minor=True)
        ax.grid(axis="y", linestyle="--", c="red", which="minor")
        colorbar = plt.colorbar(image, ax=ax)
        ax.set_title(f"Conv layer {layer_num+1}, filter {i}")
        fig.savefig(f"{output_path}/Conv layer {layer_num+1}, filter {i}.png", dpi=300)
        plt.close()

# plot cnn model output
output_path = f"predict/station_blind_Vs30_bias2closed_station_2016/{mask_after_sec} sec cnn feature map"
numeric_array = np.array(cnn_output.detach().cpu(), dtype=np.float32)
fig, ax = plt.subplots(figsize=(14, 7))
image = ax.imshow(numeric_array, cmap="gray", aspect="auto")
colorbar = plt.colorbar(image, ax=ax)
fig.savefig(f"{output_path}/cnn output.png", dpi=300)

# waveform average
waveform = sample["waveform"].numpy().reshape(25, 3000, 3)
average_waveform = np.mean(waveform, axis=2)

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
    fig.savefig(
        f"{output_path}/Conv layer {layer_num+1}, average feature map.png", dpi=300
    )

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
fig.savefig(
        f"{output_path}/Conv layer {layer_num+1}, average feature map.png", dpi=300
    )

#plot event with feature
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
fig.savefig(
        f"{output_path}/eqid_{eq_id} event input with last conv layer feature.png", dpi=300
    )

# each waveform plot and feature

for i in range(average_waveform.shape[0]):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(average_waveform[i])
    im=ax.imshow(
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
    ax.set_title(f"EQ_ID: {eq_id} input waveform{i+1}")
    cbar = plt.colorbar(im, ax=ax, pad=0.01, orientation="vertical")
    cbar.set_label("conv output value")
    fig.savefig(
        f"{output_path}/eqid_{eq_id} waveform input{i+1} with last conv layer feature.png", dpi=300
    )




# ===========load transformer parameter==============
transformer_parameter = {}
for name, param in full_model_parameter.items():
    if (
        "model_Transformer" in name
    ):  # model_CNN.conv2d1.0.weight : conv2d1.0.weight didn't match
        name = name.replace("model_Transformer.", "")
        transformer_parameter[name] = param
transformer_model.load_state_dict(transformer_parameter)
