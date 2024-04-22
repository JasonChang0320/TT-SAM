import matplotlib.pyplot as plt

plt.subplots()
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

sys.path.append("..")
from model.CNN_Transformer_Mixtureoutput_TEAM import CNN_feature_map
from data.multiple_sta_dataset import multiple_station_dataset

mask_after_sec = 5
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
data_path = "../predict/station_blind_Vs30_bias2closed_station_2016"
predict = pd.read_csv(f"{data_path}/{mask_after_sec} sec model11 with all info.csv")
single_event_prediction = predict.query(f"EQ_ID=={eq_id}")
# ===========prepare model==============
device = torch.device("cuda")
num = 11
model_path = f"../model/model{num}.pt"
CNN_model = CNN_feature_map(mlp_input=5665).cuda()

# ===========load CNN parameter==============
full_model_parameter = torch.load(model_path)
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

# plot convolution layer feature map (each layer)
for layer_num, tensor in enumerate(layer_output):  # convolution layer number
    output_path = f"../predict/station_blind_Vs30_bias2closed_station_2016/{mask_after_sec} sec cnn feature map//layer {layer_num+1}"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    print("layer_number", layer_num)
    numeric_array = np.array(tensor.detach().cpu(), dtype=np.float32)
    feature_map = np.mean(numeric_array, axis=1)
    if len(feature_map.shape) == 3:
        feature_map = np.mean(feature_map, axis=2)
    print(feature_map.shape)
    fig, ax = plt.subplots(figsize=(10, 10))
    image = ax.imshow(feature_map, cmap="Reds", aspect="auto")
    ax.set_yticks(np.arange(0 - 0.5, feature_map.shape[0] + 0.5, 1), minor=True)
    ax.grid(axis="y", linestyle="--", c="red", which="minor")
    colorbar = plt.colorbar(image, ax=ax)
    ax.set_title(f"Conv layer {layer_num+1}")
    # fig.savefig(
    #     f"{output_path}/Conv layer {layer_num+1}, average feature map.png", dpi=300
    # )
