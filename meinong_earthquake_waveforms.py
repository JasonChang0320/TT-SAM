import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from CNN_Transformer_Mixtureoutput_TEAM import (
    CNN,
    MDN,
    MLP,
    PositionEmbedding,
    TransformerEncoder,
    full_model,
)
from multiple_sta_dataset import multiple_station_dataset_outputs
from plot_predict_map import true_predicted

mask_after_sec = 3
label = ["pga"]
data = multiple_station_dataset_outputs(
    "D:/TEAM_TSMIP/data/TSMIP_filtered.hdf5",
    mode="test",
    mask_waveform_sec=mask_after_sec,
    test_year=2016,
    label_keys=label,
    input_type=["acc", "vel"],
    data_length_sec=10,
)
record_prediction = pd.read_csv(f"./predict/{mask_after_sec}_sec_meinong_eq_record_prediction.csv")
# =========================
device = torch.device("cuda")
num = 37
path = f"./model/model{num}.pt"
emb_dim = 150
mlp_dims = (150, 100, 50, 30, 10)
CNN_model = CNN(mlp_input=3665).cuda()
pos_emb_model = PositionEmbedding(emb_dim=emb_dim).cuda()
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
    data_length=2000,
).to(device)
full_Model.load_state_dict(torch.load(path))
loader = DataLoader(dataset=data, batch_size=1)

Mixture_mu = []
Label = []
P_picks = []
EQ_ID = []
Label_time = []
Sta_name = []
Lat = []
Lon = []
Elev = []
for j, sample in tqdm(enumerate(loader)):
    picks = sample["p_picks"].flatten().numpy().tolist()
    label_time = sample[f"{label[0]}_time"].flatten().numpy().tolist()
    lat = sample["target"][:, :, 0].flatten().tolist()
    lon = sample["target"][:, :, 1].flatten().tolist()
    elev = sample["target"][:, :, 2].flatten().tolist()
    P_picks.extend(picks)
    P_picks.extend([np.nan] * (25 - len(picks)))
    Label_time.extend(label_time)
    Label_time.extend([np.nan] * (25 - len(label_time)))
    Lat.extend(lat)
    Lon.extend(lon)
    Elev.extend(elev)

    eq_id = sample["EQ_ID"][:, :, 0].flatten().numpy().tolist()
    EQ_ID.extend(eq_id)
    EQ_ID.extend([np.nan] * (25 - len(eq_id)))
    if eq_id[0] == 24784:
        waveform = sample["acc"].numpy().reshape(25, 2000, 3)
        waveform_num = len(
            np.where(np.array(picks) <= picks[0] + (mask_after_sec * 200))[0]
        )
        waveforms_fig, waveforms_ax = plt.subplots(
            waveform_num,
            1,
            figsize=(14, 7),
        )
        for i in range(waveform_num):
            station_name = record_prediction["station_name"][i]
            answer = np.round(100 * (10 ** record_prediction["answer"][i]), 2)
            waveforms_ax[i].plot(waveform[i, :, 2])
            waveforms_ax[i].axvline(x=picks[i], c="r")
            waveforms_ax[i].set_yticklabels("")
            waveforms_ax[i].text(
                -0.05,
                0.5,
                f"{station_name}",
                fontsize=14,
                transform=waveforms_ax[i].transAxes,
                ha="right",
                va="center",
            )
            waveforms_ax[i].text(
                1.05,
                0.5,
                f"PGA: {answer} gal",
                fontsize=14,
                transform=waveforms_ax[i].transAxes,
                ha="left",
                va="center",
            )
            if mask_after_sec==5: 
                waveform_fig,waveform_ax=plt.subplots(3,1,figsize=(14,7))
                for j in range(3):
                    waveform_ax[j].plot(waveform[i,:,j])
                    waveform_ax[j].axvline(x=picks[i],c="r")
                waveform_ax[0].set_title(f"acc waveform, station: {station_name}, PGA: {answer} gal",size=20)
                waveform_fig.savefig(f"./predict/meinong earthquake/index{i}_{station_name}_acc_input.png")
        waveforms_ax[0].set_title(f"Meinong earthquake {mask_after_sec} sec acc records, H2 component",size=20)
        # waveforms_fig.savefig(f"./predict/meinong earthquake/{mask_after_sec}_sec_H2_acc.png",bbox_inches='tight')
        break
