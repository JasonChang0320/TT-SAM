import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from CNN_Transformer_Mixtureoutput_TEAM import (
    CNN,
    MDN,
    MLP,
    PositionEmbedding,
    TransformerEncoder,
    full_model,
)
from multiple_sta_dataset import multiple_station_dataset, multiple_station_dataset_new
from plot_predict_map import true_predicted

mask_after_sec = 7
label = "pgv"
data = multiple_station_dataset_new(
    "D:/TEAM_TSMIP/data/TSMIP_filtered.hdf5",
    mode="test",
    mask_waveform_sec=mask_after_sec,
    test_year=2016,
    label_key=label,
    mag_threshold=0,
    input_type="dis",
)
# =========================
device = torch.device("cuda")
for num in range(1, 11):  # [1,3,18,20]
    path = f"./model/model{num}.pt"
    emb_dim = 150
    mlp_dims = (150, 100, 50, 30, 10)
    CNN_model = CNN().cuda()
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
        label_time = sample[f"{label}_time"].flatten().numpy().tolist()
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
        weight, sigma, mu = full_Model(sample)

        weight = weight.cpu()
        sigma = sigma.cpu()
        mu = mu.cpu()
        if j == 0:
            Mixture_mu = torch.sum(weight * mu, dim=2).cpu().detach().numpy()
            Label = sample["label"].cpu().detach().numpy()
        else:
            Mixture_mu = np.concatenate(
                [Mixture_mu, torch.sum(weight * mu, dim=2).cpu().detach().numpy()],
                axis=1,
            )
            Label = np.concatenate(
                [Label, sample["label"].cpu().detach().numpy()], axis=1
            )
    Label = Label.flatten()
    Mixture_mu = Mixture_mu.flatten()

    output = {
        "EQ_ID": EQ_ID,
        "p_picks": P_picks,
        f"{label}_time": Label_time,
        "predict": Mixture_mu,
        "answer": Label,
        "latitude": Lat,
        "longitude": Lon,
        "elevation": Elev,
    }
    output_df = pd.DataFrame(output)
    output_df = output_df[output_df["answer"] != 0]
    output_df.to_csv(
        f"./predict/model {num} {mask_after_sec} sec prediction.csv", index=False
    )
    fig, ax = true_predicted(
        y_true=output_df["answer"],
        y_pred=output_df["predict"],
        time=mask_after_sec,
        quantile=False,
        agg="point",
        point_size=12,
        target=label,
    )

    fig.savefig(f"./predict/model{num} {mask_after_sec} sec.png")
# input_waveform_picks=np.array(data[31][4])[np.array(data[31][4])<np.array(data[31][4])[0]+mask_after_sec*200]
# wav_fig,ax=plt.subplots(len(input_waveform_picks),1,figsize=(14,7))
# for i in range(0,len(input_waveform_picks)):
#     for k in range(0,3):
#         ax[i].plot(data[31][0][i,:,k].flatten())
#         ax[i].set_yticklabels("")
#     ax[i].axvline(x=input_waveform_picks[i],c="r")
# ax[0].set_title(f"{int(sample[-1])}input")

# fig=true_predicted(y_true=output_df["answer"][output_df["EQ_ID"]==27558],y_pred=output_df["predict"][output_df["EQ_ID"]==27558],
#                 time=mask_after_sec,quantile=False,agg="point", point_size=12)

# ensemble model prediction
mask_after_sec = 10

data8 = pd.read_csv(
    f"predict/dis random sec predict pgv test 2016/model 8 {mask_after_sec} sec prediction.csv"
)
data12 = pd.read_csv(
    f"predict/dis random sec predict pgv test 2016/model 12 {mask_after_sec} sec prediction.csv"
)


output_df = (data8 + data12) / 2
fig, ax = true_predicted(
    y_true=output_df["answer"],
    y_pred=output_df["predict"],
    time=mask_after_sec,
    quantile=False,
    agg="point",
    point_size=12,
    target=label,
)

output_df.to_csv(
    f"./predict/model8 12 {mask_after_sec} sec prediction.csv", index=False
)

fig.savefig(f"./predict/model8 12 {mask_after_sec} sec.png")

# plot each events prediction
for eq_id in data.event_metadata["EQ_ID"]:
    fig, ax = true_predicted(
        y_true=output_df["answer"][output_df["EQ_ID"] == eq_id],
        y_pred=output_df["predict"][output_df["EQ_ID"] == eq_id],
        time=mask_after_sec,
        quantile=False,
        agg="point",
        point_size=70,
        target=label,
    )
    magnitude = data.event_metadata[data.event_metadata["EQ_ID"] == eq_id][
        "magnitude"
    ].values[0]
    ax.set_title(
        f"{mask_after_sec}s True Predict Plot, EQ ID:{eq_id}, magnitude: {magnitude}",
        fontsize=20,
    )
    plt.close()
    fig.savefig(
        f"./predict/dis random sec predict pgv test 2016/ok model prediction/updated dataset plot each event {mask_after_sec} sec/EQ ID_{eq_id} magnitude_{magnitude}.png"
    )
