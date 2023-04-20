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
from multiple_sta_dataset import multiple_station_dataset
from plot_predict_map import true_predicted

mask_after_sec = 3
label = "pga"
data = multiple_station_dataset(
    "D:/TEAM_TSMIP/data/TSMIP_filtered.hdf5",
    mode="test",
    mask_waveform_sec=mask_after_sec,
    test_year=2016,
    label_key=label,
    mag_threshold=0,
    input_type="acc",
    data_length_sec=30,
)
# =========================
device = torch.device("cuda")
for num in [26]:  # [4,6,11,13,14,15,19,21,26,28,30]
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
        data_length=6000,
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
    #     if eq_id[0]==24784:
    #         waveform=sample["waveform"].numpy().reshape(25,6000,3)
    #         fig,ax=plt.subplots(25,1,figsize=(14,7))
    #         for i in range(waveform.shape[0]):
    #             ax[i].plot(waveform[i,:,:])
    #             fig_1,ax_1=plt.subplots(figsize=(14,7))
    #             ax_1.plot(waveform[i,:,:])
    #             ax_1.set_title(f"index:{i}")
    #             fig_1.savefig(f"./predict/eq_id_{eq_id[0]}_predict/{mask_after_sec}_sec_model input index_{i}")
    #         fig.savefig(f"./predict/eq_id_{eq_id[0]}_predict/{mask_after_sec}_sec_model inputs")
    #         break
    # break
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
    # output_df.to_csv(
    #     f"./predict/model {num} {mask_after_sec} sec prediction.csv", index=False
    # )
    fig, ax = true_predicted(
        y_true=output_df["answer"],
        y_pred=output_df["predict"],
        time=mask_after_sec,
        quantile=False,
        agg="point",
        point_size=12,
        target=label,
    )
    eq_id=24784
    ax.scatter(output_df["answer"][output_df["EQ_ID"] == eq_id],
        output_df["predict"][output_df["EQ_ID"] == eq_id],
        c="r")
    magnitude = data.event_metadata[data.event_metadata["EQ_ID"] == eq_id][
        "magnitude"
    ].values[0]
    ax.set_title(
        f"{mask_after_sec}s True Predict Plot, EQ ID:{eq_id}, magnitude: {magnitude}",
        fontsize=20,
    )

    # fig.savefig(f"./predict/model{num} {mask_after_sec} sec.png")
    
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

data4 = pd.read_csv(
    f"predict/model 4 {mask_after_sec} sec prediction.csv"
)
# data4 = pd.read_csv(
#     f"predict/model 4 {mask_after_sec} sec prediction.csv"
# )
data26 = pd.read_csv(
    f"predict/model 6 {mask_after_sec} sec prediction.csv"
)


output_df = (data4+data26)/2
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
    f"./predict/model4 26 {mask_after_sec} sec prediction.csv", index=False
)

fig.savefig(f"./predict/model4 26 {mask_after_sec} sec.png")

# mask_after_sec = 7
# for i in [4,6,11,13,14,15,19,21,26,28,30]:
#     output_df= pd.read_csv(
#         f"predict/model {i} {mask_after_sec} sec prediction.csv"
#     )
    # plot each events prediction
for eq_id in data.event_metadata["EQ_ID"]:
    if eq_id==24784:
        fig, ax = true_predicted(
            y_true=output_df["answer"],
            y_pred=output_df["predict"],
            time=mask_after_sec,
            quantile=False,
            agg="point",
            point_size=12,
            target=label,
        )
        ax.scatter(output_df["answer"][output_df["EQ_ID"] == eq_id],
                output_df["predict"][output_df["EQ_ID"] == eq_id],
                c="r")
        # fig, ax = true_predicted(
        #     y_true=output_df["answer"][output_df["EQ_ID"] == eq_id],
        #     y_pred=output_df["predict"][output_df["EQ_ID"] == eq_id],
        #     time=mask_after_sec,
        #     quantile=False,
        #     agg="point",
        #     point_size=70,
        #     target=label,
        # )
        magnitude = data.event_metadata[data.event_metadata["EQ_ID"] == eq_id][
            "magnitude"
        ].values[0]
        ax.set_title(
            f"{mask_after_sec}s True Predict Plot, EQ ID:{eq_id}, magnitude: {magnitude}",
            fontsize=20,
        )
        plt.close()
        fig.savefig(
            f"./predict/eq_id_{eq_id}_predict/model4 26_{mask_after_sec}_EQID_{eq_id} magnitude_{magnitude}.png"
        )
