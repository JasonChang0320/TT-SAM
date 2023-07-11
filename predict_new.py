import h5py
import matplotlib.pyplot as plt
plt.subplots()
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

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

mask_after_sec = 5
label = "pga"
data = multiple_station_dataset(
    "D:/TEAM_TSMIP/data/TSMIP_2009_2019.hdf5",
    mode="test",
    mask_waveform_sec=mask_after_sec,
    test_year=2016,
    label_key=label,
    mag_threshold=0,
    input_type="acc",
    data_length_sec=15,
)
# =========================
device = torch.device("cuda")
for num in range(1,9):
    path = f"./model/model{num}.pt"
    emb_dim = 150
    mlp_dims = (150, 100, 50, 30, 10)
    CNN_model = CNN(mlp_input=5665).cuda()
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
        data_length=3000,
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
    eq_id = 24784
    ax.scatter(
        output_df["answer"][output_df["EQ_ID"] == eq_id],
        output_df["predict"][output_df["EQ_ID"] == eq_id],
        c="r",
    )
    magnitude = data.event_metadata[data.event_metadata["EQ_ID"] == eq_id][
        "magnitude"
    ].values[0]
    ax.set_title(
        f"{mask_after_sec}s True Predict Plot, EQ ID:{eq_id}, magnitude: {magnitude}",
        fontsize=20,
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
mask_after_sec=5

predict2=pd.read_csv(
    f"./predict/model 2 {mask_after_sec} sec prediction.csv"
)
predict3=pd.read_csv(
    f"./predict/model 3 {mask_after_sec} sec prediction.csv"
)
predict7=pd.read_csv(
    f"./predict/model 7 {mask_after_sec} sec prediction.csv"
)
predict8=pd.read_csv(
    f"./predict/model 8 {mask_after_sec} sec prediction.csv"
)
big_event_output = pd.read_csv(
    f"./predict/acc predict pga mag bigger than 5/model 1 {mask_after_sec} sec prediction.csv"
)
big_event_output=big_event_output.drop([1113])
big_event_output.reset_index(drop=True,inplace=True)

output_df1 = pd.read_csv(
    f"./predict/acc predict pga mag bigger than 5 double check/model 8 {mask_after_sec} sec prediction.csv"
)

origin_output=pd.read_csv(
    f"./predict/acc predict pga 1999_2019/model 2 {mask_after_sec} sec prediction.csv"
)

ensemble_predict=(big_event_output+2*origin_output)/3
fig, ax = true_predicted(
    y_true=ensemble_predict["answer"],
    y_pred=ensemble_predict["predict"],
    time=mask_after_sec,
    quantile=False,
    agg="point",
    point_size=12,
    target=label,
)
ax.scatter(
    ensemble_predict["answer"][ensemble_predict["EQ_ID"] == eq_id],
    ensemble_predict["predict"][ensemble_predict["EQ_ID"] == eq_id],
    c="r",
)
ensemble_predict.to_csv(f"./predict/{mask_after_sec} sec ensemble (origin & big event model).csv",index=False)

# plot residual
mask_after_sec = 10
output_df = pd.read_csv(
    f"./predict/acc predict pga 1999_2019/model 2 {mask_after_sec} sec prediction.csv"
)
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(ensemble_predict["answer"], ensemble_predict["predict"] - ensemble_predict["answer"], alpha=0.3)
label = ["2", "3", "4", "5-", "5+", "6-", "6+", "7"]
pga_threshold = np.log10([0.025, 0.080, 0.250, 0.80, 1.4, 2.5, 4.4, 8.0, 10])
for i in range(len(pga_threshold) - 1):
    ax.text((pga_threshold[i] + pga_threshold[i + 1]) / 2, -2, label[i])
ax.vlines(pga_threshold[1:-1], -2, 1, linestyles="dotted", color="k")
ax.text(
    -0.25,
    1.2,
    f"log std: {np.round((ensemble_predict['predict'] - ensemble_predict['answer']).std(),3)}",fontsize=12
)
ax.set_xlabel("true pga log(m/s^2)",fontsize=12)
ax.set_ylabel("log residual",fontsize=12)
ax.set_title(f"{mask_after_sec} sec predict residual (predict-answer)",fontsize=15)
# plot each events prediction
for eq_id in data.event_metadata["EQ_ID"]:
    # if eq_id==24784:
    # fig, ax = true_predicted(
    #     y_true=output_df["answer"],
    #     y_pred=output_df["predict"],
    #     time=mask_after_sec,
    #     quantile=False,
    #     agg="point",
    #     point_size=12,
    #     target=label,
    # )
    # ax.scatter(output_df["answer"][output_df["EQ_ID"] == eq_id],
    #         output_df["predict"][output_df["EQ_ID"] == eq_id],
    #         c="r")
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
    folder_path = f"./predict/acc predict pga 1999_2019/each event predict/{eq_id}_magnitude_{int(magnitude)}"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    fig.savefig(f"{folder_path}/{mask_after_sec}sec_EQID.png")

#check prediction in magnitude
mask_after_sec=7
output_df = pd.read_csv(
    f"./predict/acc predict pga mag bigger than 5/model 1 {mask_after_sec} sec prediction.csv"
)
Afile_path = "data preprocess/events_traces_catalog"

catalog = pd.read_csv(f"{Afile_path}/1999_2019_final_catalog.csv")
label="pga"
merged_df=pd.merge(output_df,catalog[["EQ_ID","magnitude"]],how="left",on="EQ_ID")
fig, ax = true_predicted(
    y_true=merged_df["answer"][merged_df["magnitude"] >= 5],
    y_pred=merged_df["predict"][merged_df["magnitude"] >= 5],
    time=mask_after_sec,
    quantile=False,
    agg="point",
    point_size=20,
    target=label,
)

ax.scatter(
    merged_df["answer"][merged_df["magnitude"] < 5],
    merged_df["predict"][merged_df["magnitude"] < 5],
    c="r",
)

fig,ax =plt.subplots()

ax.hist(catalog.query("magnitude > 5 & year !=2016")["magnitude"],bins=10,edgecolor="grey",label="train & val")
ax.hist(catalog.query("year == 2016")["magnitude"],bins=16,edgecolor="grey",label="test")
ax.legend()
ax.set_title("magnitude >= 5 case dataset")

#add part small event into mag>=5 dataset
add_small_event=catalog.query(f"magnitude < 5 & year!=2016").sample(frac=0.25,random_state=0)
big_event=catalog.query(f"magnitude >= 5 & year!=2016")
test_event=catalog.query(f"year==2016")

train_event=pd.concat([big_event,add_small_event])

fig,ax =plt.subplots()
ax.hist(train_event["magnitude"],bins=10,edgecolor="grey",label="train & val")
ax.hist(test_event["magnitude"],bins=16,edgecolor="grey",label="test")
ax.legend()
ax.set_title("magnitude >= 5 case dataset + 25% magnitude < 5")

trace= pd.read_csv(f"{Afile_path}/1999_2019_final_traces.csv")

big_event_trace=trace[trace["EQ_ID"].isin(big_event["EQ_ID"])]
small_event_trace=trace[trace["EQ_ID"].isin(add_small_event["EQ_ID"])]
train_trace=trace[trace["EQ_ID"].isin(train_event["EQ_ID"])]

plt.hist(big_event_trace["pga"],bins=32,edgecolor="grey")
plt.hist(train_trace["pga"],bins=32,edgecolor="grey")
plt.hist(small_event_trace["pga"],bins=32,edgecolor="grey")
