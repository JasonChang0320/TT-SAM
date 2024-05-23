import h5py
import matplotlib.pyplot as plt

plt.subplots()
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append("..")
from model.CNN_Transformer_Mixtureoutput_TEAM import (
    CNN,
    MDN,
    MLP,
    PositionEmbedding_Vs30,
    TransformerEncoder,
    full_model,
)
from data.multiple_sta_dataset import multiple_station_dataset
from model_performance_analysis.analysis import Intensity_Plotter

mask_after_sec = 7
label = "pga"
data = multiple_station_dataset(
    "../data/TSMIP_1999_2019_Vs30.hdf5",
    mode="test",
    mask_waveform_sec=mask_after_sec,
    test_year=2016,
    label_key=label,
    mag_threshold=0,
    input_type="acc",
    data_length_sec=15,
)
# ===========predict==============
device = torch.device("cuda")
for num in [11]:
    path = f"../model/model{num}.pt"
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
    # output_df.to_csv(
    #     f"./predict/model {num} {mask_after_sec} sec prediction.csv", index=False
    # )
    fig, ax = Intensity_Plotter.true_predicted(
        y_true=output_df["answer"],
        y_pred=output_df["predict"],
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
        f"{mask_after_sec}s True Predict Plot, 2016 data",
        fontsize=20,
    )

    # fig.savefig(f"../predict/model {num} {mask_after_sec} sec.png")

# ===========merge info==============
Afile_path = "../data_preprocess/events_traces_catalog"
output_path = "../predict/station_blind_Vs30_bias2closed_station_2016"
catalog = pd.read_csv(f"{Afile_path}/1999_2019_final_catalog.csv")
traces_info = pd.read_csv(f"{Afile_path}/1999_2019_final_traces_Vs30.csv")
ensemble_predict = pd.read_csv(
    f"{output_path}/model 11 {mask_after_sec} sec prediction.csv"
)
trace_merge_catalog = pd.merge(
    traces_info,
    catalog[
        [
            "EQ_ID",
            "lat",
            "lat_minute",
            "lon",
            "lon_minute",
            "depth",
            "magnitude",
            "nsta",
            "nearest_sta_dist (km)",
        ]
    ],
    on="EQ_ID",
    how="left",
)
trace_merge_catalog["event_lat"] = (
    trace_merge_catalog["lat"] + trace_merge_catalog["lat_minute"] / 60
)

trace_merge_catalog["event_lon"] = (
    trace_merge_catalog["lon"] + trace_merge_catalog["lon_minute"] / 60
)
trace_merge_catalog.drop(
    ["lat", "lat_minute", "lon", "lon_minute"], axis=1, inplace=True
)
trace_merge_catalog.rename(columns={"elevation (m)": "elevation"}, inplace=True)


data_path = "D:/TEAM_TSMIP/data/TSMIP_1999_2019.hdf5"
dataset = h5py.File(data_path, "r")
for eq_id in ensemble_predict["EQ_ID"].unique():
    eq_id = int(eq_id)
    station_name = dataset["data"][str(eq_id)]["station_name"][:].tolist()

    ensemble_predict.loc[
        ensemble_predict.query(f"EQ_ID=={eq_id}").index, "station_name"
    ] = station_name

ensemble_predict["station_name"] = ensemble_predict["station_name"].str.decode("utf-8")


prediction_with_info = pd.merge(
    ensemble_predict,
    trace_merge_catalog.drop(
        [
            "latitude",
            "longitude",
            "elevation",
        ],
        axis=1,
    ),
    on=["EQ_ID", "station_name"],
    how="left",
    suffixes=["_window", "_file"],
)
# prediction_with_info.to_csv(
#     f"{output_path}/{mask_after_sec} sec model11 with all info.csv", index=False
# )
