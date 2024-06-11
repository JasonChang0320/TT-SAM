import json
import pandas as pd
import torch
import sys

sys.path.append("../..")
from model.CNN_Transformer_Mixtureoutput_TEAM import (
    CNN,
    MDN,
    MLP,
    PositionEmbedding_Vs30,
    TransformerEncoder,
    full_model,
)

mask_after_sec = 3
num = 11
device = torch.device("cuda")
path = f"../../model/model{num}.pt"
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

Lat = []
Lon = []
Elev = []
Mixture_mu = []
station_name = []
for i in range(1, 15):
    print(i)
    with open(
        f"model_input/{mask_after_sec}_sec_without_broken_data/{i}.json", "r"
    ) as json_file:
        data = json.load(json_file)

    waveform = torch.tensor(data["waveform"]).to(torch.double).unsqueeze(0)

    input_station = torch.tensor(data["sta"]).to(torch.double).unsqueeze(0)

    target_station = torch.tensor(data["target"]).to(torch.double).unsqueeze(0)
    true_target_num = torch.sum(torch.all(target_station != 0, dim=-1)).item()
    sample = {"waveform": waveform, "sta": input_station, "target": target_station}

    lat = sample["target"][:, :, 0].flatten().tolist()
    lon = sample["target"][:, :, 1].flatten().tolist()
    elev = sample["target"][:, :, 2].flatten().tolist()
    Lat.extend(lat)
    Lon.extend(lon)
    Elev.extend(elev)
    weight, sigma, mu = full_Model(sample)
    Mixture_mu.append(
        torch.sum(weight * mu, dim=2).cpu().detach().numpy().flatten().tolist()
    )
    station_name += data["station_name"]
Mixture_mu = [item for sublist in Mixture_mu for item in sublist]
output = {
    "predict": Mixture_mu,
    "station_name": station_name,
    "latitude": Lat,
    "longitude": Lon,
    "elevation": Elev,
}

output_df = pd.DataFrame(output)

# output_df.to_csv(
#     f"no_include_broken_data_prediction/{mask_after_sec}_sec_prediction.csv", index=False
# )
