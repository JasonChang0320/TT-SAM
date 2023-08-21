import matplotlib.pyplot as plt
plt.subplots()
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from multiple_sta_dataset import multiple_station_dataset

mask_after_sec = 7
label = "pga"
eq_id=27558
data = multiple_station_dataset(
    "D:/TEAM_TSMIP/data/TSMIP_1999_2019_Vs30.hdf5",
    mode="test",
    mask_waveform_sec=mask_after_sec,
    test_year=2018,
    label_key=label,
    input_type="acc",
    data_length_sec=15,
)
path="./predict/station_blind_Vs30_2018/first try/mag bigger 5.5 predict from model 8"
record_prediction = pd.read_csv(
    f"{path}/{mask_after_sec}_sec_eqid{eq_id}_record_prediction.csv"
)
record_prediction=record_prediction[record_prediction["EQ_ID"]==eq_id].reset_index(drop=True)
# =========================
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

    eq_id_list = sample["EQ_ID"][:, :, 0].flatten().numpy().tolist()
    EQ_ID.extend(eq_id_list)
    EQ_ID.extend([np.nan] * (25 - len(eq_id_list)))
    if eq_id_list[0] == eq_id:
        waveform = sample["waveform"].numpy().reshape(25, 3000, 3)
        waveform_num = len(
            np.where(np.array(picks) <= picks[0] + (mask_after_sec * 200))[0]
        )
        waveforms_fig, waveforms_ax = plt.subplots(
            waveform_num,
            1,
            figsize=(7, 7),
        )
        for i in range(waveform_num):
            station_name = record_prediction["station_name"][i]
            answer = np.round(100 * (10 ** record_prediction["answer"][i]), 2)
            waveforms_ax[i].plot(waveform[i, :, 0])
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
        # if mask_after_sec==5: 
            waveform_fig,waveform_ax=plt.subplots(3,1,figsize=(7,7))
            for j in range(3):
                waveform_ax[j].plot(waveform[i,:,j])
                waveform_ax[j].axvline(x=picks[i],c="r")
                waveform_ax[0].set_title(f"acc waveform, PGA: {answer} gal",size=20)
                # waveform_fig.savefig(f"./predict/acc predict pga 1999_2019/model 2 meinong intensity map/index{i}_{station_name}_acc_input.png")
        waveforms_ax[0].set_title(f"EQID:{eq_id} {mask_after_sec} sec acc records, Z component",size=20)
        # waveforms_fig.savefig(f"{path}/eqid{eq_id}_{mask_after_sec}_sec_Z_acc.png",bbox_inches='tight')
        break
