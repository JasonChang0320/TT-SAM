import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from CNN_Transformer_Mixtureoutput_TEAM import (CNN, MDN, MLP,
                                                PositionEmbedding,
                                                TransformerEncoder, full_model)
from multiple_sta_dataset import (multiple_station_dataset,
                                  multiple_station_dataset_new)
from plot_predict_map import true_predicted

mask_after_sec=10
trigger_station_threshold=1
data=multiple_station_dataset_new("D:/TEAM_TSMIP/data/TSMIP_new.hdf5",mode="test",
                                    mask_waveform_sec=mask_after_sec,test_year=2018,
                                    trigger_station_threshold=trigger_station_threshold,mag_threshold=0)
#=========================
device = torch.device('cuda')
for num in [1,3,18,20]:
    path=f"./model/model{num}.pt"
    emb_dim=150
    mlp_dims=(150, 100, 50, 30, 10)
    CNN_model=CNN().cuda()
    pos_emb_model = PositionEmbedding(emb_dim=emb_dim).cuda()
    transformer_model=TransformerEncoder()
    mlp_model=MLP(input_shape=(emb_dim,),dims=mlp_dims).cuda()
    mdn_model=MDN(input_shape=(mlp_dims[-1],)).cuda()
    full_Model=full_model(CNN_model,pos_emb_model,transformer_model,mlp_model,mdn_model,pga_targets=25).to(device)
    full_Model.load_state_dict(torch.load(path))
    loader=DataLoader(dataset=data,batch_size=1)

    Mixture_mu = []
    PGA=[]
    P_picks=[]
    EQ_ID=[]
    PGA_time=[]
    for j,sample in tqdm(enumerate(loader)):

        picks=sample[4].flatten().numpy().tolist()
        pga_time=sample[6].flatten().numpy().tolist()
        P_picks.extend(picks)
        P_picks.extend([np.nan]*(25-len(picks)))
        PGA_time.extend(pga_time)
        PGA_time.extend([np.nan]*(25-len(pga_time)))
        eq_id=sample[5][:,:,0].flatten().numpy().tolist()
        EQ_ID.extend(sample[5][:,:,0].flatten().numpy().tolist())
        EQ_ID.extend([np.nan]*(25-len(eq_id)))
        weight,sigma,mu=full_Model(sample)
        
        weight=weight.cpu()
        sigma=sigma.cpu()
        mu=mu.cpu()
        if j==0:
            Mixture_mu=torch.sum(weight*mu,dim=2).cpu().detach().numpy()
            PGA=sample[3].cpu().detach().numpy()
        else:
            Mixture_mu=np.concatenate([Mixture_mu,torch.sum(weight*mu,dim=2).cpu().detach().numpy()],axis=1)
            PGA=np.concatenate([PGA,sample[3].cpu().detach().numpy()],axis=1)
    PGA=PGA.flatten()
    Mixture_mu=Mixture_mu.flatten()

    output={"EQ_ID":EQ_ID,"p_picks":P_picks,"pga_time":PGA_time,"predict":Mixture_mu,"answer":PGA}
    output_df=pd.DataFrame(output)
    output_df=output_df[output_df["answer"]!=0]
    output_df.to_csv(f"./predict/model{num} {mask_after_sec} sec {trigger_station_threshold} triggered station prediction.csv",index=False)
    fig,ax=true_predicted(y_true=output_df["answer"],y_pred=output_df["predict"],
                    time=mask_after_sec,quantile=False,agg="point", point_size=12)

    fig.savefig(f"./predict/model{num} {mask_after_sec} sec {trigger_station_threshold} triggered station.png")

input_waveform_picks=np.array(data[31][4])[np.array(data[31][4])<np.array(data[31][4])[0]+mask_after_sec*200]
wav_fig,ax=plt.subplots(len(input_waveform_picks),1,figsize=(14,7))
for i in range(0,len(input_waveform_picks)):
    for k in range(0,3):
        ax[i].plot(data[31][0][i,:,k].flatten())
        ax[i].set_yticklabels("")
    ax[i].axvline(x=input_waveform_picks[i],c="r")
# ax[0].set_title(f"{int(sample[-1])}input")

fig=true_predicted(y_true=output_df["answer"][output_df["EQ_ID"]==27558],y_pred=output_df["predict"][output_df["EQ_ID"]==27558],
                time=mask_after_sec,quantile=False,agg="point", point_size=12)

mask_after_sec=10
data1=pd.read_csv(f"predict/model1 {mask_after_sec} sec 1 triggered station prediction.csv")
data2=pd.read_csv(f"predict/model3 {mask_after_sec} sec 1 triggered station prediction.csv")
data3=pd.read_csv(f"predict/model18 {mask_after_sec} sec 1 triggered station prediction.csv")
data4=pd.read_csv(f"predict/model20 {mask_after_sec} sec 1 triggered station prediction.csv")

output_df=(data1+data2+data3+data4)/4
fig=true_predicted(y_true=output_df["answer"],y_pred=output_df["predict"],
                time=mask_after_sec,quantile=False,agg="point", point_size=12)

output_df.to_csv(f"./predict/model1 3 18 20 {mask_after_sec} sec {trigger_station_threshold} triggered station prediction.csv",index=False)

output_df=(data1+data2+data3+data4)/4
for eq_id in data.event_metadata["EQ_ID"]:
    fig,ax=true_predicted(y_true=output_df["answer"][output_df["EQ_ID"]==eq_id],y_pred=output_df["predict"][output_df["EQ_ID"]==eq_id],
                    time=mask_after_sec,quantile=False,agg="point", point_size=70)
    magnitude=data.event_metadata[data.event_metadata["EQ_ID"]==eq_id]["magnitude"].values[0]
    ax.set_title(f"{mask_after_sec}s True Predict Plot, EQ ID:{eq_id}, magnitude: {magnitude}",fontsize=20)
    plt.close()
    fig.savefig(f"./predict/updated dataset plot each event {mask_after_sec} sec/EQ ID_{eq_id} magnitude_{magnitude}.png")



