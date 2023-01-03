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


class predict_dataset(Dataset):
    def __init__(self,predict_catalog=None,ok_event_id=None,ok_events_index=None,p_picks=None,
                        mask_waveform_sec=3,sampling_rate=200,data_length_sec=30,
                        max_station_number=25,trigger_station_threshold=3):
        Events_index=[]
        for I,event in enumerate(ok_event_id):
            single_event_index=ok_events_index[np.where(ok_events_index[:,0]==event)[0]]
            single_event_p_picks=p_picks[np.where(ok_events_index[:,0]==event)[0]]
            if np.count_nonzero(single_event_p_picks<single_event_p_picks[0]+(mask_waveform_sec*sampling_rate))<trigger_station_threshold: 
                continue
            Events_index.append(single_event_index)
        self.predict_catalog=predict_catalog
        self.events_index=Events_index
        self.mask_waveform_sec=mask_waveform_sec
        self.sampling_rate=sampling_rate
        self.max_station_number=max_station_number
        self.data_length_sec=data_length_sec
    def __len__(self):
        return len(self.events_index)
    def __getitem__(self, index):
        specific_index=self.events_index[index]
        with h5py.File("D:/TEAM_TSMIP/data/TSMIP.hdf5", 'r') as f:
            specific_waveforms=[]
            stations_location=[]
            pga_targets_location=[]
            pga_labels=[]
            p_picks=[]
            for i,eventID in enumerate(specific_index): #trace loop

                waveform=f['data'][str(eventID[0])]["traces"][eventID[1]]
                p_pick=f['data'][str(eventID[0])]["p_picks"][eventID[1]]
                if i==0:
                    first_p_pick=p_pick=f['data'][str(eventID[0])]["p_picks"][eventID[1]]

                station_location=f['data'][str(eventID[0])]["station_location"][eventID[1]]
                if len(waveform)!=self.data_length_sec*self.sampling_rate:
                    waveform=np.pad(waveform,((0,self.data_length_sec*self.sampling_rate-len(waveform)),(0,0)),"constant")
                pga=np.array(f['data'][str(eventID[0])]["pga"][eventID[1]]).reshape(1,1)

                if p_pick<first_p_pick+(self.mask_waveform_sec*self.sampling_rate):
                    specific_waveforms.append(waveform)
                    stations_location.append(station_location)
                    p_picks.append(p_pick)
                
                pga_targets_location.append(station_location)
                pga_labels.append(pga)
            if len(stations_location)<self.max_station_number: #triggered station < max_station_number (default:25) zero padding
                for zero_pad_num in range(self.max_station_number-len(stations_location)):
                    # print(f"second {waveform.shape}")
                    specific_waveforms.append(np.zeros_like(waveform))
                    stations_location.append(np.zeros_like(station_location))
            
            elif len(stations_location)>self.max_station_number:
                specific_waveforms=specific_waveforms[:self.max_station_number]
                stations_location=stations_location[:self.max_station_number]
                p_picks=p_picks[:self.max_station_number]

            Specific_waveforms=np.array(specific_waveforms)
            for i in range(len(p_picks)):
                Specific_waveforms[i,:p_picks[i],:]=0
            if self.mask_waveform_sec:
                Specific_waveforms[:,first_p_pick+(self.mask_waveform_sec*self.sampling_rate):,:]=0

            Stations_location=np.array(stations_location)
            PGA_targets_location=np.array(pga_targets_location)
            PGA_labels=np.array(pga_labels)
            P_picks=np.array(p_picks)
            EQ_ID=eventID[0]

        return Specific_waveforms,Stations_location,PGA_targets_location,PGA_labels,P_picks,EQ_ID

mask_after_sec=5
trigger_station_threshold=2
data=multiple_station_dataset_new("D:/TEAM_TSMIP/data/TSMIP_new.hdf5",test_mode=True,
                                    mask_waveform_sec=mask_after_sec,test_year=2018,
                                    trigger_station_threshold=trigger_station_threshold)
#=========================
emb_dim=150
mlp_dims=(150, 100, 50, 30, 10)
CNN_model=CNN().cuda()
pos_emb_model = PositionEmbedding(emb_dim=emb_dim).cuda()
transformer_model=TransformerEncoder()
mlp_model=MLP(input_shape=(emb_dim,),dims=mlp_dims).cuda()
mdn_model=MDN(input_shape=(mlp_dims[-1],)).cuda()
for i in [4]:
    path=f"./model/3_sec_with_full_station_position 1991-2020/model{i}.pt"
    print(f"model{i}")

    predict_data=predict_dataset(predict_catalog=data.event_metadata,
                                    ok_event_id=data.ok_event_id,
                                    ok_events_index=data.ok_events_index,
                                    p_picks=data.p_picks,
                                    mask_waveform_sec=mask_after_sec,
                                    trigger_station_threshold=trigger_station_threshold)

    loader=DataLoader(dataset=predict_data,batch_size=1)

    device = torch.device('cuda')
    Mixture_mu = []
    PGA=[]
    P_picks=[]
    for j,sample in tqdm(enumerate(loader)):
        # print(f"index:{j}")
        full_Model=full_model(CNN_model,pos_emb_model,transformer_model,mlp_model,mdn_model,pga_targets=sample[3].shape[1]).to(device)
        full_Model.load_state_dict(torch.load(path))

        weight,sigma,mu=full_Model(sample)
        
        weight=weight.cpu()
        sigma=sigma.cpu()
        mu=mu.cpu()
        # if j==0:
        #     Mixture_mu=torch.sum(weight*mu,dim=2).cpu().detach().numpy()
        #     PGA=sample[3].cpu().detach().numpy()
        # else:
        #     Mixture_mu=np.concatenate([Mixture_mu,torch.sum(weight*mu,dim=2).cpu().detach().numpy()],axis=1)
        #     PGA=np.concatenate([PGA,sample[3].cpu().detach().numpy()],axis=1)
        # print(torch.cuda.memory_allocated()/1024/1024/1024,"GB")
        # print("=========================")
        if int(sample[-1])!=27558:
            continue
        Mixture_mu=torch.sum(weight*mu,dim=2).cpu().detach().numpy()
        PGA=sample[3].cpu().detach().numpy()
        PGA=PGA.flatten()
        Mixture_mu=Mixture_mu.flatten()
        output={"predict":Mixture_mu,"answer":PGA}
        output_df=pd.DataFrame(output)
        # output_df.to_csv(f"./predict/model{i} {mask_after_sec} sec prediction.csv",index=False)
        fig=true_predicted(y_true=output_df["answer"],y_pred=output_df["predict"],
                    time=mask_after_sec,quantile=False,agg="point", point_size=30)
        # plt.close()
        # fig.savefig(f"./predict/{int(sample[-1])} {mask_after_sec} sec.png")

        if sample[4].shape[1]==1:
            wav_fig,ax=plt.subplots(figsize=(14,7))
            for k in range(0,3):
                ax.plot(sample[0][:,0,:,k].flatten().numpy())
                ax.set_yticklabels("")
            ax.axvline(x=sample[4].flatten().numpy(),c="r")
            ax.set_title(f"EQ_ID:{int(sample[-1])} input")
            # wav_fig.savefig(f"./predict/{int(sample[-1])} input {mask_after_sec} sec.png")
        else:
            wav_fig,ax=plt.subplots(sample[4].shape[1],1,figsize=(14,7))
            for i in range(0,sample[4].shape[1]):
                for k in range(0,3):
                    ax[i].plot(sample[0][:,i,:,k].flatten().numpy())
                    ax[i].set_yticklabels("")
                ax[i].axvline(x=sample[4].flatten().numpy()[i],c="r")
            ax[0].set_title(f"{int(sample[-1])}input")
            # wav_fig.savefig(f"./predict/{int(sample[-1])} input {mask_after_sec} sec.png")
        # plt.close()
    # PGA=PGA.flatten()
    # Mixture_mu=Mixture_mu.flatten()

    # output={"predict":Mixture_mu,"answer":PGA}
    # output_df=pd.DataFrame(output)
    # # output_df.to_csv(f"./predict/model{i} {mask_after_sec} sec 2 triggered station prediction.csv",index=False)
    # fig=true_predicted(y_true=output_df["answer"],y_pred=output_df["predict"],
    #                 time=mask_after_sec,quantile=False,agg="point", point_size=12)
    # fig.savefig(f"./predict/model{i} {mask_after_sec} sec 2 triggered station.png")

# pre1=pd.read_csv("D:/TEAM_TSMIP/predict/model4 3 sec 2 triggered station prediction.csv")
# pre2=pd.read_csv("D:/TEAM_TSMIP/predict/model5 3 sec 2 triggered station prediction.csv")
# pre3=pd.read_csv("D:/TEAM_TSMIP/predict/model23 3 sec 2 triggered station prediction.csv")
# fig=true_predicted(y_true=pre1["answer"],y_pred=(pre1["predict"]+pre2["predict"]+pre3["predict"])/3,
#                     time=3,quantile=False,agg="point", point_size=12)


# plot model input waveform 
# import matplotlib.pyplot as plt

# fig,ax=plt.subplots(len(predict_data[115][4]),1,figsize=(14,7))
# for i in range(0,len(predict_data[115][4])):
#     for j in range(0,3):
#         ax[i].plot(predict_data[115][0][i,:,j])
#     ax[i].axvline(x=predict_data[115][4][i],c="r")
#     ax[i].set_yticklabels("")
#     ax[i].set_xticklabels("")
# ax[-1].set_xticklabels(["0","0","5","10","15","20","25","30"],fontsize=25)
# ax[-1].set_xlabel("Time (second)",fontsize=35)
# fig.savefig(f"./model{i} {mask_after_sec} sec.pdf")

