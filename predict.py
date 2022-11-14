import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from CNN_Transformer_Mixtureoutput_TEAM import (CNN, MDN, MLP,
                                                PositionEmbedding,
                                                TransformerEncoder, full_model)
from multiple_sta_dataset import multiple_station_dataset
from plot_predict_map import true_predicted


class predict_dataset(Dataset):
    def __init__(self,predict_catalog=None,ok_event_id=None,ok_events_index=None,
                        mask_waveform_sec=3,sampling_rate=200,data_length_sec=30,max_station_number=25):
        Events_index=[]
        for I,event in enumerate(ok_event_id):
            single_event_index=ok_events_index[np.where(ok_events_index==event)[0]]
            Events_index.append(single_event_index)
        self.predict_catalog=predict_catalog
        self.events_index=Events_index
        self.mask_waveform_sec=mask_waveform_sec
        self.sampling_rate=sampling_rate
        self.max_station_number=max_station_number
        self.data_length_sec=data_length_sec
    def __len__(self):
        return len(self.predict_catalog)
    def __getitem__(self, index):
        specific_index=self.events_index[index]
        with h5py.File("D:/TEAM_TSMIP/data/TSMIP.hdf5", 'r') as f:
            specific_waveforms=[]
            stations_location=[]
            pga_targets_location=[]
            pga_labels=[]
            p_picks=[]
            for eventID in specific_index: #trace loop
                waveform=f['data'][str(eventID[0])]["traces"][eventID[1]]
                p_pick=f['data'][str(eventID[0])]["p_picks"][eventID[1]]
                station_location=f['data'][str(eventID[0])]["station_location"][eventID[1]]
                if len(waveform)!=self.data_length_sec*self.sampling_rate:
                    waveform=np.pad(waveform,((0,self.data_length_sec*self.sampling_rate-len(waveform)),(0,0)),"constant")
                pga=np.array(f['data'][str(eventID[0])]["pga"][eventID[1]]).reshape(1,1)
                specific_waveforms.append(waveform)
                # print(f"first {waveform.shape}")
                stations_location.append(station_location)
                p_picks.append(p_pick)

                pga_targets_location.append(station_location)
                pga_labels.append(pga)
            # print(f"station number:{len(stations_location)}")
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
            if self.mask_waveform_sec:
                Specific_waveforms[:,(5+self.mask_waveform_sec)*self.sampling_rate:,:]=0

            Stations_location=np.array(stations_location)
            PGA_targets_location=np.array(pga_targets_location)
            PGA_labels=np.array(pga_labels)
            P_picks=np.array(p_picks)

        return Specific_waveforms,Stations_location,PGA_targets_location,PGA_labels,P_picks
        
for i in range(1,2):
    path=f"./model/model{i}.pt"
    print(f"model{i}")
    mask_after_sec=3
    emb_dim=150
    mlp_dims=(150, 100, 50, 30, 10)
    CNN_model=CNN().cuda()
    pos_emb_model = PositionEmbedding(emb_dim=emb_dim).cuda()
    transformer_model=TransformerEncoder()
    mlp_model=MLP(input_shape=(emb_dim,),dims=mlp_dims).cuda()
    mdn_model=MDN(input_shape=(mlp_dims[-1],)).cuda()

    data=multiple_station_dataset("D:/TEAM_TSMIP/data/TSMIP.hdf5",test_mode=True,mask_waveform_sec=3,shift_waveform=False) 

    predict_data=predict_dataset(data.event_metadata,data.ok_event_id,data.ok_events_index)

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
        if j==0:
            Mixture_mu=torch.sum(weight*mu,dim=2).cpu().detach().numpy()
            PGA=sample[3].cpu().detach().numpy()
        else:
            Mixture_mu=np.concatenate([Mixture_mu,torch.sum(weight*mu,dim=2).cpu().detach().numpy()],axis=1)
            PGA=np.concatenate([PGA,sample[3].cpu().detach().numpy()],axis=1)
        # print(torch.cuda.memory_allocated()/1024/1024/1024,"GB")
        # print("=========================")
    PGA=PGA.flatten()
    Mixture_mu=Mixture_mu.flatten()

    output={"predict":Mixture_mu,"answer":PGA}
    output_df=pd.DataFrame(output)

    fig=true_predicted(y_true=output_df["answer"],y_pred=output_df["predict"],
                        time=3,quantile=False,agg="point", point_size=12)
    fig.savefig(f"./predict/model{i}.png")
