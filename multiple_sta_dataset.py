from itertools import repeat

import h5py
import numpy as np
import pandas as pd
import torch
from numpy import ma
from torch.utils.data import DataLoader, Dataset


def shift_waveform(waveform,p_pick,start_before_sec=5,total_sec=30,sample_rate=200):

    start_point=p_pick-start_before_sec*sample_rate
    cut_waveform=waveform[start_point:,:]
    padded_waveform=np.pad(cut_waveform,((0,total_sec*sample_rate-len(cut_waveform)),(0,0)),"constant")
    #((top, bottom), (left, right)) 

    return padded_waveform

class multiple_station_dataset(Dataset):
    def __init__(self,data_path,test_year=2018,train_mode=None,test_mode=None,limit=None,
                        filter_trace_by_p_pick=True,label_key="pga",
                        mask_waveform_sec=None,mask_waveform_random=False,oversample=1,oversample_mag=4,
                        max_station_num=25,pga_target=15,shift_waveform=True,sort_by_picks=True
                        ):

        event_metadata = pd.read_hdf(data_path, 'metadata/event_metadata')
        trace_metadata = pd.read_hdf(data_path, 'metadata/traces_metadata')

        if train_mode:
            test_mask=[int(year)!=test_year for year in event_metadata["year"]]
            event_metadata=event_metadata[test_mask]
        elif test_mode:
            test_mask=[int(year)==test_year for year in event_metadata["year"]]
            event_metadata=event_metadata[test_mask]
        
        if limit:
            event_metadata = event_metadata.iloc[:limit]
        metadata = {}
        data = {}
        with h5py.File(data_path, 'r') as f:
            # for key in f['metadata'].keys():
            #     if key == 'event_metadata':
            #         continue
            #     metadata[key] = f['metadata'][key][()]
            # if overwrite_sampling_rate is not None:
            #     if metadata['sampling_rate'] % overwrite_sampling_rate != 0:
            #         raise ValueError(f'Overwrite sampling ({overwrite_sampling_rate}) rate must be true divisor of sampling'
            #                         f' rate ({metadata["sampling_rate"]})')
            #     decimate = metadata['sampling_rate'] // overwrite_sampling_rate
            #     metadata['sampling_rate'] = overwrite_sampling_rate
            # else:
            decimate = 1

            skipped = 0
            contained = []
            events_index=np.zeros((1,2),dtype=int)
            # events_index=[]
            for _, event in event_metadata.iterrows():
                event_name = str(int(event['EQ_ID']))
                # print(event_name)
                if event_name not in f['data']: #use catalog eventID campare to data eventID
                    skipped += 1
                    contained += [False]
                    continue
                contained += [True]
                g_event = f['data'][event_name]
                for key in g_event:
                    if key not in data:
                        data[key] = []
                    if key == 'traces':
                        index=np.arange(g_event[key].shape[0]).reshape(-1,1)
                        eventid = np.array([str(event_name)]*g_event[key].shape[0]).astype(np.int32).reshape(-1,1)
                        single_event_index=np.concatenate([eventid,index],axis=1)
                    else:
                        data[key] += [g_event[key][()]]
                    if key == 'p_picks':
                        data[key][-1] //= decimate

                events_index=np.append(events_index,single_event_index,axis=0)
                # events_index.append(single_event_index)
            events_index=np.delete(events_index,[0],0)
        labels = np.concatenate(data[label_key],axis=0)
        stations=np.concatenate(data["station_name"], axis=0)
        mask = (events_index != 0).any(axis=1)
        #picking over 30 seconds mask
        if filter_trace_by_p_pick:
            picks= np.concatenate(data['p_picks'], axis=0)
            mask = np.logical_and(mask, picks < 6000)
        #label is nan mask
        mask=np.logical_and(mask,~np.isnan(labels))

        labels=np.expand_dims(np.expand_dims(labels, axis=1), axis=2)
        stations=np.expand_dims(np.expand_dims(stations, axis=1), axis=2)
        p_picks=picks[mask]
        stations=stations[mask]
        labels=labels[mask]
        #new!!
        ok_events_index=events_index[mask]
        ok_event_id=np.intersect1d(np.array(event_metadata["EQ_ID"].values), ok_events_index)
        if oversample>1:
            oversampled_catalog=[]
            filter=(event_metadata["magnitude"]>=oversample_mag)
            oversample_catalog=np.intersect1d(np.array(event_metadata[filter]["EQ_ID"].values), ok_events_index)
            for eventid in oversample_catalog:
                catch_mag=(event_metadata["EQ_ID"]==eventid)
                mag=event_metadata[catch_mag]["magnitude"]
                repeat_time=int(oversample ** (mag - 1) - 1)
                oversampled_catalog.extend(repeat(eventid, repeat_time))

            oversampled_catalog=np.array(oversampled_catalog)
            ok_event_id=np.concatenate([ok_event_id,oversampled_catalog])

        Events_index=[]
        for I,event in enumerate(ok_event_id):
            single_event_index=ok_events_index[np.where(ok_events_index==event)[0]]
            single_event_p_picks=p_picks[np.where(ok_events_index==event)[0]]

            if sort_by_picks:
                sort=single_event_p_picks.argsort()
                single_event_p_picks=single_event_p_picks[sort]
                single_event_index=single_event_index[sort]

            if len(single_event_index)>25:
                time=int(np.ceil(len(single_event_index)/25)) #無條件進位
                # np.array_split(single_event_index, 25)
                splited_index=np.array_split(single_event_index, np.arange(25,25*time,25))
                for i in range(time):
                    Events_index.append(splited_index[i])
            else:
                Events_index.append(single_event_index)

        self.ok_events_index=ok_events_index
        self.ok_event_id=ok_event_id
        self.data_path=data_path
        self.event_metadata=event_metadata
        self.trace_metadata=trace_metadata
        self.data=data
        self.metadata=metadata 
        self.events_index=Events_index
        self.p_picks=p_picks
        self.stations=stations
        self.labels=labels
        self.oversample=oversample
        self.max_station_num=max_station_num
        self.pga_target=pga_target
        self.mask_waveform_random=mask_waveform_random
        self.mask_waveform_sec=mask_waveform_sec
        self.shift_waveform=shift_waveform


    def __len__(self):

        return len(self.events_index)

    def __getitem__(self, index):

        specific_index=self.events_index[index]
        max_station_number=self.max_station_num
        pga_target_number=self.pga_target

        with h5py.File("D:/TEAM_TSMIP/data/TSMIP.hdf5", 'r') as f:
            # for index in specific_index: #event loop
            specific_waveforms=[]
            stations_location=[]
            pga_targets_location=[]
            pga_labels=[]
            for eventID in specific_index: #trace loop
                waveform=f['data'][str(eventID[0])]["traces"][eventID[1]]
                p_pick=f['data'][str(eventID[0])]["p_picks"][eventID[1]]
                station_location=f['data'][str(eventID[0])]["station_location"][eventID[1]]
                if self.shift_waveform:
                    waveform=shift_waveform(waveform,p_pick)
                pga=np.array(f['data'][str(eventID[0])]["pga"][eventID[1]]).reshape(1,1)
                specific_waveforms.append(waveform)
                # print(f"first {waveform.shape}")
                stations_location.append(station_location)

                if len(pga_targets_location)<pga_target_number:
                    pga_targets_location.append(station_location)
                    pga_labels.append(pga)
            # print(f"station number:{len(stations_location)}")
            if len(stations_location)<max_station_number: #triggered station < max_station_number (default:25) zero padding
                for zero_pad_num in range(max_station_number-len(stations_location)):
                    # print(f"second {waveform.shape}")
                    specific_waveforms.append(np.zeros_like(waveform))
                    stations_location.append(np.zeros_like(station_location))
            # print("================")
            if len(pga_targets_location)<pga_target_number: #triggered station < max_station_number (default:25) zero padding
                for zero_pad_num in range(pga_target_number-len(pga_targets_location)):
                    pga_targets_location.append(np.zeros_like(station_location))
                    pga_labels.append(np.zeros_like(pga))


            Specific_waveforms=np.array(specific_waveforms)
            if self.mask_waveform_sec:
                Specific_waveforms[:,(5+self.mask_waveform_sec)*100:,:]=0
            if self.mask_waveform_random:
                for i in range(self.max_station_num):
                    if i==0:
                        random_time=100*np.random.randint(4,25)
                        mask = np.zeros(3000*3, dtype=int).reshape(3000,3)
                        mask[:random_time,:]=1
                        mask = mask.astype(bool)
                    else:
                        random_time=100*np.random.randint(4,25)
                        tmp = np.zeros(3000*3, dtype=int).reshape(3000,3)
                        tmp[:random_time,:]=1
                        tmp = tmp.astype(bool)
                        mask=np.concatenate([mask,tmp])
                mask=mask.reshape(Specific_waveforms.shape)
                Specific_waveforms=ma.masked_array(Specific_waveforms,mask=~mask)
                Specific_waveforms=Specific_waveforms.filled(fill_value=0)
            Stations_location=np.array(stations_location)
            PGA_targets_location=np.array(pga_targets_location)
            PGA_labels=np.array(pga_labels)

        return Specific_waveforms,Stations_location,PGA_targets_location,PGA_labels

# full_data=multiple_station_dataset("D:/TEAM_TSMIP/data/TSMIP.hdf5",train_mode=True,mask_waveform_sec=3,sort_by_picks=False)   

# batch_size=16
# loader=DataLoader(dataset=full_data,batch_size=batch_size,shuffle=True)
