import pickle

import mlflow.pytorch
import torch
import torch.nn as nn
from mlflow import log_artifact, log_metrics, log_param, log_params
from torch.backends import cudnn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from CNN_Transformer_Mixtureoutput_TEAM import (CNN, MDN, MLP,
                                                PositionEmbedding,
                                                TransformerEncoder, full_model,
                                                mdn_loss_fn)
from multiple_sta_dataset import (multiple_station_dataset,
                                  multiple_station_dataset_new)


def train_process(full_Model,optimizer,hyper_param,num_of_gaussian=5,train_data_size=0.8):
    experiment = mlflow.get_experiment_by_name("5_sec_with_new_dataset")
    with mlflow.start_run(run_name="TSMIP_EEW",experiment_id=experiment.experiment_id) as run:
        log_params({"epochs":hyper_param["num_epochs"],
                    "batch size":hyper_param["batch_size"],
                    "learning rate":hyper_param["learning_rate"]})
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        cudnn.benchmark = True
        full_data=multiple_station_dataset_new("D:/TEAM_TSMIP/data/TSMIP.hdf5",train_mode=True,
                                                mask_waveform_sec=5,test_year=2016) 

        train_set_size = int(len(full_data) * train_data_size)
        valid_set_size = len(full_data) - train_set_size
        torch.manual_seed(0)
        train_dataset, val_dataset = random_split(full_data, [train_set_size, valid_set_size])
        train_loader=DataLoader(dataset=train_dataset,batch_size=hyper_param["batch_size"],shuffle=True,pin_memory=True,num_workers=5,drop_last=True)
        valid_loader=DataLoader(dataset=val_dataset,batch_size=hyper_param["batch_size"],shuffle=True,pin_memory=True,num_workers=5,drop_last=True)
        gaussian_loss = nn.GaussianNLLLoss(reduction="none")
        training_loss = []
        validation_loss = []
        print(f'train {hyper_param["num_epochs"]} times')
        the_last_loss = 100 #initial early stop value
        patience = 8 
        trigger_times = 0
        for epoch in range(hyper_param["num_epochs"]):
            print(f"Epoch:{epoch}")
            print("--------------------train_start--------------------")
            for sample in tqdm(train_loader):
                optimizer.zero_grad()
                weight,sigma,mu = full_Model(sample)

                pga_label=sample[3].reshape(hyper_param["batch_size"],full_data.pga_target,1).cuda()
                mask=~pga_label.eq(0) #不讓pga zero padding去計算loss

                pga_label_masked=torch.masked_select(pga_label, mask).reshape(-1,1)
                weight_masked=torch.masked_select(weight, mask).reshape(-1,num_of_gaussian)
                sigma_masked=torch.masked_select(sigma, mask).reshape(-1,num_of_gaussian)
                mu_masked=torch.masked_select(mu, mask).reshape(-1,num_of_gaussian)
                # train_loss = mdn_loss_fn(weight_masked, sigma_masked, mu_masked, pga_label_masked).cuda()
                train_loss=torch.mean(torch.sum(weight_masked*gaussian_loss(mu_masked, pga_label_masked, sigma_masked),axis=1)).cuda()
                train_loss.backward()
                optimizer.step()
            print("train_loss",train_loss)
            training_loss.append(train_loss.data)

            for sample in tqdm(valid_loader):
                weight,sigma,mu = full_Model(sample)

                pga_label=sample[3].reshape(hyper_param["batch_size"],full_data.pga_target,1).cuda()
                mask=~pga_label.eq(0) #不讓pga zero padding去計算loss

                pga_label_masked=torch.masked_select(pga_label, mask).reshape(-1,1)
                weight_masked=torch.masked_select(weight, mask).reshape(-1,num_of_gaussian)
                sigma_masked=torch.masked_select(sigma, mask).reshape(-1,num_of_gaussian)
                mu_masked=torch.masked_select(mu, mask).reshape(-1,num_of_gaussian)
                # val_loss = mdn_loss_fn(weight_masked, sigma_masked, mu_masked, pga_label_masked).cuda()
                val_loss = torch.mean(torch.sum(weight_masked*gaussian_loss(mu_masked, pga_label_masked, sigma_masked),axis=1)).cuda()
            print("val_loss",val_loss)
            validation_loss.append(val_loss.data)
            log_metrics({'train_loss': train_loss.item(), 'val_loss': val_loss.item()},step=epoch)   
            #epoch early stopping:
            current_loss=val_loss.data
            if current_loss > the_last_loss:
                trigger_times += 1
                print('early stop trigger times:', trigger_times)

                if trigger_times >= patience:
                    print(f"Early stopping! stop at epoch: {epoch}")
                    path="./model"
                    model_file = f"{path}/model{hyper_param['model_index']}.pt"
                    torch.save(full_Model.state_dict(), model_file)
                    log_artifact(model_file)
                    with open(f"{path}/train loss{hyper_param['model_index']}", "wb") as fp:
                        pickle.dump(training_loss, fp)
                        log_artifact(f"{path}/train loss{hyper_param['model_index']}")
                    with open(f"{path}/validation loss{hyper_param['model_index']}", "wb") as fp:
                        pickle.dump(validation_loss, fp)
                        log_artifact(f"{path}/validation loss{hyper_param['model_index']}")
                    log_param("epoch early stop",epoch)
                    return full_Model,training_loss,validation_loss
                    
                continue

            else:
                print('trigger 0 time')
                trigger_times = 0

            the_last_loss = current_loss
        print('Train Epoch: {}/{} Traing_Loss: {} Val_Loss: {}'.format(epoch+1, hyper_param["num_epochs"], train_loss.data, val_loss.data))
        path="./model"
        model_file = f"{path}/model{hyper_param['model_index']}.pt"
        torch.save(full_Model.state_dict(), model_file)
        log_artifact(model_file)
        with open(f"{path}/train loss{hyper_param['model_index']}", "wb") as fp:
            pickle.dump(training_loss, fp)
            log_artifact(f"{path}/train loss{hyper_param['model_index']}")
        with open(f"{path}/validation loss{hyper_param['model_index']}", "wb") as fp:
            pickle.dump(validation_loss, fp)
            log_artifact(f"{path}/validation loss{hyper_param['model_index']}")
            log_param("no early stop",epoch+1)
    return full_Model,training_loss,validation_loss


if __name__ == '__main__':
    train_data_size=0.8
    model_index=0
    num_epochs=100
    # batch_size=16
    for batch_size in [16,32]:
        for LR in [1e-4]:
            for i in range(5):
                model_index+=1
                hyper_param={
                            "model_index":model_index,
                            "num_epochs":num_epochs,
                            "batch_size":batch_size,
                            "learning_rate":LR
                            }
                print(f"learning rate: {LR}")
                num_of_gaussian=5
                emb_dim=150
                mlp_dims=(150, 100, 50, 30, 10)

                CNN_model=CNN().cuda()
                pos_emb_model = PositionEmbedding(emb_dim=emb_dim).cuda()
                transformer_model=TransformerEncoder()  
                mlp_model=MLP(input_shape=(emb_dim,),dims=mlp_dims).cuda()
                mdn_model=MDN(input_shape=(mlp_dims[-1],)).cuda()

                full_Model=full_model(CNN_model,pos_emb_model,transformer_model,mlp_model,mdn_model,pga_targets=25)

                optimizer = torch.optim.Adam([
                                        {'params':CNN_model.parameters()},
                                        {'params':transformer_model.parameters()},
                                        {"params":mlp_model.parameters()},
                                        {'params':mdn_model.parameters()}
                                        ], lr=LR)
                full_Model,training_loss,validation_loss=train_process(full_Model,optimizer,hyper_param)
            # save model
            # path="../multi-station/consider station zero padding mask/mask after p_picking 3 sec"
            # FILE = f'{path}/model/target position not influence each other/{num_epochs} epoch model_lr{LR}_batch_size{batch_size}_earlystop oversample sort_picks.pt'
            # torch.save(full_Model.state_dict(), FILE)
            # #save loss result
            # with open(f"{path}/loss/target position not influence each other/{num_epochs} epoch mdn_training loss_lr{LR}_batch_size{batch_size}_earlystop oversample sort_picks", "wb") as fp:
            #     pickle.dump(training_loss, fp)
            # with open(f"{path}/loss/target position not influence each other/{ensamble_index} {num_epochs} epoch mdn_validation loss_lr{LR}_batch_size{batch_size}_earlystop oversample sort_picks", "wb") as fp:
            #     pickle.dump(validation_loss, fp)
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt

# path="../multi-station/consider station zero padding mask/mask after p_picking 3 sec/loss/target position not influence each other"
# for i in range(20,40):
#     train_file=f"{i} 75 epoch mdn_training loss_lr5e-05_batch_size32_earlystop oversample sort_picks"
#     val_file=f"{i} 75 epoch mdn_validation loss_lr5e-05_batch_size32_earlystop oversample sort_picks"

#     training_data=open(f"{path}/{train_file}", "rb")
#     val_data=open(f"{path}/{val_file}", "rb")

#     training_loss= pickle.load(training_data)
#     validation_loss = pickle.load(val_data)

#     training_loss=[loss.cpu().numpy() for loss in training_loss]
#     validation_loss=[loss.cpu().numpy() for loss in validation_loss]

#     fig,ax=plt.subplots(figsize=(7,7))


#     ax.plot(np.arange(len(training_loss)),training_loss)
#     ax.plot(np.arange(len(validation_loss)),validation_loss)
#     ax.set_xlabel("epoch")
#     ax.set_ylabel("MDN loss")
#     # ax.set_ylim(-0.2,1)
#     ax.set_title(f"multiple station model{i} loss, lr5e-05, epoch early stop")
#     ax.legend(["train","validation"])
#     plt.close()
#     fig.savefig(f"{path}/model{i} loss curve.png")

