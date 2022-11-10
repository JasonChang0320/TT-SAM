from CNN_Transformer_Mixtureoutput_TEAM import CNN, PositionEmbedding,\
                                                TransformerEncoder,MLP,MDN,full_model,mdn_loss_fn
from multiple_sta_dataset import multiple_station_dataset
from torch.utils.data import  DataLoader, random_split
import torch
from torch.backends import cudnn
from tqdm import tqdm
import pickle





def train_process(full_Model,optimizer,num_epochs,batch_size,num_of_gaussian=5,train_data_size=0.8):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True
    full_data=multiple_station_dataset("D:/TEAM/italy.hdf5",train_mode=True,mask_waveform_sec=3,oversample=1.5,oversample_mag=4) 

    train_set_size = int(len(full_data) * train_data_size)
    valid_set_size = len(full_data) - train_set_size
    torch.manual_seed(0)
    train_dataset, val_dataset = random_split(full_data, [train_set_size, valid_set_size])
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=5,drop_last=True)
    valid_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=5,drop_last=True)

    training_loss = []
    validation_loss = []
    print(f'train {num_epochs} times')
    the_last_loss = 100 #initial early stop value
    patience = 8 
    trigger_times = 0
    for epoch in range(num_epochs):
        print(f"Epoch:{epoch}")
        print("--------------------train_start--------------------")
        for sample in tqdm(train_loader):
            optimizer.zero_grad()
            weight,sigma,mu = full_Model(sample)

            pga_label=sample[3].reshape(batch_size,full_data.pga_target,1).cuda()
            mask=~pga_label.eq(0) #不讓pga zero padding去計算loss

            pga_label_masked=torch.masked_select(pga_label, mask).reshape(-1,1)
            weight_masked=torch.masked_select(weight, mask).reshape(-1,num_of_gaussian)
            sigma_masked=torch.masked_select(sigma, mask).reshape(-1,num_of_gaussian)
            mu_masked=torch.masked_select(mu, mask).reshape(-1,num_of_gaussian)
            train_loss = mdn_loss_fn(weight_masked, sigma_masked, mu_masked, pga_label_masked).cuda()
            train_loss.backward()
            optimizer.step()
        print("train_loss",train_loss)
        training_loss.append(train_loss.data)

        for sample in tqdm(valid_loader):
            weight,sigma,mu = full_Model(sample)

            pga_label=sample[3].reshape(batch_size,full_data.pga_target,1).cuda()
            mask=~pga_label.eq(0) #不讓pga zero padding去計算loss

            pga_label_masked=torch.masked_select(pga_label, mask).reshape(-1,1)
            weight_masked=torch.masked_select(weight, mask).reshape(-1,num_of_gaussian)
            sigma_masked=torch.masked_select(sigma, mask).reshape(-1,num_of_gaussian)
            mu_masked=torch.masked_select(mu, mask).reshape(-1,num_of_gaussian)
            val_loss = mdn_loss_fn(weight_masked, sigma_masked, mu_masked, pga_label_masked).cuda()
        print("val_loss",val_loss)
        validation_loss.append(val_loss.data)

        #epoch early stopping:
        current_loss=val_loss.data
        if current_loss > the_last_loss:
            trigger_times += 1
            print('early stop trigger times:', trigger_times)

            if trigger_times >= patience:
                print(f"Early stopping! stop at epoch: {epoch}")

                return full_Model,training_loss,validation_loss
            
            continue

        else:
            print('trigger 0 time')
            trigger_times = 0

        the_last_loss = current_loss
    print('Train Epoch: {}/{} Traing_Loss: {} Val_Loss: {}'.format(epoch+1, num_epochs, train_loss.data, val_loss.data))
    return full_Model,training_loss,validation_loss

# train_process(full_Model,optimizer,num_epochs,batch_size)

if __name__ == '__main__':
    
    batch_size=32
    train_data_size=0.8
    for ensamble_index in range(20,40):
        print(f"ensamble_index: {ensamble_index}")
        for num_epochs in [75]:
            for LR in [5*1e-5]:
                print(f"learning rate: {LR}")
                # num_epochs=30
                # LR=1e-4
                num_of_gaussian=5
                emb_dim=150
                mlp_dims=(150, 100, 50, 30, 10)

                CNN_model=CNN().cuda()
                pos_emb_model = PositionEmbedding(emb_dim=emb_dim).cuda()
                transformer_model=TransformerEncoder()  
                mlp_model=MLP(input_shape=(emb_dim,),dims=mlp_dims).cuda()
                mdn_model=MDN(input_shape=(mlp_dims[-1],)).cuda()

                full_Model=full_model(CNN_model,pos_emb_model,transformer_model,mlp_model,mdn_model)

                optimizer = torch.optim.Adam([
                                        {'params':CNN_model.parameters()},
                                        {'params':transformer_model.parameters()},
                                        {"params":mlp_model.parameters()},
                                        {'params':mdn_model.parameters()}
                                        ], lr=LR)
                full_Model,training_loss,validation_loss=train_process(full_Model,optimizer,num_epochs,batch_size)
                #save model
                path="../multi-station/consider station zero padding mask/mask after p_picking 3 sec"
                FILE = f'{path}/model/target position not influence each other/{ensamble_index} {num_epochs} epoch model_lr{LR}_batch_size{batch_size}_earlystop oversample sort_picks.pt'
                torch.save(full_Model.state_dict(), FILE)
                #save loss result
                with open(f"{path}/loss/target position not influence each other/{ensamble_index} {num_epochs} epoch mdn_training loss_lr{LR}_batch_size{batch_size}_earlystop oversample sort_picks", "wb") as fp:
                    pickle.dump(training_loss, fp)
                with open(f"{path}/loss/target position not influence each other/{ensamble_index} {num_epochs} epoch mdn_validation loss_lr{LR}_batch_size{batch_size}_earlystop oversample sort_picks", "wb") as fp:
                    pickle.dump(validation_loss, fp)


import numpy as np
import pickle
import matplotlib.pyplot as plt

path="../multi-station/consider station zero padding mask/mask after p_picking 3 sec/loss/target position not influence each other"
for i in range(20,40):
    train_file=f"{i} 75 epoch mdn_training loss_lr5e-05_batch_size32_earlystop oversample sort_picks"
    val_file=f"{i} 75 epoch mdn_validation loss_lr5e-05_batch_size32_earlystop oversample sort_picks"

    training_data=open(f"{path}/{train_file}", "rb")
    val_data=open(f"{path}/{val_file}", "rb")

    training_loss= pickle.load(training_data)
    validation_loss = pickle.load(val_data)

    training_loss=[loss.cpu().numpy() for loss in training_loss]
    validation_loss=[loss.cpu().numpy() for loss in validation_loss]

    fig,ax=plt.subplots(figsize=(7,7))


    ax.plot(np.arange(len(training_loss)),training_loss)
    ax.plot(np.arange(len(validation_loss)),validation_loss)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MDN loss")
    # ax.set_ylim(-0.2,1)
    ax.set_title(f"multiple station model{i} loss, lr5e-05, epoch early stop")
    ax.legend(["train","validation"])
    plt.close()
    fig.savefig(f"{path}/model{i} loss curve.png")

