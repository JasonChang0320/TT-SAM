import pickle
import os
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from mlflow import log_artifact, log_metrics, log_param, log_params
from torch.backends import cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from tqdm import tqdm

from CNN_Transformer_Mixtureoutput_TEAM import (
    CNN,
    MDN,
    MLP,
    PositionEmbedding,
    TransformerEncoder,
    full_model,
    PositionEmbedding_Vs30,
)
from multiple_sta_dataset import CustomSubset, multiple_station_dataset


def train_process(
    full_Model,
    full_data,
    optimizer,
    hyper_param,
    num_of_gaussian=5,
    train_data_size=0.8,
    experiment_name=None,
    run_name=None,
):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    with mlflow.start_run(
        run_name=run_name,
        experiment_id=experiment.experiment_id,
    ) as run:
        log_params(
            {
                "epochs": hyper_param["num_epochs"],
                "batch size": hyper_param["batch_size"],
                "learning rate": hyper_param["learning_rate"],
            }
        )
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        cudnn.benchmark = True
        # full_data=multiple_station_dataset("D:/TEAM_TSMIP/data/TSMIP_new.hdf5",train_mode=True,oversample=1.5,
        #                                         mask_waveform_sec=3,test_year=2018)

        train_set_size = int(len(full_data) * train_data_size)
        valid_set_size = len(full_data) - train_set_size
        torch.manual_seed(0)
        # for pga training
        train_dataset, val_dataset = random_split(
            full_data, [train_set_size, valid_set_size]
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=hyper_param["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=5,
            drop_last=True,
        )
        valid_loader = DataLoader(
            dataset=val_dataset,
            batch_size=hyper_param["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=5,
            drop_last=True,
        )
        # for pgv training
        # indice = np.arange(len(full_data))
        # np.random.seed(0)
        # np.random.shuffle(indice)
        # train_indice, valid_indice = np.array_split(indice, [train_set_size])
        # train_dataset = CustomSubset(full_data, train_indice)
        # val_dataset = CustomSubset(full_data, valid_indice)

        # train_sampler = WeightedRandomSampler(
        #     weights=train_dataset.weight,
        #     num_samples=len(train_dataset),
        #     replacement=True,
        # )
        # train_loader = DataLoader(
        #     dataset=train_dataset,
        #     batch_size=hyper_param["batch_size"],
        #     sampler=train_sampler,
        #     shuffle=False,
        #     pin_memory=True,
        #     num_workers=5,
        #     drop_last=True,
        # )
        # valid_loader = DataLoader(
        #     dataset=val_dataset,
        #     batch_size=hyper_param["batch_size"],
        #     shuffle=True,
        #     pin_memory=True,
        #     num_workers=5,
        #     drop_last=True,
        # )

        gaussian_loss = nn.GaussianNLLLoss(reduction="none")
        training_loss = []
        validation_loss = []
        print(f'train {hyper_param["num_epochs"]} times')
        the_last_loss = 100  # initial early stop value
        if hyper_param["learning_rate"] >= 5e-05:
            patience = 10
        elif hyper_param["learning_rate"] >= 0:
            patience = 15
        print("patience", patience)
        trigger_times = 0
        for epoch in range(hyper_param["num_epochs"]):
            print(f"Epoch:{epoch+1}")
            print("--------------------train_start--------------------")
            for sample in tqdm(train_loader):
                optimizer.zero_grad()
                weight, sigma, mu = full_Model(sample)
                pga_label = (
                    sample["label"]
                    .reshape(hyper_param["batch_size"], full_data.label_target, 1)
                    .cuda()
                )
                mask = ~pga_label.eq(0)  # 不讓pga zero padding去計算loss

                pga_label_masked = torch.masked_select(pga_label, mask).reshape(-1, 1)
                weight_masked = torch.masked_select(weight, mask).reshape(
                    -1, num_of_gaussian
                )
                sigma_masked = torch.masked_select(sigma, mask).reshape(
                    -1, num_of_gaussian
                )
                mu_masked = torch.masked_select(mu, mask).reshape(-1, num_of_gaussian)
                # print("pga",pga_label_masked),print("w",weight_masked),print("s",sigma_masked),print("mu",mu_masked)
                # train_loss = mdn_loss_fn(weight_masked, sigma_masked, mu_masked, pga_label_masked).cuda()
                train_loss = torch.mean(
                    torch.sum(
                        weight_masked
                        * gaussian_loss(mu_masked, pga_label_masked, sigma_masked),
                        axis=1,
                    )
                ).cuda()
                train_loss.backward()
                optimizer.step()
            print("train_loss", train_loss)
            training_loss.append(train_loss.data)

            for sample in tqdm(valid_loader):
                weight, sigma, mu = full_Model(sample)

                pga_label = (
                    sample["label"]
                    .reshape(hyper_param["batch_size"], full_data.label_target, 1)
                    .cuda()
                )
                mask = ~pga_label.eq(0)  # 不讓pga zero padding去計算loss

                pga_label_masked = torch.masked_select(pga_label, mask).reshape(-1, 1)
                weight_masked = torch.masked_select(weight, mask).reshape(
                    -1, num_of_gaussian
                )
                sigma_masked = torch.masked_select(sigma, mask).reshape(
                    -1, num_of_gaussian
                )
                mu_masked = torch.masked_select(mu, mask).reshape(-1, num_of_gaussian)
                # val_loss = mdn_loss_fn(weight_masked, sigma_masked, mu_masked, pga_label_masked).cuda()
                val_loss = torch.mean(
                    torch.sum(
                        weight_masked
                        * gaussian_loss(mu_masked, pga_label_masked, sigma_masked),
                        axis=1,
                    )
                ).cuda()
            print("val_loss", val_loss)
            validation_loss.append(val_loss.data)
            log_metrics(
                {"train_loss": train_loss.item(), "val_loss": val_loss.item()},
                step=epoch + 1,
            )
            # checkpoint
            if train_loss.data < -1 and (epoch + 1) % 5 == 0:
                checkpoint_path = (
                    f"./model/model{hyper_param['model_index']}_checkpoints"
                )
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                torch.save(
                    full_Model.state_dict(),
                    f"{checkpoint_path}/epoch{epoch+1}_model.pt",
                )
            # epoch early stopping:
            current_loss = val_loss.data
            if the_last_loss < -1:
                patience = 15
            if current_loss > the_last_loss:
                trigger_times += 1
                print("early stop trigger times:", trigger_times)

                if trigger_times >= patience:
                    print(f"Early stopping! stop at epoch: {epoch+1}")
                    with open(
                        f"{path}/train loss{hyper_param['model_index']}", "wb"
                    ) as fp:
                        pickle.dump(training_loss, fp)
                        log_artifact(f"{path}/train loss{hyper_param['model_index']}")
                    with open(
                        f"{path}/validation loss{hyper_param['model_index']}", "wb"
                    ) as fp:
                        pickle.dump(validation_loss, fp)
                        log_artifact(
                            f"{path}/validation loss{hyper_param['model_index']}"
                        )
                    log_param("epoch early stop", epoch + 1)
                    return training_loss, validation_loss

                continue

            else:
                print("trigger 0 time")
                trigger_times = 0
                path = "./model"
                model_file = f"{path}/model{hyper_param['model_index']}.pt"
                torch.save(full_Model.state_dict(), model_file)
                log_artifact(model_file)

            the_last_loss = current_loss
        print(
            "Train Epoch: {}/{} Traing_Loss: {} Val_Loss: {}".format(
                epoch + 1, hyper_param["num_epochs"], train_loss.data, val_loss.data
            )
        )


if __name__ == "__main__":
    train_data_size = 0.8
    model_index = 0
    num_epochs = 300
    # batch_size=16
    for batch_size in [32, 16]:
        for LR in [5e-5, 2.5e-5]:
            for i in range(3):
                model_index += 1
                hyper_param = {
                    "model_index": model_index,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": LR,
                }
                print(f"learning rate: {LR}")
                num_of_gaussian = 5
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
                )
                # full_Model.load_state_dict(torch.load("mlruns/2/ebd8e01aefae4fc59929c5a07cc6fffb/artifacts/model2.pt"))
                # full_Model.train()
                optimizer = torch.optim.Adam(
                    [
                        {"params": CNN_model.parameters()},
                        {"params": transformer_model.parameters()},
                        {"params": mlp_model.parameters()},
                        {"params": mdn_model.parameters()},
                    ],
                    lr=LR,
                )
                full_data = multiple_station_dataset(
                    "D:/TEAM_TSMIP/data/TSMIP_1999_2019_Vs30.hdf5",
                    mode="train",
                    mask_waveform_sec=3,
                    weight_label=False,
                    oversample=1.5,
                    oversample_mag=4,
                    test_year=2016,
                    mask_waveform_random=True,
                    mag_threshold=0,
                    label_key="pga",
                    input_type="acc",
                    data_length_sec=15,
                    station_blind=True,
                    bias_to_closer_station=True,
                )
                training_loss, validation_loss = train_process(
                    full_Model,
                    full_data,
                    optimizer,
                    hyper_param,
                    experiment_name="bias to close station",
                    run_name="test 2016, oversample,vs30, station_blind",
                )
