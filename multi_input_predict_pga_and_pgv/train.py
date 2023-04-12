import pickle

import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from mlflow import log_artifact, log_metrics, log_param, log_params
from torch.backends import cudnn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from tqdm import tqdm

from model_multi_input import (
    MCCNN,
    MHCNN,
    MLP,
    PositionEmbedding,
    TransformerEncoder,
    MDN,
    full_model_MCC,
    full_model_MHC,
)
from dataset_acc_vel_dis import CustomSubset, multiple_station_dataset_outputs


def train_process(
    full_Model,
    optimizer,
    hyper_param,
    train_data_size=0.8,
    num_of_gaussian=10,
):
    experiment = mlflow.get_experiment_by_name("all predict pga and pgv")
    with mlflow.start_run(
        run_name="MHC", experiment_id=experiment.experiment_id
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
        full_data = multiple_station_dataset_outputs(
            "D:/TEAM_TSMIP/data/TSMIP_filtered.hdf5",
            mode="train",
            mask_waveform_sec=3,
            weight_label=False,
            test_year=2016,
            mask_waveform_random=True,
            input_type=["acc", "vel", "dis"],
            label_keys=["pga", "pgv"],
            data_length_sec=10,
        )
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

        loss_func = nn.GaussianNLLLoss(reduction="none")
        training_loss = []
        validation_loss = []
        print(f'train {hyper_param["num_epochs"]} times')
        the_last_loss = 100  # initial early stop value
        patience = 8
        trigger_times = 0
        for epoch in range(hyper_param["num_epochs"]):
            print(f"Epoch:{epoch}")
            print("--------------------train_start--------------------")
            for sample in tqdm(train_loader):
                optimizer.zero_grad()
                weight, sigma, mu = full_Model(sample)
                train_loss = []
                for index, label_type in enumerate(["pga", "pgv"]):
                    begin = int(index * (num_of_gaussian / 2))
                    end = int(index * (num_of_gaussian / 2) + (num_of_gaussian / 2))
                    label = (
                        sample[f"{label_type}"]
                        .reshape(hyper_param["batch_size"], full_data.label_target, 1)
                        .cuda()
                    )
                    mask = ~label.eq(0)  # 不讓pga zero padding去計算loss

                    label_masked = torch.masked_select(label, mask).reshape(-1, 1)
                    weight_masked = torch.masked_select(
                        weight[:, :, begin:end], mask
                    ).reshape(-1, int(num_of_gaussian / 2))
                    sigma_masked = torch.masked_select(
                        sigma[:, :, begin:end], mask
                    ).reshape(-1, int(num_of_gaussian / 2))
                    mu_masked = torch.masked_select(mu[:, :, begin:end], mask).reshape(
                        -1, int(num_of_gaussian / 2)
                    )
                    loss = torch.mean(
                        torch.sum(
                            weight_masked
                            * loss_func(mu_masked, label_masked, sigma_masked),
                            axis=1,
                        )
                    ).cuda()
                    train_loss.append(loss)
                train_loss_mean = sum(train_loss) / len(train_loss)
                train_loss_mean.backward()
                optimizer.step()
            print("train_loss", train_loss_mean)
            training_loss.append(train_loss_mean.data)

            for sample in tqdm(valid_loader):
                weight, sigma, mu = full_Model(sample)
                val_loss = []
                for index, label_type in enumerate(["pga", "pgv"]):
                    begin = int(index * (num_of_gaussian / 2))
                    end = int(index * (num_of_gaussian / 2) + (num_of_gaussian / 2))
                    label = (
                        sample[f"{label_type}"]
                        .reshape(hyper_param["batch_size"], full_data.label_target, 1)
                        .cuda()
                    )
                    mask = ~label.eq(0)  # 不讓pga zero padding去計算loss

                    label_masked = torch.masked_select(label, mask).reshape(-1, 1)
                    weight_masked = torch.masked_select(
                        weight[:, :, begin:end], mask
                    ).reshape(-1, int(num_of_gaussian / 2))
                    sigma_masked = torch.masked_select(
                        sigma[:, :, begin:end], mask
                    ).reshape(-1, int(num_of_gaussian / 2))
                    mu_masked = torch.masked_select(mu[:, :, begin:end], mask).reshape(
                        -1, int(num_of_gaussian / 2)
                    )
                    loss = torch.mean(
                        torch.sum(
                            weight_masked
                            * loss_func(mu_masked, label_masked, sigma_masked),
                            axis=1,
                        )
                    ).cuda()
                    val_loss.append(loss)
                val_loss_mean = sum(val_loss) / len(val_loss)
            print("val_loss", val_loss_mean)
            validation_loss.append(val_loss_mean.data)
            log_metrics(
                {
                    "train_loss": train_loss_mean.item(),
                    "val_loss": val_loss_mean.item(),
                },
                step=epoch,
            )
            # epoch early stopping:
            current_loss = val_loss_mean.data
            if current_loss > the_last_loss:
                trigger_times += 1
                print("early stop trigger times:", trigger_times)
                if trigger_times >= patience:
                    print(f"Early stop! stop at epoch: {epoch}")
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
                    log_param("epoch early stop", epoch)
                    return training_loss, validation_loss

                continue

            else:
                print("trigger 0 time")
                trigger_times = 0
                # save model
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
    num_epochs = 100
    # batch_size=16
    for batch_size in [16, 32]:
        for LR in [1e-5, 5e-5]:
            for i in range(3):
                model_index += 1
                hyper_param = {
                    "model_index": model_index,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": LR,
                }
                print(f"learning rate: {LR}")
                emb_dim = 450
                mlp_dims = (450, 300, 150, 100, 50, 30)

                acc_MHCNN_model = MHCNN().cuda()
                vel_MHCNN_model = MHCNN().cuda()
                dis_MHCNN_model = MHCNN().cuda()
                pos_emb_model = PositionEmbedding(emb_dim=emb_dim).cuda()
                transformer_model = TransformerEncoder(d_model=emb_dim)
                mlp_model = MLP(input_shape=(emb_dim,), dims=mlp_dims).cuda()
                mdn_model = MDN(input_shape=(mlp_dims[-1],)).cuda()

                full_Model_mhc = full_model_MHC(
                    acc_MHCNN_model,
                    vel_MHCNN_model,
                    dis_MHCNN_model,
                    pos_emb_model,
                    transformer_model,
                    mlp_model,
                    mdn_model,
                    emb_dim=emb_dim,
                    pga_targets=25,
                )

                optimizer = torch.optim.Adam(
                    [
                        {"params": acc_MHCNN_model.parameters()},
                        {"params": vel_MHCNN_model.parameters()},
                        {"params": dis_MHCNN_model.parameters()},
                        {"params": transformer_model.parameters()},
                        {"params": mlp_model.parameters()},
                        {"params": mdn_model.parameters()},
                    ],
                    lr=LR,
                )
                training_loss, validation_loss = train_process(
                    full_Model_mhc, optimizer, hyper_param
                )
