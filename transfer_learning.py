import pickle

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
)
from multiple_sta_dataset import CustomSubset, multiple_station_dataset
from multi_station_training import train_process

if __name__ == "__main__":
    model_index = 0
    for batch_size in [32, 16]:
        for LR in [2.5e-5, 1e-5]:
            for i in range(2):
                if LR < 5e-5:
                    num_epochs = 300
                model_index += 1
                hyper_param = {
                    "model_index": model_index,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": LR,
                }
                num_of_gaussian = 5
                emb_dim = 150
                mlp_dims = (150, 100, 50, 30, 10)

                full_model_parameter = torch.load(
                    "origin model for transfer learning/model2.pt"
                )
                CNN_model = CNN(mlp_input=5665).cuda()
                pos_emb_model = PositionEmbedding(emb_dim=emb_dim).cuda()
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

                full_model_parameter = torch.load(
                    "origin model for transfer learning/model2.pt"
                )
                # ===========only load CNN parameter==============
                CNN_parameter={}
                for name, param in full_model_parameter.items():
                    if "model_CNN" in name: #model_CNN.conv2d1.0.weight : conv2d1.0.weight didn't match
                        name = name.replace("model_CNN.","")
                        CNN_parameter[name] = param
                full_Model.model_CNN.load_state_dict(CNN_parameter)
                # ================================================

                # full_Model.load_state_dict(full_model_parameter)

                # ===========freeze parts of CNN parameter=================
                # for name, param in full_Model.named_parameters():
                #     if name not in [
                #         "model_CNN.conv2d1.0.weight",
                #         "model_CNN.conv2d1.0.bias",
                #         "model_CNN.conv2d2.0.weight",
                #         "model_CNN.conv2d2.0.bias",
                #     ]:
                #         continue
                #     param.requires_grad = False

                # for name, param in full_Model.named_parameters():
                #     print("name: ", name)
                #     print("requires_grad: ", param.requires_grad)
                # ==========================================================
                trainable_param = filter(lambda p: p.requires_grad, full_Model.parameters())
                optimizer = torch.optim.Adam(
                    trainable_param, lr=LR, weight_decay=1e-5
                )  # L2 regularization
                full_data = multiple_station_dataset(
                    "D:/TEAM_TSMIP/data/TSMIP_1999_2019.hdf5",
                    mode="train",
                    mask_waveform_sec=3,
                    weight_label=False,
                    oversample=1,
                    oversample_mag=4,
                    test_year=2016,
                    mask_waveform_random=True,
                    mag_threshold=5,
                    label_key="pga",
                    input_type="acc",
                    data_length_sec=15,
                    # part_small_event=True
                )
                train_process(
                    full_Model,
                    full_data,
                    optimizer,
                    hyper_param,
                    experiment_name="transfer learning (origin->mag>5)",
                    run_name="copy cnn and all trainable",
                )
