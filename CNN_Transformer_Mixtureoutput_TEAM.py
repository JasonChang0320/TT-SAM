import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.autograd import Variable
from torchsummary import summary


class LambdaLayer(nn.Module):
    def __init__(self, lambd, eps=1e-4):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, x):
        return self.lambd(x) + self.eps


class MLP(nn.Module):
    def __init__(
        self,
        input_shape,
        dims=(500, 300, 200, 150),
        activation=nn.ReLU(),
        last_activation=None,
    ):
        super(MLP, self).__init__()
        if last_activation is None:
            last_activation = activation
        self.dims = dims
        self.first_fc = nn.Linear(input_shape[0], dims[0])
        self.first_activation = activation

        more_hidden = []
        if len(self.dims) > 2:
            for index in range(1, len(self.dims) - 1):
                more_hidden.append(nn.Linear(self.dims[index - 1], self.dims[index]))
                # more_hidden.append(activation)
                more_hidden.append(nn.ReLU())

        self.more_hidden = nn.ModuleList(more_hidden)

        self.last_fc = nn.Linear(dims[-2], dims[-1])
        self.last_activation = last_activation

    def forward(self, x):
        output = self.first_fc(x)
        output = self.first_activation(output)
        if self.more_hidden:
            for layer in self.more_hidden:
                output = layer(output)
        output = self.last_fc(output)
        output = self.last_activation(output)
        return output


# model = MLP((5665,),dims=(100,50)).cuda()
# summary(model,(5665,))


# #input_shape -> BatchSize, Channels, Height, Width
class CNN(nn.Module):
    def __init__(
        self,
        input_shape=(-1, 6000, 3),
        activation=nn.ReLU(),
        downsample=1,
        mlp_dims=(500, 300, 200, 150),
        eps=1e-8,
    ):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.activation = activation
        self.downsample = downsample
        self.mlp_dims = mlp_dims
        self.eps = eps

        self.lambda_layer_1 = LambdaLayer(
            lambda t: t
            / (
                torch.max(
                    torch.max(torch.abs(t), dim=1, keepdim=True).values,
                    dim=2,
                    keepdim=True,
                ).values
                + self.eps
            )
        )
        self.unsqueeze_layer1 = LambdaLayer(lambda t: torch.unsqueeze(t, dim=1))
        self.lambda_layer_2 = LambdaLayer(
            lambda t: torch.log(
                torch.max(torch.max(torch.abs(t), dim=1).values, dim=1).values
                + self.eps
            )
            / 100
        )
        self.unsqueeze_layer2 = LambdaLayer(lambda t: torch.unsqueeze(t, dim=1))
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, downsample), stride=(1, downsample)),
            nn.ReLU(),  # 用self.activation會有兩個ReLU
        )
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=(16, 3), stride=(1, 3)), nn.ReLU()
        )

        self.conv1d1 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=16), nn.ReLU())
        self.maxpooling = nn.MaxPool1d(2)

        self.conv1d2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=16), nn.ReLU())
        self.conv1d3 = nn.Sequential(nn.Conv1d(128, 32, kernel_size=8), nn.ReLU())
        self.conv1d4 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=8), nn.ReLU())
        self.conv1d5 = nn.Sequential(nn.Conv1d(32, 16, kernel_size=4), nn.ReLU())
        self.mlp = MLP((11665,), dims=self.mlp_dims)

    def forward(self, x):
        # print("intitial shape", x.size())
        output = self.lambda_layer_1(x)
        output = self.unsqueeze_layer1(output)

        scale = self.lambda_layer_2(x)
        # print("scale before:", scale.size())
        scale = self.unsqueeze_layer2(scale)
        # print("scale after:", scale.size())

        output = self.conv2d1(output)
        output = self.conv2d2(output)
        # print(output.shape)
        output = torch.squeeze(output, dim=-1)
        # print(output.shape)
        output = self.conv1d1(output)
        output = self.maxpooling(output)
        output = self.conv1d2(output)
        output = self.maxpooling(output)
        output = self.conv1d3(output)
        output = self.maxpooling(output)
        output = self.conv1d4(output)
        output = self.conv1d5(output)
        output = torch.flatten(output, start_dim=1)
        # print("scale:", scale.size())
        output = torch.cat((output, scale), dim=1)
        # print(output.size()[-1])
        output = self.mlp(output)
        # print("output:", output.size())

        return output


# model = CNN().cuda()
# CNN_input = torch.Tensor(np.random.rand(27000)*100).reshape(-1, 3000, 3).cuda()
# model(CNN_input)
# summary(model, (3000, 3))


class CNN_example(nn.Module):
    def __init__(self):
        super(CNN_example, self).__init__()
        # Convolution 1 , input_shape=(1,3000,3)
        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(1, 1), stride=(1, 1))
        self.relu1 = nn.ReLU()  # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv1d(
            in_channels=16, out_channels=8, kernel_size=1, stride=1, padding=0
        )
        self.relu2 = nn.ReLU()  # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool1d(kernel_size=3)
        # Convolution 3
        self.cnn3 = nn.Conv1d(
            in_channels=8, out_channels=2, kernel_size=1, stride=1, padding=0
        )
        self.relu3 = nn.ReLU()  # activation
        # Max pool 3
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        # Fully connected 1 ,#input_shape=(32*4*4)
        self.fc1 = nn.Linear(2 * 250, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)
        out = torch.squeeze(out, dim=-1)
        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        # # Max pool 2
        out = self.maxpool2(out)
        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        # out = out.view(out.size(0), -1)
        # # Linear function (readout)
        # out = self.fc1(out)
        # out = self.fc2(out)
        # out = self.fc3(out)
        return out


# sample=CNN_example().cuda()
# CNN_output = sample(torch.Tensor(np.random.randn(3000*3*250).reshape(-1,1, 3000, 3)).cuda())
# print("CNN_output: ", CNN_output.shape)


class PositionEmbedding(nn.Module):  # paper page11 B.2
    def __init__(
        self, wavelengths=((5, 30), (110, 123), (0.01, 5000)), emb_dim=500, **kwargs
    ):
        super(PositionEmbedding, self).__init__(**kwargs)
        # Format: [(min_lat, max_lat), (min_lon, max_lon), (min_depth, max_depth)]
        self.wavelengths = wavelengths
        self.emb_dim = emb_dim

        min_lat, max_lat = wavelengths[0]
        min_lon, max_lon = wavelengths[1]
        min_depth, max_depth = wavelengths[2]
        assert emb_dim % 10 == 0
        lat_dim = emb_dim // 5
        lon_dim = emb_dim // 5
        depth_dim = emb_dim // 10

        self.lat_coeff = (
            2
            * np.pi
            * 1.0
            / min_lat
            * ((min_lat / max_lat) ** (np.arange(lat_dim) / lat_dim))
        )
        self.lon_coeff = (
            2
            * np.pi
            * 1.0
            / min_lon
            * ((min_lon / max_lon) ** (np.arange(lon_dim) / lon_dim))
        )
        self.depth_coeff = (
            2
            * np.pi
            * 1.0
            / min_depth
            * ((min_depth / max_depth) ** (np.arange(depth_dim) / depth_dim))
        )

        lat_sin_mask = np.arange(emb_dim) % 5 == 0
        # 0~emb_dim % 5==0 -> True --> 一堆True False的矩陣
        # 共500個T F
        lat_cos_mask = np.arange(emb_dim) % 5 == 1
        lon_sin_mask = np.arange(emb_dim) % 5 == 2
        lon_cos_mask = np.arange(emb_dim) % 5 == 3
        depth_sin_mask = np.arange(emb_dim) % 10 == 4
        depth_cos_mask = np.arange(emb_dim) % 10 == 9

        self.mask = np.zeros(emb_dim)
        self.mask[lat_sin_mask] = np.arange(lat_dim)
        # mask範圍共1000個，lat_sin_mask裡面有200個True，若是True就按照順序把np.arange(lat_dim)塞進去
        self.mask[lat_cos_mask] = lat_dim + np.arange(lat_dim)
        self.mask[lon_sin_mask] = 2 * lat_dim + np.arange(lon_dim)
        self.mask[lon_cos_mask] = 2 * lat_dim + lon_dim + np.arange(lon_dim)
        self.mask[depth_sin_mask] = 2 * lat_dim + 2 * lon_dim + np.arange(depth_dim)
        self.mask[depth_cos_mask] = (
            2 * lat_dim + 2 * lon_dim + depth_dim + np.arange(depth_dim)
        )
        self.mask = self.mask.astype("int32")

    def forward(self, x):
        lat_base = (
            x[:, :, 0:1].cuda() * torch.Tensor(self.lat_coeff).cuda()
        )  # 這裡沒用到cuda!!
        lon_base = x[:, :, 1:2].cuda() * torch.Tensor(self.lon_coeff).cuda()
        depth_base = x[:, :, 2:3].cuda() * torch.Tensor(self.depth_coeff).cuda()
        # print(self.lat_coeff.shape)
        # print(x[:, :, 0:1].shape, 888)
        # print(lat_base.shape, "555")
        output = torch.cat(
            [
                torch.sin(lat_base),
                torch.cos(lat_base),
                torch.sin(lon_base),
                torch.cos(lon_base),
                torch.sin(depth_base),
                torch.cos(depth_base),
            ],
            dim=-1,
        )
        # print(torch.Tensor(self.mask).shape)
        # print("output", output.shape)
        maskk = torch.from_numpy(np.array(self.mask)).long()
        index = (
            (maskk.unsqueeze(0).unsqueeze(0)).expand(x.shape[0], 1, self.emb_dim).cuda()
        )
        output = torch.gather(output, -1, index).cuda()
        return output


class Element_Wise(nn.Module):  # paper page11 B.2
    def __init__(
        self,
        waveforms_emb=torch.arange(1500).reshape(3, 1, 500),
    ):
        super(Element_Wise, self).__init__()
        self.waveforms_emb = waveforms_emb

    def forward(self, x):  # x = coords_emb
        emb = torch.add(self.waveforms_emb, x)

        return emb


class test:
    def __init__(self, d_model, dropout):
        self.dropout = dropout
        self.d_model = d_model

    def forward(self, x):
        x = x + self.dropout
        return x


test(5, 6).forward(1)

# posi_dim = 500
# wavelength = ((min_lat, max_lat), (min_lon, max_lon), (min_depth, max_depth))
# coords_emb = PositionEmbedding(
#     wavelengths=wavelength, emb_dim=posi_dim)(PositionEmbedding_output).cuda()
# Element_Wise_output = Element_Wise(waveforms_emb=CNN_output)(coords_emb).cuda()
# print("Element_Wise_output: ", Element_Wise_output.shape)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model=150,
        nhead=10,
        batch_first=True,
        activation="gelu",
        dropout=0.0,
        dim_feedforward=1000,
    ):
        super(TransformerEncoder, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=batch_first,
            activation=activation,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        ).cuda()
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 6).cuda()

    def forward(self, x, src_key_padding_mask=None):
        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # out = out.view(out.size(0), -1)
        return out


# model=TransformerEncoder()
# a=torch.rand(32, 40, 150).cuda()
# model(a).shape


# batch_size = 3
# dim_input = 500
# src = torch.rand(batch_size, 1, dim_input).cuda()  # input data
# Transformer_output = TransformerEncoder().forward(PositionalEncoding_output).cuda()
# print("Transformer_output: ", Transformer_output.shape)
class MDN(nn.Module):
    def __init__(self, input_shape=(150,), n_hidden=20, n_gaussians=5):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(nn.Linear(input_shape[0], n_hidden), nn.Tanh())
        self.z_weight = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        weight = nn.functional.softmax(self.z_weight(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return weight, sigma, mu


def gaussian_distribution(y, mu, sigma):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0 * np.pi)  # normalization factor for Gaussians
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI


def mdn_loss_fn(pi, sigma, mu, y):
    result = gaussian_distribution(y, mu, sigma) * pi
    # print(result.shape)
    result = torch.sum(result, dim=1)
    # print(result.shape)
    result = -torch.log(result)
    return torch.mean(result)


class full_model(nn.Module):
    def __init__(
        self,
        model_CNN,
        model_Position,
        model_Transformer,
        model_mlp,
        model_MDN,
        max_station=25,
        pga_targets=15,
        emb_dim=150,
    ):
        super(full_model, self).__init__()
        self.model_CNN = model_CNN
        self.model_Position = model_Position
        self.model_Transformer = model_Transformer
        self.model_mlp = model_mlp
        self.model_MDN = model_MDN
        self.max_station = max_station
        self.pga_targets = pga_targets
        self.emb_dim = emb_dim

    def forward(self, data):
        CNN_output = self.model_CNN(
            torch.DoubleTensor(data[0].reshape(-1, 6000, 3)).float().cuda()
        )
        CNN_output_reshape = torch.reshape(
            CNN_output, (-1, self.max_station, self.emb_dim)
        )

        emb_output = self.model_Position(
            torch.DoubleTensor(data[1].reshape(-1, 1, 3)).float().cuda()
        )
        emb_output = emb_output.reshape(-1, self.max_station, self.emb_dim)
        # data[1] 做一個padding mask [batchsize, station number (25)], value: True, False (True: should mask)
        station_pad_mask = data[1] == 0
        station_pad_mask = torch.all(station_pad_mask, 2)

        pga_pos_emb_output = self.model_Position(
            torch.DoubleTensor(data[2].reshape(-1, 1, 3)).float().cuda()
        )
        pga_pos_emb_output = pga_pos_emb_output.reshape(
            -1, self.pga_targets, self.emb_dim
        )
        # data[2] 做一個padding mask [batchsize, PGA_target (15)], value: True, False (True: should mask)
        target_pad_mask = torch.ones_like(
            data[2], dtype=torch.bool
        )  # 避免 target position 在self-attention互相影響結果
        # target_pad_mask= data[2] ==0
        target_pad_mask = torch.all(target_pad_mask, 2)

        # concat two mask, [batchsize, station_number+PGA_target (40)], value: True, False (True: should mask)
        pad_mask = torch.cat((station_pad_mask, target_pad_mask), dim=1).cuda()

        add_PE_CNNoutput = torch.add(CNN_output_reshape, emb_output)
        transformer_input = torch.cat((add_PE_CNNoutput, pga_pos_emb_output), dim=1)
        transformer_output = self.model_Transformer(transformer_input, pad_mask)

        mlp_input = transformer_output[:, -self.pga_targets :, :].cuda()

        mlp_output = self.model_mlp(mlp_input)

        weight, sigma, mu = self.model_MDN(mlp_output)

        return weight, sigma, mu


class MixtureOutput(nn.Module):
    def __init__(self, input_shape=None, n=4, d=1, activation=nn.ReLU(), eps=1e-4):
        super(MixtureOutput, self).__init__()
        self.activation = activation
        self.n = n
        self.d = d
        self.eps = eps
        self.alpha_layer = nn.Linear(input_shape[0], self.n * self.d)
        self.mu_layer = nn.Linear(input_shape[0], self.n * self.d)
        self.sigma_layer = nn.Linear(input_shape[0], self.n * self.d)

    def forward(self, x):
        x = torch.flatten(x)  # not sure 硬加的
        alpha = self.alpha_layer(x)
        alpha = self.activation(alpha)
        print(alpha.shape)
        alpha = torch.reshape(alpha, (self.n, self.d))
        print(alpha.shape)
        mu = self.mu_layer(x)
        mu = self.activation(mu)
        mu = torch.reshape(mu, (self.n, self.d))
        print(mu.shape)
        sigma = self.sigma_layer(x)
        sigma = LambdaLayer(lambda x: x, self.eps)(
            sigma
        )  # Add epsilon to avoid division by 0
        sigma = torch.reshape(sigma, (self.n, self.d))
        print(sigma.shape)
        out = torch.cat([alpha, mu, sigma], dim=1)

        return out


# # bbb = torch.arange(10).reshape(1, 10)
# # bbb = bbb.type(torch.FloatTensor)
# MixtureOutput_model = MixtureOutput((500,), n=5, d=1).cuda()
# print(MixtureOutput_model)
# MixtureOutput_output = MixtureOutput_model(Transformer_output[0, :, :])
# print("MixtureOutput_output: ", MixtureOutput_output.shape)


# y_pred = MixtureOutput_output(全部測站的)
def mixture_density_loss(y_true, y_pred, eps=1e-6, d=1, mean=True, print_shapes=True):
    if print_shapes:
        print(f"True: {y_true.shape}")
        print(f"Pred: {y_pred.shape}")
    alpha = y_pred[:, :, 0]
    density = torch.ones_like(
        y_pred[:, :, 0]
    )  # Create an array of ones of correct size
    for j in range(d):
        mu = y_pred[:, :, j + 1]
        sigma = y_pred[:, :, j + 1 + d]
        sigma = torch.maximum(sigma, eps)
        density *= (
            1
            / (np.sqrt(2 * np.pi) * sigma)
            * torch.exp(-((y_true[:, j] - mu) ** 2) / (2 * sigma**2))
        )
    density *= alpha
    density = torch.sum(density, dim=1)
    density += eps
    loss = -torch.log(density)

    if mean:
        return torch.mean(loss)
    else:
        return loss


def concate_mixture_output(data):
    x = np.linspace(-7, 1, 100)
    for station in range(data.shape[0]):
        mix_model = np.zeros(100)
        for i in range(data.shape[1]):
            weight = data[station, i, 0]
            mu = data[station, i, 1]
            sigma = data[station, i, 2]

            pdf = weight * stats.norm.pdf(x, mu, sigma)
            mix_model += pdf
            plt.plot(x, pdf, color="grey", alpha=0.2)
        plt.plot(x, mix_model, color="red")
        plt.show()


# x = np.linspace(-7,7, 100)
# plt.plot(x, stats.norm.pdf(x,2,2),'r-', lw=5, alpha=0.6, label='norm pdf')
