import torch
import torch.nn as nn
import spconv

from ....ops.RandLA.LAmodule import SharedMLP,LocalSpatialEncoding,AttentivePooling,LocalFeatureAggregation
try:
    from torch_points import knn
except (ModuleNotFoundError, ImportError):
    from torch_points_kernels import knn
import spconv
class devoxelization(nn.Module):
    def __init__(self,model_cfg,input_channels,indice_key=None):
        super(devoxelization, self).__init__()
        self.cfg = model_cfg
        self.devoxel = spconv.SparseConv3d(input_channels,4,1,indice=indice_key)
    def forward(self,batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        batch_size = batch_dict['batch_size']
        batch_idx = batch_dict['voxel_coords']
        multi_scale_3d_features = batch_dict['multi_scale_3d_features']
        reshaped = {}
        for i in multi_scale_3d_features.keys():
            devoxelization = self.devoxel(multi_scale_3d_features[i].shape[-1], indice=i + '_reshape')
            reshaped.update({
                i+'_reshape',devoxelization(multi_scale_3d_features[i])
            })


class ShareMLP(SharedMLP):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, transpose=False, padding_mode='zeros', bn=False, activation_fn=None):
        r'''

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param transpose:
        :param padding_mode:
        :param bn:
        :param activation_fn:

        It is wired that the position of the transpose convolution
        '''

        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d  # TODO: need to be changed to sparse conv

        self.conv = conv_fn(in_channels, out_channels, kernel_size, stride=stride,
                            padding_mode=padding_mode)  ##TODO:Why is there a convolution?
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True,
                             activation_fn=nn.ReLU())  # TODO:This part needs to be changed. Needs to be equal to the output of the sparse convolution.

        self.device = device

    def forward(self, coords: torch.Tensor, features: torch.Tensor, knn_output):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud [d: feature of demonsions]
            neighbors: tuple

            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx, dist = knn_output
        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx)  # shape (B, 3, N, K)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()

        # relative point position encoding
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3).to(self.device)
        return torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)

class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True)  # shape (B, d_in, N, 1)

        return self.mlp(features)
