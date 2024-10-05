import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch_geometric.nn import MessagePassing
from torch_sparse import matmul



class ResidualVectorQuantizer(nn.Module):

    def __init__(self, codebook_num, codebook_size, codebook_dim, dist ='cos', tau=0.2, vq_ema=False):
        super().__init__()
        self.codebook_num = codebook_num
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.dist = dist
        self.tau = tau

        self.vq_ema = vq_ema

        if self.vq_ema:
            self.vq_layers = torch.nn.ModuleList([
                EMAVectorQuantizer(codebook_size=self.codebook_size, codebook_dim=self.codebook_dim, dist=self.dist,
                                tau=self.tau)
                for _ in range(self.codebook_num)
            ])
        else:
            self.vq_layers = torch.nn.ModuleList([
                VectorQuantizer(codebook_size=self.codebook_size, codebook_dim=self.codebook_dim, dist=self.dist,
                                tau=self.tau)
                for _ in range(self.codebook_num)
            ])


    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.cat(all_codebook, dim=0)

    @torch.no_grad()
    def get_codes(self, x):
        all_codes = []
        residual = x
        for i in range(len(self.vq_layers)):
            x_res, _, codes = self.vq_layers[i](residual)
            residual = residual - x_res
            all_codes.append(codes)

        all_codes = torch.stack(all_codes, dim=-1)

        return all_codes


    def forward(self, x):

        all_com_losses = []
        all_codes = []


        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, com_loss, codes = quantizer(residual)
            residual = residual - x_res

            x_q = x_q + x_res

            all_com_losses.append(com_loss)
            all_codes.append(codes)

        mean_com_loss = torch.stack(all_com_losses).mean()
        all_codes = torch.stack(all_codes, dim=-1)

        return x_q, mean_com_loss, all_codes


class ProductVectorQuantizer(nn.Module):

    def __init__(self, codebook_num, codebook_size, codebook_dim, dist ='cos', tau=0.2, vq_ema=False):
        super().__init__()
        self.codebook_num = codebook_num
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.dist = dist
        self.tau = tau
        self.vq_ema = vq_ema

        if self.vq_ema:
            self.vq_layers = torch.nn.ModuleList([
                EMAVectorQuantizer(codebook_size=self.codebook_size, codebook_dim=self.codebook_dim, dist=self.dist,
                                tau=self.tau)
                for _ in range(self.codebook_num)
            ])
        else:
            self.vq_layers = torch.nn.ModuleList([
                VectorQuantizer(codebook_size=self.codebook_size, codebook_dim=self.codebook_dim, dist=self.dist,
                                tau=self.tau)
                for _ in range(self.codebook_num)
            ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.cat(all_codebook, dim=0)

    @torch.no_grad()
    def get_codes(self, x):
        all_codes = []
        x_chunk = torch.chunk(x, self.codebook_num, dim=-1)
        for idx, quantizer in enumerate(self.vq_layers):
            _, _, codes = quantizer(x_chunk[idx])
            all_codes.append(codes)

        all_codes = torch.stack(all_codes, dim=-1)

        return all_codes

    def forward(self, x):

        all_com_losses = []
        all_codes = []
        all_x_q = []

        x_chunk = torch.chunk(x, self.codebook_num, dim=-1)
        for idx, quantizer in enumerate(self.vq_layers):
            x_q, com_loss, codes = quantizer(x_chunk[idx])

            all_x_q.append(x_q)
            all_com_losses.append(com_loss)
            all_codes.append(codes)

        all_x_q = torch.cat(all_x_q, dim=-1)
        mean_com_loss = torch.stack(all_com_losses).mean()
        all_codes = torch.stack(all_codes, dim=-1)

        return all_x_q, mean_com_loss, all_codes




def kmeans(
        samples,
        num_clusters,
):

    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters=num_clusters).fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)

    return tensor_centers


def replace_nan(x):

    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(-distances / epsilon).t()

    B = Q.shape[1]
    K = Q.shape[0]  # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = torch.clamp(torch.sum(Q), min=1e-5)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()

class VectorQuantizer(nn.Module):

    def __init__(self, codebook_size, codebook_dim, dist ='cos', tau=0.2, sk_epsilon=0.005, sk_iters=3):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.dist = dist
        self.tau = tau
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.initted = False
        self.embedding.weight.data.zero_()


    def get_codebook(self):
        return self.embedding.weight

    def get_quantize_emb(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.codebook_size,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True


    @staticmethod
    def center_distance_for_constraint(distances):

        distances = replace_nan(distances)
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        if torch.isnan(amplitude):
            amplitude = 1e-5

        amplitude = torch.clamp(amplitude, min=1e-5)

        if amplitude <= 0:
            amplitude = 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x):
        # Flatten input
        latent = x.view(-1, self.codebook_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the distances between latent and Embedded weights
        if self.dist.lower() == 'l2':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.weight.t())
        elif self.dist.lower() == 'cos':
            cos_sim = torch.matmul(F.normalize(latent, dim=-1), F.normalize(self.embedding.weight, dim=-1).t())
            d = (1 - cos_sim) / self.tau
        else:
            raise NotImplementedError

        if torch.isnan(d).any() or torch.isinf(d).any():
            print(f"Distance have nan/inf values.")


        if self.sk_epsilon > 0 and self.training:

            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)



        # compute loss for embedding
        if self.dist.lower() == 'l2':
            codebook_loss = F.mse_loss(x_q, x.detach())
            # commitment_loss = F.mse_loss(x_q.detach(), x)
            # loss = codebook_loss + self.beta * commitment_loss
            loss = codebook_loss
        elif self.dist.lower() == 'cos':
            loss = F.cross_entropy(cos_sim / self.tau, indices.detach())
        else:
            raise NotImplementedError


        x_q = x + (x_q - x).detach()


        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices





class EMAVectorQuantizer(nn.Module):

    def __init__(self, codebook_size, codebook_dim, dist='cos', tau=0.2,
                 sk_epsilon=0.005, sk_iters=3, decay=0.99,  eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.decay = decay
        self.eps = eps
        self.dist = dist
        self.tau = tau

        embedding = torch.randn(self.codebook_size, self.codebook_dim)

        self.register_buffer('embedding', embedding)
        self.register_buffer('embedding_avg', embedding.clone())
        self.register_buffer('cluster_size', torch.ones(codebook_size))

        self.initted = False


    def get_codebook(self):
        return self.embedding

    def get_quantize_emb(self, indices, shape=None):
        # get quantized latent vectors
        z_q = F.embedding(indices, self.embedding)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def _tile(self, x):
        n, d = x.shape
        if n < self.codebook_size:
            n_repeats = (self.codebook_size + n - 1) // n
            std = 0.01 / np.sqrt(d)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x


    def init_emb(self, data):

        centers = kmeans(
            data,
            self.codebook_size,
        )

        self.embedding.data.copy_(centers)
        self.embedding_avg.data.copy_(centers)
        self.initted = True


    def moving_average(self, moving_avg, new, decay):
        moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

    def laplace_smoothing(self, x, n_categories, eps=1e-5):
        return (x + eps) / (x.sum() + n_categories * eps)

    @staticmethod
    def center_distance_for_constraint(distances):

        distances = replace_nan(distances)
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        if torch.isnan(amplitude):
            amplitude = 1e-5

        amplitude = torch.clamp(amplitude, min=1e-5)

        if amplitude <= 0:
            amplitude = 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances


    def forward(self, x):
        # Flatten input
        latent = x.view(-1, self.codebook_dim)


        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the distances between latent and Embedded weights
        if self.dist.lower() == 'l2':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.t())
        elif self.dist.lower() == 'cos':
            cos_sim = torch.matmul(F.normalize(latent, dim=-1), F.normalize(self.embedding, dim=-1).t())
            d = (1 - cos_sim) / self.tau
        else:
            raise NotImplementedError


        if torch.isnan(d).any() or torch.isinf(d).any():
            print(f"Distance have nan/inf values.")


        if self.sk_epsilon > 0 and self.training:
            d_norm = self.center_distance_for_constraint(d)
            d_norm = d_norm.double()
            Q = sinkhorn_algorithm(d_norm, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(d, dim=-1)

        x_q = F.embedding(indices, self.embedding).view(x.shape)


        loss = torch.zeros(1).to(x.device)

        if self.training:
            embedding_onehot = F.one_hot(indices, self.codebook_size).type(latent.dtype)
            embedding_sum = embedding_onehot.t() @ latent
            self.moving_average(self.cluster_size, embedding_onehot.sum(0), self.decay)
            self.moving_average(self.embedding_avg, embedding_sum, self.decay)
            n = self.cluster_size.sum()
            weights = self.laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * n
            embedding_normalized = self.embedding_avg / weights.unsqueeze(1)
            self.embedding.data.copy_(embedding_normalized)

            # temp = self._tile(latent)
            # temp = temp[torch.randperm(temp.size(0))][:self.codebook_size]
            # usage = (self.cluster_size.view(self.codebook_size, 1) >= 1).float()
            # self.embedding.data.mul_(usage).add_(temp * (1 - usage))

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices



class LightGCNConv(MessagePassing):
    def __init__(self, dim):
        super(LightGCNConv, self).__init__(aggr='add')
        self.dim = dim

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)


class BipartiteGCNConv(MessagePassing):
    def __init__(self, dim):
        super(BipartiteGCNConv, self).__init__(aggr='add')
        self.dim = dim

    def forward(self, x, edge_index, edge_weight, size):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.dim)


class BiGNNConv(MessagePassing):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.in_channels, self.out_channels = in_channels, out_channels
        self.lin1 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)
        self.lin2 = torch.nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, edge_index, edge_weight):
        x_prop = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        x_trans = self.lin1(x_prop + x)
        x_inter = self.lin2(torch.mul(x_prop, x))
        return x_trans + x_inter

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class SRGNNConv(MessagePassing):
    def __init__(self, dim):
        # mean aggregation to incorporate weight naturally
        super(SRGNNConv, self).__init__(aggr='mean')

        self.lin = torch.nn.Linear(dim, dim)

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)


class SRGNNCell(nn.Module):
    def __init__(self, dim):
        super(SRGNNCell, self).__init__()

        self.dim = dim
        self.incomming_conv = SRGNNConv(dim)
        self.outcomming_conv = SRGNNConv(dim)

        self.lin_ih = nn.Linear(2 * dim, 3 * dim)
        self.lin_hh = nn.Linear(dim, 3 * dim)

        self._reset_parameters()

    def forward(self, hidden, edge_index):
        input_in = self.incomming_conv(hidden, edge_index)
        reversed_edge_index = torch.flip(edge_index, dims=[0])
        input_out = self.outcomming_conv(hidden, reversed_edge_index)
        inputs = torch.cat([input_in, input_out], dim=-1)

        gi = self.lin_ih(inputs)
        gh = self.lin_hh(hidden)
        i_r, i_i, i_n = gi.chunk(3, -1)
        h_r, h_i, h_n = gh.chunk(3, -1)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def _reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
