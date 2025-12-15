import math

import torch
from torch import nn
from torch_scatter import scatter



class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies = 10, n_space = 3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class CSPLayer(nn.Module):
    """ Message passing layer for cspnet."""

    def __init__(
        self,
        hidden_dim=128,
        act_fn=nn.SiLU(),
        dis_emb=None,
        ln=False,
    ):
        super(CSPLayer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = True
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn)
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def edge_model(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff = None):

        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1,-2)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = torch.cat([hi, hj, lattice_ips_flatten_edges, frac_diff], dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):

        agg = scatter(edge_features, edge_index[1], dim = 0, reduce='mean', dim_size=node_features.shape[0])
        agg = torch.cat([node_features, agg], dim = 1)
        out = self.node_mlp(agg)
        return out

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff):

        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff)
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output, edge_features


class EGNN(nn.Module):
    def __init__(
        self,
        hidden_dim = 128,
        latent_dim = 256,
        num_layers = 4,
        max_atoms = 100,
        act_fn = 'silu',
        dis_emb = 'sin',
        num_freqs = 10,
        ln = False,
        pred_scalar = False,
        **kwargs,
    ):
        super(EGNN, self).__init__()

        self.t_embedding = SinusoidalTimeEmbeddings(latent_dim)
        self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies = num_freqs)
        elif dis_emb == 'none':
            self.dis_emb = None
        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i, CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln)
            )            
        self.num_layers = num_layers
        self.coord_out = nn.Linear(hidden_dim, 3, bias = False)
        self.ln = ln
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.pred_scalar = pred_scalar
        if not self.pred_scalar:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.act_fn,
                nn.Linear(hidden_dim, 1, bias=False),
            )
            
    def coor_model(self, coord, edge_index, coord_diff, edge_features):
        trans = coord_diff * self.coord_mlp(edge_features)
        agg = scatter(trans, edge_index[1], dim = 0, reduce='mean', dim_size=coord.shape[0])
        return agg
    
    def forward(self, atom_types, t, frac_coords, edge_index, lattices, node2graph, **kargs):
        shift = frac_coords[edge_index[1]] - frac_coords[edge_index[0]]
        frac_diff = shift - torch.floor(shift + 0.5 + 1e-4)
        edge2graph = node2graph[edge_index[0]]
        
        t = self.t_embedding(t)
        node_features = self.node_embedding(atom_types - 1)
        node_features = torch.cat([node_features, t], dim=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(0, self.num_layers):
            node_features, edge_features = self._modules["csp_layer_%d" % i](
                node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        if self.pred_scalar:
            coord_out = self.coord_out(node_features)
        else:
            coord_out = self.coor_model(frac_coords, edge_index, frac_diff, edge_features)

        return coord_out
