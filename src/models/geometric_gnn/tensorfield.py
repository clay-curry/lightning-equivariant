###########################################################################################
# TensorfieldNet
# Paper: Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds
# Authors: Yi Liu, Limei Wang, Meng Liu, Yuchao Lin, Xuan Zhang, Bora Oztekin, Shuiwang Ji
# Date: 14 Mar 2022
# Comments: 
# Repo: 
# Paper: https://arxiv.org/abs/1802.08219
# This program is distributed under an GNU General Public License v3.0
###########################################################################################


import lightning.pytorch as L

import torch
from e3nn import o3
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool

from src.blocks.radial import RadialEmbeddingBlock
from src.blocks.tensorproduct import TensorProductConvLayer

class TFNModel(L.LightningModule):
    def __init__(
        self,
        r_max=10.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        num_layers=5,
        emb_dim=64,
        in_dim=1,
        out_dim=1,
        aggr="sum",
        pool="sum",
        residual=True,
        scalar_pred=True
    ):
        super().__init__()
        self.r_max = r_max
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.residual = residual
        self.scalar_pred = scalar_pred
        # Embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Embedding lookup for initial node features
        self.emb_in = torch.nn.Embedding(in_dim, emb_dim)

        self.convs = torch.nn.ModuleList()
        hidden_irreps = (sh_irreps * emb_dim).sort()[0].simplify()
        irrep_seq = [
            o3.Irreps(f'{emb_dim}x0e'),
            # o3.Irreps(f'{emb_dim}x0e + {emb_dim}x1o + {emb_dim}x2e'),
            # o3.Irreps(f'{emb_dim//2}x0e + {emb_dim//2}x0o + {emb_dim//2}x1e + {emb_dim//2}x1o + {emb_dim//2}x2e + {emb_dim//2}x2o'),
            hidden_irreps
        ]
        for i in range(num_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            conv = TensorProductConvLayer(
                in_irreps=in_irreps,
                out_irreps=out_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                hidden_dim=emb_dim,
                gate=True,
                aggr=aggr,
            )
            self.convs.append(conv)

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        if self.scalar_pred:
            # Predictor MLP
            self.pred = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, out_dim)
            )
        else:
            self.pred = torch.nn.Linear(hidden_irreps.dim, out_dim)
    
    def forward(self, batch):
        h = self.emb_in(batch.atoms)  # (n,) -> (n, d)

        # Edge features
        vectors = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        
        for conv in self.convs:
            # Message passing layer
            h_update = conv(h, batch.edge_index, edge_attrs, edge_feats)

            # Update node features
            h = h_update + F.pad(h, (0, h_update.shape[-1] - h.shape[-1])) if self.residual else h_update

        if self.scalar_pred:
            # Select only scalars for prediction
            h = h[:,:self.emb_dim]
        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        return self.pred(out)  # (batch_size, out_dim)
