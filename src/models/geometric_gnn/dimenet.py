###########################################################################################
# DimeNet++
# Paper: Directional Message Passing for Molecular Graphs
# Authors: Johannes Gasteiger, Janek Groß, Stephan Günnemann
# Date: 6 Mar 2020
# Comments: International Conference on Learning Representations (ICLR) 2020 (spotlight)
# Repo: https://github.com/gasteigerjo/dimenet (MIT License)
# Paper: https://arxiv.org/abs/2003.03123
# This program is distributed under an MIT License
###########################################################################################

import torch
from torch_scatter import scatter
from typing import Callable, Union
from torch_geometric.nn.models.dimenet import DimeNetPlusPlus

class DimeNetPPModel(DimeNetPlusPlus):
    def __init__(
        self, 
        hidden_channels: int = 128, 
        in_dim: int = 1,
        out_dim: int = 1, 
        num_layers: int = 4, 
        int_emb_size: int = 64, 
        basis_emb_size: int = 8, 
        out_emb_channels: int = 256, 
        num_spherical: int = 7, 
        num_radial: int = 6, 
        cutoff: float = 10, 
        max_num_neighbors: int = 32, 
        envelope_exponent: int = 5, 
        num_before_skip: int = 1, 
        num_after_skip: int = 2, 
        num_output_layers: int = 3, 
        act: Union[str, Callable] = 'swish'
    ):
        super().__init__(hidden_channels, out_dim, num_layers, int_emb_size, basis_emb_size, out_emb_channels, num_spherical, num_radial, cutoff, max_num_neighbors, envelope_exponent, num_before_skip, num_after_skip, num_output_layers, act)

    def forward(self, batch):
        
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            batch.edge_index, num_nodes=batch.token.size(0))

        # Calculate distances.
        dist = (batch.pos[i] - batch.pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = batch.pos[idx_i]
        pos_ji, pos_ki = batch.pos[idx_j] - pos_i, batch.pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(batch.token, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=batch.pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)

        return P.sum(dim=0) if batch is None else scatter(P, batch.batch, dim=0)
