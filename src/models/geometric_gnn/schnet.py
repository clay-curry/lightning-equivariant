###########################################################################################
# SE3 Transformer
# Paper: SchNet: A continuous-filter convolutional neural network for modeling quantum interactions
# Authors: Kristof T. Schütt, Pieter-Jan Kindermans, Huziel E. Sauceda, Stefan Chmiela, Alexandre Tkatchenko, Klaus-Robert Müller
# Date: 26 Jun 2017 
# Comments: Advances in Neural Information Processing Systems 30 (2017), pp. 992-1002
# Repo 1: https://github.com/atomistic-machine-learning/schnetpack
# Paper: https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html
# This program is distributed under an MIT License
###########################################################################################


import torch
import lightning.pytorch as L

from typing import Optional
from torch_scatter import scatter
from torch_geometric.nn import SchNet


class SchNetModel(SchNet, L.LightningModule):
    def __init__(
        self, 
        hidden_channels: int = 128, 
        in_dim: int = 1,
        out_dim: int = 1, 
        num_filters: int = 128, 
        num_layers: int = 6,
        num_gaussians: int = 50, 
        cutoff: float = 10, 
        max_num_neighbors: int = 32, 
        readout: str = 'add', 
        dipole: bool = False,
        mean: Optional[float] = None, 
        std: Optional[float] = None, 
        atomref: Optional[torch.Tensor] = None,
    ):
        super().__init__(hidden_channels, num_filters, num_layers, num_gaussians, cutoff, max_num_neighbors, readout, dipole, mean, std, atomref)

        # Overwrite atom embedding and final predictor
        self.lin2 = torch.nn.Linear(hidden_channels // 2, out_dim)

    def forward(self, batch):
        h = self.embedding(batch.token)

        row, col = batch.edge_index
        edge_weight = (batch.pos[row] - batch.pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, batch.edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        out = scatter(h, batch.batch, dim=0, reduce=self.readout)
        return out
