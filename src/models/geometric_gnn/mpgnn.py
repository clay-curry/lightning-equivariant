# A boring example of a message passing network in torch_geometric

import torch
import lightning.pytorch as L
from torch_scatter import scatter
from torch.nn import Linear, ReLU, SiLU, Sequential
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool


class MPNNModel(L.LightningModule):
    def __init__(
        self,
        num_layers=5,
        emb_dim=128,
        in_dim=1,
        out_dim=1,
        activation="relu",
        norm="layer",
        aggr="sum",
        pool="sum",
        residual=True
    ):
        """Vanilla Message Passing GNN model
        
        Args:
            num_layers: (int) - number of message passing layers
            emb_dim: (int) - hidden dimension
            in_dim: (int) - initial node feature dimension
            out_dim: (int) - output number of classes
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
            pool: (str) - global pooling function (sum/mean)
            residual: (bool) - whether to use residual connections
        """
        super().__init__()

        # Embedding lookup for initial node features
        self.emb_in = torch.nn.Embedding(in_dim, emb_dim)

        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, activation, norm, aggr))

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        # Predictor MLP
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, out_dim)
        )
        self.residual = residual

    def forward(self, batch):
        
        h = self.emb_in(batch.token)  # (n,) -> (n, d)
        
        for conv in self.convs:
            # Message passing layer and residual connection
            h = h + conv(h, batch.edge_index) if self.residual else conv(h, batch.edge_index)

        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        return self.pred(out)  # (batch_size, out_dim)


class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim, activation="relu", norm="layer", aggr="add"):
        """Vanilla Message Passing GNN layer
        
        Args:
            emb_dim: (int) - hidden dimension `d`
            activation: (str) - non-linearity within MLPs (swish/relu)
            norm: (str) - normalisation layer (layer/batch)
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.activation = {"swish": SiLU(), "relu": ReLU()}[activation]
        self.norm = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d}[norm]

        # MLP `\psi_h` for computing messages `m_ij`
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )
        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
            Linear(emb_dim, emb_dim),
            self.norm(emb_dim),
            self.activation,
        )

    def forward(self, h, edge_index):
        """
        Args:
            h: (n, d) - initial node features
            edge_index: (e, 2) - pairs of edges (i, j)
        Returns:
            out: (n, d) - updated node features
        """
        out = self.propagate(edge_index, h=h)
        return out

    def message(self, h_i, h_j):
        # Compute messages
        msg = torch.cat([h_i, h_j], dim=-1)
        msg = self.mlp_msg(msg)
        return msg

    def aggregate(self, inputs, index):
        # Aggregate messages
        msg_aggr = scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
        return msg_aggr

    def update(self, aggr_out, h):
        upd_out = self.mlp_upd(torch.cat([h, aggr_out], dim=-1))
        return upd_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"
