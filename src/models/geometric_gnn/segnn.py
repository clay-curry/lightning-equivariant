###########################################################################################
# Steerable E(3) Graph and Convolutional Neural Networks
# Paper: Geometric and Physical Quantities improve E(3) Equivariant Message Passing
# Authors: Johannes Brandstetter, Rob Hesselink, Elise van der Pol, Erik Bekkers and Max Welling.
# Date: 08 May 2022
# Comments: Published at ICLR 2022 (Spotlight paper)
# Repo: https://github.com/RobDHess/Steerable-E3-GNN (MIT License)
# Paper: https://openreview.net/forum?id=_xwr8gOBeV1
# This program is distributed under an MIT License
###########################################################################################

import torch
import lightning.pytorch as L

import numpy as np
from torch import nn
from math import sqrt
import torch.nn.functional as F
from e3nn.nn import BatchNorm, Gate
from e3nn.o3 import Irreps, FullyConnectedTensorProduct
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool


class SEConv(L.LightningModule):
    """Steerable E(3) equivariant (non-linear) convolutional network"""

    def __init__(
        self,
        input_irreps=None,
        hidden_irreps=None,
        output_irreps=None,
        edge_attr_irreps=None,
        node_attr_irreps=None,
        num_layers=1,
        norm=None,
        pool="avg",
        task="graph",
        additional_message_irreps=None,
        conv_type="linear",
    ):
        input_irreps = Irreps(input_irreps) if input_irreps is not None else Irreps("1x0e")
        hidden_irreps = Irreps(hidden_irreps) if hidden_irreps is not None else Irreps("1x0e")
        output_irreps = Irreps(output_irreps) if output_irreps is not None else Irreps("1x0e")
        edge_attr_irreps = Irreps(edge_attr_irreps) if edge_attr_irreps is not None else Irreps("1x0e")
        node_attr_irreps = Irreps(node_attr_irreps) if node_attr_irreps is not None else Irreps("1x0e")

        super().__init__()
        self.task = task

        # Create network, embedding first
        self.embedding_layer = O3TensorProduct(
            input_irreps, hidden_irreps, node_attr_irreps
        )

        # Message passing layers.
        layers = []
        for i in range(num_layers):
            layers.append(
                SEConvLayer(
                    hidden_irreps,
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                    conv_type=conv_type,
                )
            )
        self.layers = nn.ModuleList(layers)

        # Prepare for output irreps, since the attrs will disappear after pooling
        if task == "graph":
            pooled_irreps = (
                (output_irreps * hidden_irreps.num_irreps).simplify().sort().irreps
            )
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, pooled_irreps, node_attr_irreps
            )
            self.post_pool1 = O3TensorProductSwishGate(pooled_irreps, pooled_irreps)
            self.post_pool2 = O3TensorProduct(pooled_irreps, output_irreps)
            self.init_pooler(pool)
        elif task == "node":
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, output_irreps, node_attr_irreps
            )

    def init_pooler(self, pool):
        """Initialise pooling mechanism"""
        if pool == "avg":
            self.pooler = global_mean_pool
        elif pool == "sum":
            self.pooler = global_add_pool

    def catch_isolated_nodes(self, graph):
        """Isolated nodes should also obtain attributes"""
        if (
            graph.contains_isolated_nodes()
            and graph.edge_index.max().item() + 1 != graph.num_nodes
        ):
            nr_add_attr = graph.num_nodes - (graph.edge_index.max().item() + 1)
            add_attr = graph.node_attr.new_tensor(
                np.zeros((nr_add_attr, graph.node_attr.shape[-1]))
            )
            graph.node_attr = torch.cat((graph.node_attr, add_attr), -2)
        # Trivial irrep value should always be 1 (is automatically so for connected nodes, but isolated nodes are now 0)
        graph.node_attr[:, 0] = 1.0

    def forward(self, graph):
        """SEGNN forward pass"""
        x, pos, edge_index, edge_attr, node_attr, batch = (
            graph.x,
            graph.pos,
            graph.edge_index,
            graph.edge_attr,
            graph.node_attr,
            graph.batch,
        )
        try:
            additional_message_features = graph.additional_message_features
        except AttributeError:
            additional_message_features = None

        self.catch_isolated_nodes(graph)

        # Embed
        x = self.embedding_layer(x, node_attr)

        # Pass messages
        for layer in self.layers:
            x = layer(
                x, edge_index, edge_attr, node_attr, batch, additional_message_features
            )

        # Pre pool
        x = self.pre_pool1(x, node_attr)
        x = self.pre_pool2(x, node_attr)

        if self.task == "graph":
            # Pool over nodes
            x = self.pooler(x, batch)

            # Predict
            x = self.post_pool1(x)
            x = self.post_pool2(x)
        return x

class SEGNN(L.LightningModule):
    """Steerable E(3) equivariant message passing network"""

    def __init__(
        self,
        input_irreps=None,
        hidden_irreps=None,
        output_irreps=None,
        edge_attr_irreps=None,
        node_attr_irreps=None,
        num_layers=1,
        norm=None,
        pool="avg",
        task="graph",
        additional_message_irreps=None,
    ):
        input_irreps = Irreps(input_irreps) if input_irreps is not None else Irreps("1x0e")
        hidden_irreps = Irreps(hidden_irreps) if hidden_irreps is not None else Irreps("1x0e")
        output_irreps = Irreps(output_irreps) if output_irreps is not None else Irreps("1x0e")
        edge_attr_irreps = Irreps(edge_attr_irreps) if edge_attr_irreps is not None else Irreps("1x0e")
        node_attr_irreps = Irreps(node_attr_irreps) if node_attr_irreps is not None else Irreps("1x0e")

        super().__init__()
        self.task = task
        # Create network, embedding first
        # self.embedding_layer_1 = O3TensorProductSwishGate(
        #     input_irreps, hidden_irreps, node_attr_irreps
        # )
        # self.embedding_layer_2 = O3TensorProduct(
        #     hidden_irreps, hidden_irreps, node_attr_irreps
        # )

        self.embedding_layer = O3TensorProduct(
            input_irreps, hidden_irreps, node_attr_irreps
        )

        # Message passing layers.
        layers = []
        for i in range(num_layers):
            layers.append(
                SEGNNLayer(
                    hidden_irreps,
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                )
            )
        self.layers = nn.ModuleList(layers)

        # Prepare for output irreps, since the attrs will disappear after pooling
        if task == "graph":
            pooled_irreps = (
                (output_irreps * hidden_irreps.num_irreps).simplify().sort().irreps
            )
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, pooled_irreps, node_attr_irreps
            )
            self.post_pool1 = O3TensorProductSwishGate(pooled_irreps, pooled_irreps)
            self.post_pool2 = O3TensorProduct(pooled_irreps, output_irreps)
            self.init_pooler(pool)
        elif task == "node":
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, output_irreps, node_attr_irreps
            )

    def init_pooler(self, pool):
        """Initialise pooling mechanism"""
        if pool == "avg":
            self.pooler = global_mean_pool
        elif pool == "sum":
            self.pooler = global_add_pool

    def catch_isolated_nodes(self, graph):
        """Isolated nodes should also obtain attributes"""
        if (
            graph.contains_isolated_nodes()
            and graph.edge_index.max().item() + 1 != graph.num_nodes
        ):
            nr_add_attr = graph.num_nodes - (graph.edge_index.max().item() + 1)
            add_attr = graph.node_attr.new_tensor(
                np.zeros((nr_add_attr, graph.node_attr.shape[-1]))
            )
            graph.node_attr = torch.cat((graph.node_attr, add_attr), -2)
        # Trivial irrep value should always be 1 (is automatically so for connected nodes, but isolated nodes are now 0)
        graph.node_attr[:, 0] = 1.0

    def forward(self, graph):
        """SEGNN forward pass"""
        x, pos, edge_index, edge_attr, node_attr, batch = (
            graph.x,
            graph.pos,
            graph.edge_index,
            graph.edge_attr,
            graph.node_attr,
            graph.batch,
        )
        try:
            additional_message_features = graph.additional_message_features
        except AttributeError:
            additional_message_features = None

        self.catch_isolated_nodes(graph)

        # Embed
        # x = self.embedding_layer_1(x, node_attr)
        # x = self.embedding_layer_2(x, node_attr)
        x = self.embedding_layer(x, node_attr)

        # Pass messages
        for layer in self.layers:
            x = layer(
                x, edge_index, edge_attr, node_attr, batch, additional_message_features
            )

        # Pre pool
        x = self.pre_pool1(x, node_attr)
        x = self.pre_pool2(x, node_attr)

        if self.task == "graph":
            # Pool over nodes
            x = self.pooler(x, batch)

            # Predict
            x = self.post_pool1(x)
            x = self.post_pool2(x)
        return x



def BalancedIrreps(lmax, vec_dim, sh_type=True):
    """ Allocates irreps equally along channel budget, resulting
        in unequal numbers of irreps in ratios of 2l_i + 1 to 2l_j + 1.

    Parameters
    ----------
    lmax : int
        Maximum order of irreps.
    vec_dim : int
        Dim of feature vector.
    sh_type : bool
        if true, use spherical harmonics. Else the full set of irreps (with redundance).

    Returns
    -------
    Irreps
        Resulting irreps for feature vectors.

    """
    irrep_spec = "0e"
    for l in range(1, lmax + 1):
        if sh_type:
            irrep_spec += " + {0}".format(l) + ('e' if (l % 2) == 0 else 'o')
        else:
            irrep_spec += " + {0}e + {0}o".format(l)
    irrep_spec_split = irrep_spec.split(" + ")
    dims = [int(irrep[0]) * 2 + 1 for irrep in irrep_spec_split]
    # Compute ratios
    ratios = [1 / dim for dim in dims]
    # Determine how many copies per irrep
    irrep_copies = [int(vec_dim * r / len(ratios)) for r in ratios]
    # Determine the current effective irrep sizes
    irrep_dims = [n * dim for (n, dim) in zip(irrep_copies, dims)]
    # Add trivial irreps until the desired size is reached
    irrep_copies[0] += vec_dim - sum(irrep_dims)

    # Convert to string
    str_out = ''
    for (spec, dim) in zip(irrep_spec_split, irrep_copies):
        str_out += str(dim) + 'x' + spec
        str_out += ' + '
    str_out = str_out[:-3]
    # Generate the irrep
    return Irreps(str_out)

def WeightBalancedIrreps(irreps_in1_scalar, irreps_in2, sh=True, lmax=None):
    """Determines an irreps_in1 type of order irreps_in2.lmax that when used in a tensor product
    irreps_in1 x irreps_in2 -> irreps_in1
    would have the same number of weights as for a standard linear layer, e.g. a tensor product
    irreps_in1_scalar x "1x0e" -> irreps_in1_scalar

    Parameters
    ----------
    irreps_in1_scalar : o3.Irreps
        Number of hidden features, represented by zeroth order irreps.
    irreps_in2 : o3.Irreps
        Irreps related to edge attributes.
    sh : bool
        if true, yields equal number of every order. Else returns balanced irrep.
    lmax : int
        Maximum order irreps to be considered.

    Returns
    -------
    o3.Irreps
        Irreps for hidden feaure vectors.

    """

    n = 1
    if lmax == None:
        lmax = irreps_in2.lmax
    irreps_in1 = (Irreps.spherical_harmonics(lmax) * n).sort().irreps.simplify() if sh else BalancedIrreps(lmax, n)
    weight_numel1 = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_in1).weight_numel
    weight_numel_scalar = FullyConnectedTensorProduct(irreps_in1_scalar, Irreps("1x0e"), irreps_in1_scalar).weight_numel
    while weight_numel1 < weight_numel_scalar:  # TODO: somewhat suboptimal implementation...
        n += 1
        irreps_in1 = (Irreps.spherical_harmonics(lmax) * n).sort().irreps.simplify() if sh else BalancedIrreps(lmax, n)
        weight_numel1 = FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_in1).weight_numel
    print('Determined irrep type:', irreps_in1)
    return Irreps(irreps_in1)

class O3TensorProduct(nn.Module):
    """ A bilinear layer, computing CG tensorproduct and normalising them.

    Parameters
    ----------
    irreps_in1 : o3.Irreps
        Input irreps.
    irreps_out : o3.Irreps
        Output irreps.
    irreps_in2 : o3.Irreps
        Second input irreps.
    tp_rescale : bool
        If true, rescales the tensor product.

    """

    def __init__(self, irreps_in1, irreps_out, irreps_in2=None, tp_rescale=True) -> None:
        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_out = irreps_out
        # Init irreps_in2
        if irreps_in2 == None:
            self.irreps_in2_provided = False
            self.irreps_in2 = Irreps("1x0e")
        else:
            self.irreps_in2_provided = True
            self.irreps_in2 = irreps_in2
        self.tp_rescale = tp_rescale

        # Build the layers
        self.tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out, shared_weights=True, normalization='component')

        # For each zeroth order output irrep we need a bias
        # So first determine the order for each output tensor and their dims
        self.irreps_out_orders = [int(irrep_str[-2]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_slices = irreps_out.slices()
        # Store tuples of slices and corresponding biases in a list
        self.biases = []
        self.biases_slices = []
        self.biases_slice_idx = []
        for slice_idx in range(len(self.irreps_out_orders)):
            if self.irreps_out_orders[slice_idx] == 0:
                out_slice = irreps_out.slices()[slice_idx]
                out_bias = torch.zeros(self.irreps_out_dims[slice_idx], dtype=self.tp.weight.dtype)
                self.biases += [out_bias]
                self.biases_slices += [out_slice]
                self.biases_slice_idx += [slice_idx]

        # Initialize the correction factors
        self.slices_sqrt_k = {}

        # Initialize similar to the torch.nn.Linear
        self.tensor_product_init()
        # Adapt parameters so they can be applied using vector operations.
        self.vectorise()

    def tensor_product_init(self) -> None:
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                slice_idx = instr[2]
                mul_1, mul_2, mul_out = weight.shape
                fan_in = mul_1 * mul_2
                slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] +
                                            fan_in if slice_idx in slices_fan_in.keys() else fan_in)
            # Do the initialization of the weights in each instruction
            for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                slice_idx = instr[2]
                if self.tp_rescale:
                    sqrt_k = 1 / sqrt(slices_fan_in[slice_idx])
                else:
                    sqrt_k = 1.
                weight.data.uniform_(-sqrt_k, sqrt_k)
                self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            for (out_slice_idx, out_slice, out_bias) in zip(self.biases_slice_idx, self.biases_slices, self.biases):
                sqrt_k = 1 / sqrt(slices_fan_in[out_slice_idx])
                out_bias.uniform_(-sqrt_k, sqrt_k)

    def vectorise(self):
        """ Adapts the bias parameter and the sqrt_k corrections so they can be applied using vectorised operations """

        # Vectorise the bias parameters
        if len(self.biases) > 0:
            with torch.no_grad():
                self.biases = torch.cat(self.biases, dim=0)
            self.biases = nn.Parameter(self.biases)

            # Compute broadcast indices.
            bias_idx = torch.LongTensor()
            for slice_idx in range(len(self.irreps_out_orders)):
                if self.irreps_out_orders[slice_idx] == 0:
                    out_slice = self.irreps_out.slices()[slice_idx]
                    bias_idx = torch.cat((bias_idx, torch.arange(out_slice.start, out_slice.stop).long()), dim=0)

            self.register_buffer("bias_idx", bias_idx, persistent=False)
        else:
            self.biases = None

        # Now onto the sqrt_k correction
        sqrt_k_correction = torch.zeros(self.irreps_out.dim)
        for instr in self.tp.instructions:
            slice_idx = instr[2]
            slice, sqrt_k = self.slices_sqrt_k[slice_idx]
            sqrt_k_correction[slice] = sqrt_k

        # Make sure bias_idx and sqrt_k_correction are on same device as module
        self.register_buffer("sqrt_k_correction", sqrt_k_correction, persistent=False)

    def forward_tp_rescale_bias(self, data_in1, data_in2=None) -> torch.Tensor:
        if data_in2 == None:
            data_in2 = torch.ones_like(data_in1[:, 0:1])

        data_out = self.tp(data_in1, data_in2)

        # Apply corrections
        if self.tp_rescale:
            data_out /= self.sqrt_k_correction

        # Add the biases
        if self.biases is not None:
            data_out[:, self.bias_idx] += self.biases
        return data_out

    def forward(self, data_in1, data_in2=None) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2)
        return data_out

class O3TensorProductSwishGate(O3TensorProduct):
    def __init__(self, irreps_in1, irreps_out, irreps_in2=None) -> None:
        # For the gate the output of the linear needs to have an extra number of scalar irreps equal to the amount of
        # non scalar irreps:
        # The first type is assumed to be scalar and passed through the activation
        irreps_g_scalars = Irreps(str(irreps_out[0]))
        # The remaining types are gated
        irreps_g_gate = Irreps("{}x0e".format(irreps_out.num_irreps - irreps_g_scalars.num_irreps))
        irreps_g_gated = Irreps(str(irreps_out[1:]))
        # So the gate needs the following irrep as input, this is the output irrep of the tensor product
        irreps_g = (irreps_g_scalars + irreps_g_gate + irreps_g_gated).simplify()

        # Build the layers
        super(O3TensorProductSwishGate, self).__init__(irreps_in1, irreps_g, irreps_in2)
        if irreps_g_gated.num_irreps > 0:
            self.gate = Gate(irreps_g_scalars, [nn.SiLU()], irreps_g_gate, [torch.sigmoid], irreps_g_gated)
        else:
            self.gate = nn.SiLU()

    def forward(self, data_in1, data_in2=None) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2)
        # Apply the gate
        data_out = self.gate(data_out)
        # Return result
        return data_out

class O3SwishGate(torch.nn.Module):
    def __init__(self, irreps_g_scalars, irreps_g_gate, irreps_g_gated) -> None:
        super().__init__()
        if irreps_g_gated.num_irreps > 0:
            self.gate = Gate(irreps_g_scalars, [nn.SiLU()], irreps_g_gate, [torch.sigmoid], irreps_g_gated)
        else:
            self.gate = nn.SiLU()

    def forward(self, data_in) -> torch.Tensor:
        data_out = self.gate(data_in)
        return data_out

class InstanceNorm(nn.Module):
    '''Instance normalization for orthonormal representations
    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.
    Parameters
    ----------
    irreps : `Irreps`
        representation
    eps : float
        avoid division by zero when we normalize by the variance
    affine : bool
        do we have weight and bias parameters
    reduce : {'mean', 'max'}
        method used to reduce
    '''

    def __init__(self, irreps, eps=1e-5, affine=True, reduce='mean', normalization='component'):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0)
        num_features = self.irreps.num_irreps

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ['mean', 'max'], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        assert normalization in ['norm', 'component'], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, input, batch):
        '''evaluate
        Parameters
        ----------
        input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        '''
        # batch, *size, dim = input.shape  # TODO: deal with batch
        # input = input.reshape(batch, -1, dim)  # [batch, sample, stacked features]
        # input has shape [batch * nodes, dim], but with variable nr of nodes.
        # the input batch slices this into separate graphs
        dim = input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            field = input[:, ix: ix + mul * d]  # [batch * sample, mul * repr]
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)

            # For scalars first compute and subtract the mean
            if ir.l == 0:
                # Compute the mean
                field_mean = global_mean_pool(field, batch).reshape(-1, mul, 1)  # [batch, mul, 1]]
                # Subtract the mean
                field = field - field_mean[batch]

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == 'component':
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError("Invalid normalization option {}".format(self.normalization))
            # Reduction method
            if self.reduce == 'mean':
                field_norm = global_mean_pool(field_norm, batch)  # [batch, mul]
            elif self.reduce == 'max':
                field_norm = global_max_pool(field_norm, batch)  # [batch, mul]
            else:
                raise ValueError("Invalid reduce option {}".format(self.reduce))

            # Then apply the rescaling (divide by the sqrt of the squared_norm, i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.weight[None, iw: iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm[batch].reshape(-1, mul, 1)  # [batch * sample, mul, repr]

            if self.affine and d == 1:  # scalars
                bias = self.bias[ib: ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output

class SEGNNLayer(MessagePassing):
    """E(3) equivariant message passing layer."""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        norm=None,
        additional_message_irreps=None,
    ):
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = hidden_irreps

        message_input_irreps = (2 * input_irreps + additional_message_irreps).simplify()
        update_input_irreps = (input_irreps + hidden_irreps).simplify()

        self.message_layer_1 = O3TensorProductSwishGate(
            message_input_irreps, hidden_irreps, edge_attr_irreps
        )
        self.message_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, edge_attr_irreps
        )
        self.update_layer_1 = O3TensorProductSwishGate(
            update_input_irreps, hidden_irreps, node_attr_irreps
        )
        self.update_layer_2 = O3TensorProduct(
            hidden_irreps, hidden_irreps, node_attr_irreps
        )

        self.setup_normalisation(norm)

    def setup_normalisation(self, norm):
        """Set up normalisation, either batch or instance norm"""
        self.norm = norm
        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(self.hidden_irreps)
            self.message_norm = BatchNorm(self.hidden_irreps)
        elif norm == "instance":
            self.feature_norm = InstanceNorm(self.hidden_irreps)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        node_attr,
        batch,
        additional_message_features=None,
    ):
        """Propagate messages along edges"""
        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
        )
        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i, x_j, edge_attr, additional_message_features):
        """Create messages"""
        if additional_message_features is None:
            input = torch.cat((x_i, x_j), dim=-1)
        else:
            input = torch.cat((x_i, x_j, additional_message_features), dim=-1)

        message = self.message_layer_1(input, edge_attr)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x, node_attr):
        """Update note features"""
        input = torch.cat((x, message), dim=-1)
        update = self.update_layer_1(input, node_attr)
        update = self.update_layer_2(update, node_attr)
        x += update  # Residual connection
        return x

class SEConvLayer(MessagePassing):
    """E(3) equivariant (non-linear) convolutional layer."""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        norm=None,
        additional_message_irreps=None,
        conv_type="linear",
    ):
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = hidden_irreps
        self.conv_type = conv_type

        message_input_irreps = (2 * input_irreps + additional_message_irreps).simplify()
        self.setup_gate(hidden_irreps)
        if self.conv_type == "linear":
            self.message_layer = O3TensorProduct(
                message_input_irreps, self.irreps_g, edge_attr_irreps
            )
        elif self.conv_type == "nonlinear":
            self.message_layer_1 = O3TensorProductSwishGate(
                message_input_irreps, hidden_irreps, edge_attr_irreps
            )
            self.message_layer_2 = O3TensorProduct(
                hidden_irreps, self.irreps_g, edge_attr_irreps
            )
        else:
            raise Exception("Invalid convolution type for SEConvLayer")

        self.setup_normalisation(norm)

    def setup_gate(self, hidden_irreps):
        """Add necessary scalar irreps for gate to output_irreps, similar to O3TensorProductSwishGate"""
        irreps_g_scalars = Irreps(str(hidden_irreps[0]))
        irreps_g_gate = Irreps(
            "{}x0e".format(hidden_irreps.num_irreps - irreps_g_scalars.num_irreps)
        )
        irreps_g_gated = Irreps(str(hidden_irreps[1:]))
        irreps_g = (irreps_g_scalars + irreps_g_gate + irreps_g_gated).simplify()
        self.gate = O3SwishGate(irreps_g_scalars, irreps_g_gate, irreps_g_gated)
        self.irreps_g = irreps_g

    def setup_normalisation(self, norm):
        """Set up normalisation, either batch or instance norm"""
        self.norm = norm
        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(self.hidden_irreps)
            self.message_norm = BatchNorm(self.irreps_g)
        elif norm == "instance":
            self.feature_norm = InstanceNorm(self.hidden_irreps)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        node_attr,
        batch,
        additional_message_features=None,
    ):
        """Propagate messages along edges"""
        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
        )
        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i, x_j, edge_attr, additional_message_features):
        """Create messages"""
        if additional_message_features is None:
            input = torch.cat((x_i, x_j), dim=-1)
        else:
            input = torch.cat((x_i, x_j, additional_message_features), dim=-1)

        if self.conv_type == "linear":
            message = self.message_layer(input, edge_attr)
        elif self.conv_type == "nonlinear":
            message = self.message_layer_1(input, edge_attr)
            message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x, node_attr):
        """Update note features"""
        update = self.gate(message)
        x += update  # Residual connection
        return x
