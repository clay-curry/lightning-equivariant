###########################################################################################
# Cormorant
# Paper: Cormorant: Covariant Molecular Neural Networks

# Authors: Brandon Anderson, Truong-Son Hy, Risi Kondor
# Date: 06 Jun 2019
# Comments: 33rd Conference on Neural Information Processing Systems (NeurIPS 2019)
# Repo: https://github.com/risilab/cormorant (MIT License)
# Paper: https://arxiv.org/abs/1906.04015
# This program is distributed under an Educational and Not-for-Profit Research License
###########################################################################################

import torch
import logging
import warnings
import numpy as np
from numpy import pi
from torch import nn, Module
from functools import reduce
from math import sqrt, pi, inf
from itertools import zip_longest
from abc import ABC, abstractmethod
from scipy.special import factorial
logger = logging.getLogger(__name__)

class CormorantModel():
    pass # TODO

SO3Tau = so3_tau.SO3Tau
SO3Tensor = so3_tensor.SO3Tensor

class CGModule(nn.Module):
    """
    Clebsch-Gordan module. This functions identically to a normal PyTorch
    nn.Module, except for it adds the ability to specify a
    Clebsch-Gordan dictionary, and has additional tracking behavior set up
    to allow the CG Dictionary to be compatible with the DataParallel module.

    If `cg_dict` is specified upon instantiation, then the specified
    `cg_dict` is set as the Clebsch-Gordan dictionary for the CG module.

    If `cg_dict` is not specified, and `maxl` is specified, then CGModule
    will attempt to set the local `cg_dict` based upon the global
    `cormorant.cg_lib.global_cg_dicts`. If the dictionary has not been initialized
    with the appropriate `dtype`, `device`, and `maxl`, it will be initialized
    and stored in the `global_cg_dicts`, and then set to the local `cg_dict`.

    In this way, if there are many modules that need `CGDicts`, only a single
    `CGDict` will be initialized and automatically set up.

    Parameters
    ----------
    cg_dict : :class:`CGDict`, optional
        Specify an input CGDict to use for Clebsch-Gordan operations.
    maxl : :class:`int`, optional
        Maximum weight to initialize the Clebsch-Gordan dictionary.
    device : :class:`torch.torch.device`, optional
        Device to initialize the module and Clebsch-Gordan dictionary to.
    dtype : :class:`torch.torch.dtype`, optional
        Data type to initialize the module and Clebsch-Gordan dictionary to.
    """
    def __init__(self, cg_dict=None, maxl=None, device=None, dtype=None, *args, **kwargs):
        self._init_device_dtype(device, dtype)
        self._init_cg_dict(cg_dict, maxl)

        super().__init__(*args, **kwargs)

    def _init_device_dtype(self, device, dtype):
        """
        Initialize the default device and data type.

        device : :class:`torch.torch.device`, optional
            Set device for CGDict and related. If unset defaults to torch.device('cpu').

        dtype : :class:`torch.torch.dtype`, optional
            Set device for CGDict and related. If unset defaults to torch.float.

        """
        if device is None:
            self._device = torch.device('cpu')
        else:
            self._device = device

        if dtype is None:
            self._dtype = torch.float
        else:
            if not (dtype == torch.half or dtype == torch.float or dtype == torch.double):
                raise ValueError('CG Module only takes internal data types of half/float/double. Got: {}'.format(dtype))
            self._dtype = dtype

    def _init_cg_dict(self, cg_dict, maxl):
        """
        Initialize the Clebsch-Gordan dictionary.

        If cg_dict is set, check the following::
        - The dtype of cg_dict matches with self.
        - The devices of cg_dict matches with self.
        - The desired :maxl: <= :cg_dict.maxl: so that the CGDict will contain
            all necessary coefficients

        If :cg_dict: is not set, but :maxl: is set, get the cg_dict from a
        dict of global CGDict() objects.
        """
        # If cg_dict is defined, check it has the right properties
        if cg_dict is not None:
            if cg_dict.dtype != self.dtype:
                raise ValueError('CGDict dtype ({}) not match CGModule() dtype ({})'.format(cg_dict.dtype, self.dtype))

            if cg_dict.device != self.device:
                raise ValueError('CGDict device ({}) not match CGModule() device ({})'.format(cg_dict.device, self.device))

            if maxl is None:
                Warning('maxl is not defined, setting maxl based upon CGDict maxl ({}!'.format(cg_dict.maxl))

            elif maxl > cg_dict.maxl:
                Warning('CGDict maxl ({}) is smaller than CGModule() maxl ({}). Updating!'.format(cg_dict.maxl, maxl))
                cg_dict.update_maxl(maxl)

            self.cg_dict = cg_dict
            self._maxl = maxl

        # If cg_dict is not defined, but
        elif cg_dict is None and maxl is not None:

            self.cg_dict = CGDict(maxl=maxl, device=self.device, dtype=self.dtype)
            self._maxl = maxl

        else:
            self.cg_dict = None
            self._maxl = None

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def maxl(self):
        return self._maxl

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        if self.cg_dict is not None:
            self.cg_dict.to(device=device, dtype=dtype)

        if device is not None:
            self._device = device

        if dtype is not None:
            self._dtype = dtype

        return self

    def cuda(self, device=None):
        if device is None:
            device = torch.device('cuda')
        elif device in range(torch.cuda.device_count()):
            device = torch.device('cuda:{}'.format(device))
        else:
            ValueError('Incorrect choice of device!')

        super().cuda(device=device)

        if self.cg_dict is not None:
            self.cg_dict.to(device=device)

        self._device = device

        return self

    def cpu(self):
        super().cpu()

        if self.cg_dict is not None:
            self.cg_dict.to(device=torch.device('cpu'))

        self._device = torch.device('cpu')

        return self

    def half(self):
        super().half()

        if self.cg_dict is not None:
            self.cg_dict.to(dtype=torch.half)

        self._dtype = torch.half

        return self

    def float(self):
        super().float()

        if self.cg_dict is not None:
            self.cg_dict.to(dtype=torch.float)

        self._dtype = torch.float

        return self

    def double(self):
        super().double()

        if self.cg_dict is not None:
            self.cg_dict.to(dtype=torch.double)

        self._dtype = torch.double

        return self

class SO3WignerD(SO3Tensor):
    """
    Core class for creating and tracking WignerD matrices.

    At the core of each :obj:`SO3WignerD` is a list of :obj:`torch.Tensors` with
    shape `(2*l+1, 2*l+1, 2)`, where:

    * `2*l+1` is the size of an irrep of weight `l`.
    * `2` corresponds to the real/imaginary parts of the complex dimension.

    Note
    ----

    For now, there is no batch or channel dimensions included. Although a
    SO3 covariant network architecture with Wigner-D matrices is possible,
    the current scheme using PyTorch built-ins would be too slow to implement.
    A custom CUDA kernel would likely be necessary, and is a work in progress.

    Warning
    -------
    The constructor __init__() does not check that the tensor is actually
    a Wigner-D matrix, (that is an irreducible representation of the group SO3)
    so it is important to ensure that the input tensor is generated appropraitely.

    Parameters
    ----------

    data : iterable of of `torch.Tensor` with appropriate shape
        Input of a SO(3) vector.
    """

    @property
    def bdim(self):
        return None

    @property
    def cdim(self):
        return None

    @property
    def rdim1(self):
        return 0

    @property
    def rdim2(self):
        return 1

    rdim = rdim2

    @property
    def zdim(self):
        return 2

    @property
    def ells(self):
        return [(shape[self.rdim] - 1)//2 for shape in self.shapes]

    @staticmethod
    def _get_shape(batch, l, channels):
        return (2*l+1, 2*l+1, 2)

    def check_data(self, data):
        if any(part.numel() == 0 for part in data):
            raise NotImplementedError('Non-zero parts in SO3WignerD not currrently enabled!')

        shapes = [part.shape for part in data]

        rdims1 = [shape[self.rdim1] for shape in shapes]
        rdims2 = [shape[self.rdim2] for shape in shapes]
        zdims = [shape[self.zdim] for shape in shapes]

        if not all([rdim1 == 2*l+1 and rdim2 == 2*l+1 for l, (rdim1, rdim2) in enumerate(zip(rdims1, rdims2))]):
            raise ValueError('Irrep dimension (dim={}) of each tensor should have shape 2*l+1! Found: {}'.format(self.rdim, list(enumerate(rdims))))

        if not all([zdim == 2 for zdim in zdims]):
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, zdims))

    @staticmethod
    def _bin_op_type_check(type1, type2):
        if type1 == SO3WignerD and type2 == SO3WignerD:
            raise ValueError('Cannot multiply two SO3WignerD!')

    @staticmethod
    def euler(maxl, angles=None, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new :obj:`SO3Weight`.

        If `angles=None`, will generate a uniformly distributed random Euler
        angle and then instantiate a SO3WignerD accordingly.
        """

        if angles == None:
            alpha, beta, gamma = torch.rand(3) * 2 * pi
            beta = beta / 2

        wigner_d = rot.WignerD_list(maxl, alpha, beta, gamma, device=device, dtype=dtype)

        return SO3WignerD(wigner_d)

    @staticmethod
    def rand(maxl, device=None, dtype=None, requires_grad=False):
        """ Overwrite factor method inherited from :obj:`SO3Tensor` since
        it would break covariance """
        raise NotImplementedError('Does not make sense as it would break covariance!')


    @staticmethod
    def randn(maxl, device=None, dtype=None, requires_grad=False):
        """ Overwrite factor method inherited from :obj:`SO3Tensor` since
        it would break covariance """
        raise NotImplementedError('Does not make sense as it would break covariance!')

    @staticmethod
    def zeros(maxl, device=None, dtype=None, requires_grad=False):
        """ Overwrite factor method inherited from :obj:`SO3Tensor` since
        it would break covariance """
        raise NotImplementedError('Does not make sense as it would break covariance!')

    @staticmethod
    def ones(maxl, device=None, dtype=None, requires_grad=False):
        """ Overwrite factor method inherited from :obj:`SO3Tensor` since
        it would break covariance """
        raise NotImplementedError('Does not make sense as it would break covariance!')

class DotMatrix(CGModule):
    r"""
    Constructs a matrix of dot-products between scalars of the same representation type, as used in the edge levels.

    """
    def __init__(self, tau_in=None, cat=True, device=None, dtype=None):
        super().__init__(device=device, dtype=dtype)
        self.tau_in = tau_in
        self.cat = cat

        if self.tau_in is not None:
            if cat:
                self.tau = SO3Tau([sum(tau_in)] * len(tau_in))
            else:
                self.tau = SO3Tau([t for t in tau_in])
            self.signs = [torch.tensor(-1.).pow(torch.arange(-ell, ell+1).float()).to(device=self.device, dtype=self.dtype).unsqueeze(-1) for ell in range(len(tau_in)+1)]
            self.conj = torch.tensor([1., -1.]).to(device=self.device, dtype=self.dtype)
        else:
            self.tau = None
            self.signs = None

    def forward(self, reps):
        """
        Performs the forward pass.

        Parameters
        ----------
        reps : :class:`SO3Vec <cormorant.so3_lib.SO3Vec>`
            Input SO3 Vector. 
        
        Returns
        -------
        dot_products : :class:`SO3Scalar <cormorant.so3_lib.SO3Scalar>`
            SO3 scalars representing a Matrix of form :math:`(\psi_i \cdot \psi_j)_c`, where c is a channel index with :math:`|C| = \sum_l \tau_l`.
        """
        if self.tau_in is not None and self.tau_in != reps.tau:
            raise ValueError('Initialized tau not consistent with tau from forward! {} {}'.format(self.tau_in, reps.tau))

        signs = self.signs
        conj = self.conj

        reps1 = [part.unsqueeze(-4) for part in reps]
        reps2 = [part.unsqueeze(-5) for part in reps]

        reps2 = [part.flip(-2)*sign for part, sign in zip(reps2, signs)]

        dot_product_r = [(part1*part2*conj).sum(dim=(-2, -1)) for part1, part2 in zip(reps1, reps2)]
        dot_product_i = [(part1*part2.flip(-1)).sum(dim=(-2, -1)) for part1, part2 in zip(reps1, reps2)]

        dot_products = [torch.stack([prod_r, prod_i], dim=-1) for prod_r, prod_i in zip(dot_product_r, dot_product_i)]

        if self.cat:
            dot_products = torch.cat(dot_products, dim=-2)
            dot_products = [dot_products] * len(reps)

        return SO3Scalar(dot_products)

class BasicMLP(nn.Module):
    """
    Multilayer perceptron used in various locations.  Operates only on the last axis of the data.

    Parameters
    ----------
    num_in : int
        Number of input channels
    num_out : int
        Number of output channels
    num_hidden : int, optional
        Number of hidden layers.
    layer_width : int, optional
        Width of each hidden layer (number of channels).
    activation : string, optional
        Type of nonlinearity to use.
    device : :obj:`torch.device`, optional
        Device to initialize the level to
    dtype : :obj:`torch.dtype`, optional
        Data type to initialize the level to
    """

    def __init__(self, num_in, num_out, num_hidden=1, layer_width=256, activation='leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(BasicMLP, self).__init__()

        self.num_in = num_in

        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(num_in, layer_width))
        for i in range(num_hidden-1):
            self.linear.append(nn.Linear(layer_width, layer_width))
        self.linear.append(nn.Linear(layer_width, num_out))

        activation_fn = get_activation_fn(activation)

        self.activations = nn.ModuleList()
        for i in range(num_hidden):
            self.activations.append(activation_fn)

        self.zero = torch.tensor(0, device=device, dtype=dtype)

        self.to(device=device, dtype=dtype)

    def forward(self, x, mask=None):
        # Standard MLP. Loop over a linear layer followed by a non-linear activation
        for (lin, activation) in zip(self.linear, self.activations):
            x = activation(lin(x))

        # After last non-linearity, apply a final linear mixing layer
        x = self.linear[-1](x)

        # If mask is included, mask the output
        if mask is not None:
            x = torch.where(mask, x, self.zero)

        return x

    def scale_weights(self, scale):
        self.linear[-1].weight *= scale
        if self.linear[-1].bias is not None:
            self.linear[-1].bias *= scale

def get_activation_fn(activation):
    activation = activation.lower()
    if activation == 'leakyrelu':
        activation_fn = nn.LeakyReLU()
    elif activation == 'relu':
        activation_fn = nn.ReLU()
    elif activation == 'elu':
        activation_fn = nn.ELU()
    elif activation == 'sigmoid':
        activation_fn = nn.Sigmoid()
    else:
        raise ValueError('Activation function {} not implemented!'.format(activation))
    return activation_fn


############# Input to network #############

class InputLinear(nn.Module):
    """
    Module to create rotationally invariant atom feature vectors
    at the input level.

    This module applies a simple linear mixing matrix to a one-hot of atom
    embeddings based upon the number of atomic types.

    Parameters
    ----------
    channels_in : :class:`int`
        Number of input features before mixing (i.e., a one-hot atom-type embedding).
    channels_out : :class:`int`
        Number of output features after mixing.
    bias : :class:`bool`, optional
        Include a bias term in the linear mixing level.
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, channels_in, channels_out, bias=True,
                 device=torch.device('cpu'), dtype=torch.float):
        super(InputLinear, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out

        self.lin = nn.Linear(channels_in, 2*channels_out, bias=bias)
        self.lin.to(device=device, dtype=dtype)

        self.zero = torch.tensor(0, dtype=dtype, device=device)

    def forward(self, atom_features, atom_mask, ignore, edge_mask, norms):
        """
        Forward pass for :class:`InputLinear` layer.

        Parameters
        ----------
        atom_features : :class:`torch.Tensor`
            Input atom features, i.e., a one-hot embedding of the atom type,
            atom charge, and any other related inputs.
        atom_mask : :class:`torch.Tensor`
            Mask used to account for padded atoms for unequal batch sizes.
        edge_features : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.
        edge_mask : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.
        norms : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.

        Returns
        -------
        :class:`SO3Vec`
            Processed atom features to be used as input to Clebsch-Gordan layers
            as part of Cormorant.
        """
        atom_mask = atom_mask.unsqueeze(-1)

        out = torch.where(atom_mask, self.lin(atom_features), self.zero)
        out = out.view(atom_features.shape[0:2] + (self.channels_out, 1, 2))

        return SO3Vec([out])

    @property
    def tau(self):
        return SO3Tau([self.channels_out])

class InputMPNN(nn.Module):
    """
    Module to create rotationally invariant atom feature vectors
    at the input level.

    This module applies creates a scalar

    Parameters
    ----------
    channels_in : :class:`int`
        Number of input features before mixing (i.e., a one-hot atom-type embedding).
    channels_out : :class:`int`
        Number of output features after mixing.
    num_layers : :class:`int`
        Number of message passing layers.
    soft_cut_rad : :class:`float`
        Radius of the soft cutoff used in the radial position functions.
    soft_cut_width : :class:`float`
        Radius of the soft cutoff used in the radial position functions.
    hard_cut_rad : :class:`float`
        Radius of the soft cutoff used in the radial position functions.
    bias : :class:`bool`, optional
        Include a bias term in the linear mixing level.
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, channels_in, channels_out, num_layers=1,
                 soft_cut_rad=None, soft_cut_width=None, hard_cut_rad=None, cutoff_type=['learn'],
                 channels_mlp=-1, num_hidden=1, layer_width=256,
                 activation='leakyrelu', basis_set=(3, 3),
                 device=torch.device('cpu'), dtype=torch.float):
        super(InputMPNN, self).__init__()

        self.soft_cut_rad = soft_cut_rad
        self.soft_cut_width = soft_cut_width
        self.hard_cut_rad = hard_cut_rad

        if channels_mlp < 0:
            channels_mlp = max(channels_in, channels_out)

        # List of channels at each level. The factor of two accounts for
        # the fact that the passed messages are concatenated with the input states.
        channels_lvls = [channels_in] + [channels_mlp]*(num_layers-1) + [2*channels_out]

        self.channels_in = channels_in
        self.channels_mlp = channels_mlp
        self.channels_out = channels_out

        # Set up MLPs
        self.mlps = nn.ModuleList()
        self.masks = nn.ModuleList()
        self.rad_filts = nn.ModuleList()

        for chan_in, chan_out in zip(channels_lvls[:-1], channels_lvls[1:]):
            rad_filt = RadPolyTrig(0, basis_set, chan_in, mix='real', device=device, dtype=dtype)
            mask = MaskLevel(1, hard_cut_rad, soft_cut_rad, soft_cut_width, ['soft', 'hard'], device=device, dtype=dtype)
            mlp = BasicMLP(2*chan_in, chan_out, num_hidden=num_hidden, layer_width=layer_width, device=device, dtype=dtype)

            self.mlps.append(mlp)
            self.masks.append(mask)
            self.rad_filts.append(rad_filt)

        self.dtype = dtype
        self.device = device

    def forward(self, features, atom_mask, edge_features, edge_mask, norms):
        """
        Forward pass for :class:`InputMPNN` layer.

        Parameters
        ----------
        features : :class:`torch.Tensor`
            Input atom features, i.e., a one-hot embedding of the atom type,
            atom charge, and any other related inputs.
        atom_mask : :class:`torch.Tensor`
            Mask used to account for padded atoms for unequal batch sizes.
        edge_features : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.
        edge_mask : :class:`torch.Tensor`
            Mask used to account for padded edges for unequal batch sizes.
        norms : :class:`torch.Tensor`
            Matrix of relative distances between pairs of atoms.

        Returns
        -------
        :class:`SO3Vec`
            Processed atom features to be used as input to Clebsch-Gordan layers
            as part of Cormorant.
        """
        # Unsqueeze the atom mask to match the appropriate dimensions later
        atom_mask = atom_mask.unsqueeze(-1)

        # Get the shape of the input to reshape at the end
        s = features.shape

        # Loop over MPNN levels. There is no "edge network" here.
        # Instead, there is just masked radial functions, that take
        # the role of the adjacency matrix.
        for mlp, rad_filt, mask in zip(self.mlps, self.rad_filts, self.masks):
            # Construct the learnable radial functions
            rad = rad_filt(norms, edge_mask)

            # TODO: Real-valued SO3Scalar so we don't need any hacks
            # Convert to a form that MaskLevel expects
            # Hack to account for the lack of real-valued SO3Scalar and
            # structure of RadialFilters.
            rad = rad[0][..., 0].unsqueeze(-1)

            # OLD:
            # Convert to a form that MaskLevel expects
            # rad[0] = rad[0].unsqueeze(-1)

            # Mask the position function if desired
            edge = mask(rad, edge_mask, norms)
            # Convert to a form that MatMul expects
            edge = edge.squeeze(-1)

            # Now pass messages using matrix multiplication with the edge features
            # Einsum b: batch, a: atom, c: channel, x: to be summed over
            features_mp = torch.einsum('baxc,bxc->bac', edge, features)

            # Concatenate the passed messages with the original features
            features_mp = torch.cat([features_mp, features], dim=-1)

            # Now apply a masked MLP
            features = mlp(features_mp, mask=atom_mask)

        # The output are the MLP features reshaped into a set of complex numbers.
        out = features.view(s[0:2] + (self.channels_out, 1, 2))

        return SO3Vec([out])

    @property
    def tau(self):
        return SO3Tau([self.channels_out])

class MaskLevel(nn.Module):
    """
    Mask level for implementing hard and soft cutoffs. With the current
    architecutre, we have all-to-all communication.

    This mask takes relative position vectors :math:`r_{ij} = r_i - r_j`
    and implements either a hard cutoff, a soft cutoff, or both. The soft
    cutoffs can also be made learnable.

    Parameters
    ----------
    num_channels : :class:`int`
        Number of channels to mask out.
    hard_cut_rad : :class:`float`
        Hard cutoff radius. Beyond this radius two atoms will never communicate.
    soft_cut_rad : :class:`float`
        Soft cutoff radius used in cutoff function.
    soft_cut_width : :class:`float`
        Soft cutoff width if ``sigmoid`` form of cutoff cuntion is used.
    cutoff_type : :class:`list` of :class:`str`
        Specify what types of cutoffs to use: `hard`, `soft`, `learn`-albe soft cutoff.
    gaussian_mask : :class:`bool`
        Mask using gaussians instead of sigmoids.
    eps : :class:`float`
        Numerical minimum to use in case learnable cutoff paramaeters are driven towards zero.
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, num_channels, hard_cut_rad, soft_cut_rad, soft_cut_width, cutoff_type,
                 gaussian_mask=False, eps=1e-3, device=torch.device('cpu'), dtype=torch.float):
        super(MaskLevel, self).__init__()

        self.gaussian_mask = gaussian_mask

        self.num_channels = num_channels

        # Initialize hard/soft cutoff to None as default.
        self.hard_cut_rad = None
        self.soft_cut_rad = None
        self.soft_cut_width = None

        if 'hard' in cutoff_type:
            self.hard_cut_rad = hard_cut_rad

        if ('soft' in cutoff_type) or ('learn' in cutoff_type) or ('learn_rad' in cutoff_type) or ('learn_width' in cutoff_type):

            self.soft_cut_rad = soft_cut_rad*torch.ones(num_channels, device=device, dtype=dtype).view((1, 1, 1, -1))
            self.soft_cut_width = soft_cut_width*torch.ones(num_channels, device=device, dtype=dtype).view((1, 1, 1, -1))

            if ('learn' in cutoff_type) or ('learn_rad' in cutoff_type):
                self.soft_cut_rad = nn.Parameter(self.soft_cut_rad)

            if ('learn' in cutoff_type) or ('learn_width' in cutoff_type):
                self.soft_cut_width = nn.Parameter(self.soft_cut_width)

        # Standard bookkeeping
        self.dtype = dtype
        self.device = device

        self.zero = torch.tensor(0, device=device, dtype=dtype)
        self.eps = torch.tensor(eps, device=device, dtype=dtype)

    def forward(self, edge_net, edge_mask, norms):
        """
        Forward pass for :class:`MaskLevel`

        Parameters
        ----------
        edge_net : :class:`torch.Tensor`
            Edge scalars or edge `SO3Vec` to apply mask to.
        edge_mask : :class:`torch.Tensor`
            Mask to account for padded batches.
        norms : :class:`torch.Tensor`
            Pairwise distance matrices.

        Returns
        -------
        edge_net : :class:`torch.Tensor`
            Input ``edge_net`` with mask applied.
        """
        if self.hard_cut_rad is not None:
            edge_mask = (edge_mask * (norms < self.hard_cut_rad))

        edge_mask = edge_mask.to(self.dtype).unsqueeze(-1).to(self.dtype)

        if self.soft_cut_rad is not None:
            cut_width = torch.max(self.eps, self.soft_cut_width.abs())
            cut_rad = torch.max(self.eps, self.soft_cut_rad.abs())

            if self.gaussian_mask:
                edge_mask = edge_mask * torch.exp(-(norms.unsqueeze(-1)/cut_rad).pow(2))
            else:
                edge_mask = edge_mask * torch.sigmoid((cut_rad - norms.unsqueeze(-1))/cut_width)

        edge_mask = edge_mask.unsqueeze(-1)

        edge_net = edge_net * edge_mask

        return edge_net

############# Get Scalars #############

class GetScalarsAtom(nn.Module):
    """
    Construct a set of scalar feature vectors for each atom by using the
    covariant atom :class:`SO3Vec` representations at various levels.

    Parameters
    ----------
    tau_levels : :class:`list` of :class:`SO3Tau`
        Multiplicities of the output :class:`SO3Vec` at each level.
    full_scalars : :class:`bool`, optional
        Construct a more complete set of scalar invariants from the full
        :class:`SO3Vec` (``true``), or just use the :math:``\ell=0`` component
        (``false``).
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, tau_levels, full_scalars=True, device=torch.device('cpu'), dtype=torch.float):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.maxl = max([len(tau) for tau in tau_levels]) - 1

        signs_tr = [torch.pow(-1, torch.arange(-m, m+1.)) for m in range(self.maxl+1)]
        signs_tr = [torch.stack([s, -s], dim=-1) for s in signs_tr]
        self.signs_tr = [s.view(1, 1, 1, -1, 2).to(device=device, dtype=dtype) for s in signs_tr]

        split_l0 = [tau[0] for tau in tau_levels]
        split_full = [sum(tau) for tau in tau_levels]

        self.full_scalars = full_scalars
        if full_scalars:
            self.num_scalars = sum(split_l0) + sum(split_full)
            self.split = split_l0 + split_full
        else:
            self.num_scalars = sum(split_l0)
            self.split = split_l0

        print('Number of scalars at top:', self.num_scalars)

    def forward(self, reps_all_levels):
        """
        Forward step for :class:`GetScalarsAtom`

        Parameters
        ----------
        reps_all_levels : :class:`list` of :class:`SO3Vec`
            List of covariant atom features at each level

        Returns
        -------
        scalars : :class:`torch.Tensor`
            Invariant scalar atom features constructed from ``reps_all_levels``
        """

        reps = cat(reps_all_levels)

        scalars = reps[0]

        if self.full_scalars:
            scalars_tr  = [(sign*part*part.flip(-2)).sum(dim=(-1, -2), keepdim=True) for part, sign in zip(reps, self.signs_tr)]
            scalars_mag = [(part*part).sum(dim=(-1, -2), keepdim=True) for part in reps]

            scalars_full = [torch.cat([s_tr, s_mag], dim=-1) for s_tr, s_mag in zip(scalars_tr, scalars_mag)]

            scalars = [scalars] + scalars_full

            scalars = torch.cat(scalars, dim=-3)

        return scalars

############# Output of network #############

class OutputLinear(nn.Module):
    """
    Module to create prediction based upon a set of rotationally invariant
    atom feature vectors. This is performed in a permutation invariant way
    by using a (batch-masked) sum over all atoms, and then applying a
    linear mixing layer to predict a single output.

    Parameters
    ----------
    num_scalars : :class:`int`
        Number scalars that will be used in the prediction at the output
        of the network.
    bias : :class:`bool`, optional
        Include a bias term in the linear mixing level.
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, num_scalars, bias=True, device=torch.device('cpu'), dtype=torch.float):
        super(OutputLinear, self).__init__()

        self.num_scalars = num_scalars
        self.bias = bias

        self.lin = nn.Linear(2*num_scalars, 1, bias=bias)
        self.lin.to(device=device, dtype=dtype)

        self.zero = torch.tensor(0, dtype=dtype, device=device)

    def forward(self, atom_scalars, atom_mask):
        """
        Forward step for :class:`OutputLinear`

        Parameters
        ----------
        atom_scalars : :class:`torch.Tensor`
            Scalar features for each atom used to predict the final learning target.
        atom_mask : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.

        Returns
        -------
        predict : :class:`torch.Tensor`
            Tensor used for predictions.
        """
        s = atom_scalars.shape
        atom_scalars = atom_scalars.view((s[0], s[1], -1)).sum(1)  # No masking needed b/c summing over atoms

        predict = self.lin(atom_scalars)

        predict = predict.squeeze(-1)

        return predict

class OutputPMLP(nn.Module):
    """
    Module to create prediction based upon a set of rotationally invariant
    atom feature vectors.

    This is peformed in a three-step process::

    (1) A MLP is applied to each set of scalar atom-features.
    (2) The environments are summed up.
    (3) Another MLP is applied to the output to predict a single learning target.

    Parameters
    ----------
    num_scalars : :class:`int`
        Number scalars that will be used in the prediction at the output
        of the network.
    bias : :class:`bool`, optional
        Include a bias term in the linear mixing level.
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, num_scalars, num_mixed=64, activation='leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(OutputPMLP, self).__init__()

        self.num_scalars = num_scalars
        self.num_mixed = num_mixed

        self.mlp1 = BasicMLP(2*num_scalars, num_mixed, num_hidden=1, activation=activation, device=device, dtype=dtype)
        self.mlp2 = BasicMLP(num_mixed, 1, num_hidden=1, activation=activation, device=device, dtype=dtype)

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, atom_scalars, atom_mask):
        """
        Forward step for :class:`OutputPMLP`

        Parameters
        ----------
        atom_scalars : :class:`torch.Tensor`
            Scalar features for each atom used to predict the final learning target.
        atom_mask : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.

        Returns
        -------
        predict : :class:`torch.Tensor`
            Tensor used for predictions.
        """
        # Reshape scalars appropriately
        atom_scalars = atom_scalars.view(atom_scalars.shape[:2] + (2*self.num_scalars,))

        # First MLP applied to each atom
        x = self.mlp1(atom_scalars)

        # Reshape to sum over each atom in molecules, setting non-existent atoms to zero.
        atom_mask = atom_mask.unsqueeze(-1)
        x = torch.where(atom_mask, x, self.zero).sum(1)

        # Prediction on permutation invariant representation of molecules
        predict = self.mlp2(x)

        predict = predict.squeeze(-1)

        return predict

class RadialFilters(nn.Module):
    """
    Generate a set of learnable scalar functions for the aggregation/point-wise
    convolution step.

    One set of radial filters is created for each irrep (l = 0, ..., max_sh).

    Parameters
    ----------
    max_sh : :class:`int`
        Maximum l to use for the spherical harmonics.
    basis_set : iterable of :class:`int`
        Parameters of basis set to use. See :class:`RadPolyTrig` for more details.
    num_channels_out : :class:`int`
        Number of output channels to mix the resulting function into if mix
        is set to True in RadPolyTrig
    num_levels : :class:`int`
        Number of CG levels in the Cormorant.
    """
    def __init__(self, max_sh, basis_set, num_channels_out,
                 num_levels, device=torch.device('cpu'), dtype=torch.float):
        super(RadialFilters, self).__init__()

        self.num_levels = num_levels
        self.max_sh = max_sh

        rad_funcs = [RadPolyTrig(max_sh[level], basis_set, num_channels_out[level], device=device, dtype=dtype) for level in range(self.num_levels)]
        self.rad_funcs = nn.ModuleList(rad_funcs)
        self.tau = [rad_func.tau for rad_func in self.rad_funcs]

        self.num_rad_channels = self.tau[0][0]

        # Other things
        self.device = device
        self.dtype = dtype

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, norms, base_mask):
        """
        Forward pass of the network.

        Parameters
        ----------
        norms : :class:`torch.Tensor`
            Pairwise distance matrix between atoms.
        base_mask : :class:`torch.Tensor`
            Masking tensor with 1s on locations that correspond to active edges
            and zero otherwise.

        Returns
        -------
        rad_func_vals :  list of :class:`RadPolyTrig`
            Values of the radial functions.
        """

        return [rad_func(norms, base_mask) for rad_func in self.rad_funcs]

class RadPolyTrig(nn.Module):
    """
    A variation/generalization of spherical bessel functions.
    Rather than than introducing the bessel functions explicitly we just write out a basis
    that can produce them. Then, when apply a weight mixing matrix to reduce the number of channels
    at the end.
    """
    def __init__(self, max_sh, basis_set, num_channels, mix=False, device=torch.device('cpu'), dtype=torch.float):
        super(RadPolyTrig, self).__init__()

        trig_basis, rpow = basis_set
        self.rpow = rpow
        self.max_sh = max_sh

        assert(trig_basis >= 0 and rpow >= 0)

        self.num_rad = (trig_basis+1)*(rpow+1)
        self.num_channels = num_channels

        # This instantiates a set of functions sin(2*pi*n*x/a), cos(2*pi*n*x/a) with a=1.
        self.scales = torch.cat([torch.arange(trig_basis+1), torch.arange(trig_basis+1)]).view(1, 1, 1, -1).to(device=device, dtype=dtype)
        self.phases = torch.cat([torch.zeros(trig_basis+1), pi/2*torch.ones(trig_basis+1)]).view(1, 1, 1, -1).to(device=device, dtype=dtype)

        # This avoids the sin(0*r + 0) = 0 part from wasting computations.
        self.phases[0, 0, 0, 0] = pi/2

        # Now, make the above learnable
        self.scales = nn.Parameter(self.scales)
        self.phases = nn.Parameter(self.phases)

        # If desired, mix the radial components to a desired shape
        self.mix = mix
        if (mix == 'cplx') or (mix is True):
            self.linear = nn.ModuleList([nn.Linear(2*self.num_rad, 2*self.num_channels).to(device=device, dtype=dtype) for _ in range(max_sh+1)])
            self.tau = SO3Tau((num_channels,) * (max_sh + 1))
        elif mix == 'real':
            self.linear = nn.ModuleList([nn.Linear(2*self.num_rad, self.num_channels).to(device=device, dtype=dtype) for _ in range(max_sh+1)])
            self.tau = SO3Tau((num_channels,) * (max_sh + 1))
        elif (mix == 'none') or (mix is False):
            self.linear = None
            self.tau = SO3Tau((self.num_rad,) * (max_sh + 1))
        else:
            raise ValueError('Can only specify mix = real, cplx, or none! {}'.format(mix))

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, norms, edge_mask):
        # Shape to resize at end
        s = norms.shape

        # Mask and reshape
        edge_mask = (edge_mask * (norms > 0)).unsqueeze(-1)
        norms = norms.unsqueeze(-1)

        # Get inverse powers
        rad_powers = torch.stack([torch.where(edge_mask, norms.pow(-pow), self.zero) for pow in range(self.rpow+1)], dim=-1)

        # Calculate trig functions
        rad_trig = torch.where(edge_mask, torch.sin((2*pi*self.scales)*norms+self.phases), self.zero).unsqueeze(-1)

        # Take the product of the radial powers and the trig components and reshape
        rad_prod = (rad_powers*rad_trig).view(s + (1, 2*self.num_rad,))

        # Apply linear mixing function, if desired
        if self.mix == 'cplx':
            radial_functions = [linear(rad_prod).view(s + (self.num_channels, 2)) for linear in self.linear]
        elif self.mix == 'real':
            radial_functions = [linear(rad_prod).view(s + (self.num_channels,)) for linear in self.linear]
            # Hack because real-valued SO3Scalar class has not been implemented yet.
            # TODO: Implement real-valued SO3Scalar and fix this...
            radial_functions = [torch.stack([rad, torch.zeros_like(rad)], dim=-1) for rad in radial_functions]
        else:
            radial_functions = [rad_prod.view(s + (self.num_rad, 2))] * (self.max_sh + 1)

        return SO3Scalar(radial_functions)

class MixReps(CGModule):
    """
    Module to linearly mix a representation from an input type `tau_in` to an
    output type `tau_out`.

    Input must have pre-defined types `tau_in` and `tau_out`.

    Parameters
    ----------
    tau_in : :obj:`SO3Tau` (or compatible object).
        Input tau of representation.
    tau_out : :obj:`SO3Tau` (or compatible object), or :obj:`int`.
        Input tau of representation. If an :obj:`int` is input,
        the output type will be set to `tau_out` for each
        parameter in the network.
    real : :obj:`bool`, optional
        Use purely real mixing weights.
    weight_init : :obj:`str`, optional
        String to set type of weight initialization.
    gain : :obj:`float`, optional
        Gain to scale initialized weights to.

    device : :obj:`torch.device`, optional
        Device to initialize weights to.
    dtype : :obj:`torch.dtype`, optional
        Data type to initialize weights to.

    """
    def __init__(self, tau_in, tau_out, real=False, weight_init='randn', gain=1,
                 device=None, dtype=None):
        super().__init__(device=device, dtype=dtype)
        tau_in = SO3Tau(tau_in)
        tau_out = SO3Tau(tau_out) if type(tau_out) is not int else tau_out

        # Allow one to set the output tau to a pre-specified number of output channels.
        if type(tau_out) is int:
            tau_out = [tau_out] * len(tau_in)

        self.tau_in = SO3Tau(tau_in)
        self.tau_out = SO3Tau(tau_out)
        self.real = real

        if weight_init is 'randn':
            weights = SO3Weight.randn(self.tau_in, self.tau_out, device=device, dtype=dtype)
        elif weight_init is 'rand':
            weights = SO3Weight.rand(self.tau_in, self.tau_out, device=device, dtype=dtype)
            weights = 2*weights - 1
        else:
            raise NotImplementedError('weight_init can only be randn or rand for now')

        gain = [gain / max(shape) for shape in weights.shapes]
        weights = gain * weights

        self.weights = weights.as_parameter()

    def forward(self, rep):
        """
        Linearly mix a represention.

        Parameters
        ----------
        rep : :obj:`list` of :obj:`torch.Tensor`
            Representation to mix.

        Returns
        -------
        rep : :obj:`list` of :obj:`torch.Tensor`
            Mixed representation.
        """
        if SO3Tau.from_rep(rep) != self.tau_in:
            raise ValueError('Tau of input rep does not match initialized tau!'
                            ' rep: {} tau: {}'.format(SO3Tau.from_rep(rep), self.tau_in))

        return so3_torch.mix(self.weights, rep)

    @property
    def tau(self):
        return self.tau_out

class CatReps(Module):
    """
    Module to concanteate a list of reps. Specify input type for error checking
    and to allow network to fit into main architecture.

    Parameters
    ----------
    taus_in : :obj:`list` of :obj:`SO3Tau` or compatible.
        List of taus of input reps.
    maxl : :obj:`bool`, optional
        Maximum weight to include in concatenation.
    """
    def __init__(self, taus_in, maxl=None):
        super().__init__()

        self.taus_in = taus_in = [SO3Tau(tau) for tau in taus_in if tau]

        if maxl is None:
            maxl = max([tau.maxl for tau in taus_in])
        self.maxl = maxl

        self.tau_out = reduce(lambda x,y: x & y, taus_in)[:self.maxl+1]

    def forward(self, reps):
        """
        Concatenate a list of reps

        Parameters
        ----------
        reps : :obj:`list` of :obj:`SO3Tensor` subclasses
            List of representations to concatenate.

        Returns
        -------
        reps_cat : :obj:`list` of :obj:`torch.Tensor`

        """
        # Drop Nones
        reps = [rep for rep in reps if rep is not None]

        # Error checking
        reps_taus_in = [rep.tau for rep in reps]
        if reps_taus_in != self.taus_in:
            raise ValueError('Tau of input reps does not match predefined version!'
                                'got: {} expected: {}'.format(reps_taus_in, self.taus_in))

        if self.maxl is not None:
            reps = [rep.truncate(self.maxl) for rep in reps]

        return so3_torch.cat(reps)

    @property
    def tau(self):
        return self.tau_out

class CatMixReps(CGModule):
    """
    Module to concatenate mix a list of representation representations using
    :obj:`cormorant.nn.CatReps`, and then linearly mix them using
    :obj:`cormorant.nn.MixReps`.

    Parameters
    ----------
    taus_in : List of :obj:`SO3Tau` (or compatible object).
        List of input tau of representation.
    tau_out : :obj:`SO3Tau` (or compatible object), or :obj:`int`.
        Input tau of representation. If an :obj:`int` is input,
        the output type will be set to `tau_out` for each
        parameter in the network.
    maxl : :obj:`bool`, optional
        Maximum weight to include in concatenation.
    real : :obj:`bool`, optional
        Use purely real mixing weights.
    weight_init : :obj:`str`, optional
        String to set type of weight initialization.
    gain : :obj:`float`, optional
        Gain to scale initialized weights to.

    device : :obj:`torch.device`, optional
        Device to initialize weights to.
    dtype : :obj:`torch.dtype`, optional
        Data type to initialize weights to.

    """
    def __init__(self, taus_in, tau_out, maxl=None,
                 real=False, weight_init='randn', gain=1,
                 device=None, dtype=None):
        super().__init__(device=device, dtype=dtype)

        self.cat_reps = CatReps(taus_in, maxl=maxl)
        self.mix_reps = MixReps(self.cat_reps.tau, tau_out,
                                real=real, weight_init=weight_init, gain=gain,
                                device=device, dtype=dtype)

        self.taus_in = taus_in
        self.tau_out = SO3Tau(self.mix_reps)

    def forward(self, reps_in):
        """
        Concatenate and linearly mix a list of representations.

        Parameters
        ----------
        reps_in : :obj:`list` of :obj:`list` of :obj:`torch.Tensors`
            List of input representations.

        Returns
        -------
        reps_out : :obj:`list` of :obj:`torch.Tensors`
            Representation as a result of combining and mixing input reps.
        """
        reps_cat = self.cat_reps(reps_in)
        reps_out = self.mix_reps(reps_cat)

        return reps_out

    @property
    def tau(self):
        return self.tau_out

class NoLayer(nn.Module):
    """
    Layer that does nothing in the Cormorant architecture.

    This exists just to demonstrate the structure one would want if edge
    features were desired at the input/output.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    def forward(self, *args, **kwargs):
        return None

    @property
    def tau(self):
        return SO3Tau([])

    @property
    def num_scalars(self):
        return 0

# Save reps

def save_grads(reps):
    for part in reps: part.requires_grad_()
    def closure(part):
        def assign_grad(grad):
            if grad is not None: part.add_(grad)
            return None
        return assign_grad
    grads = [torch.zeros_like(part) for part in reps]
    for (part, grad) in zip(reps, grads): part.register_hook(closure(grad))

    return grads

def save_reps(reps_dict, to_save, retain_grad=False):
    if 'reps_out' not in to_save:
        to_save.append('reps_out')

    reps_dict = {key: val for key, val in reps_dict.items() if (key in to_save and len(val) > 0)}

    if retain_grad:
        reps_dict.update({key+'_grad': save_grads(val) for key, val in reps_dict.items()})

    return reps_dict

def broadcastable(tau1, tau2):
    for t1, t2 in zip(tau1[::-1], tau2[::-1]):
        if not (t1 == 1 or t2 == 1 or t1 == t2):
            return False
    return True

def conjugate_rep(rep):
    repc = [part.clone() for part in rep]
    for part in repc:
        part[..., 1] *= -1
    return repc

class CormorantQM9(CGModule):
    """
    Basic Cormorant Network used to train QM9 results in Cormorant paper.

    Parameters
    ----------
    maxl : :obj:`int` of :obj:`list` of :obj:`int`
        Maximum weight in the output of CG products. (Expanded to list of
        length :obj:`num_cg_levels`)
    max_sh : :obj:`int` of :obj:`list` of :obj:`int`
        Maximum weight in the output of the spherical harmonics  (Expanded to list of
        length :obj:`num_cg_levels`)
    num_cg_levels : :obj:`int`
        Number of cg levels to use.
    num_channels : :obj:`int` of :obj:`list` of :obj:`int`
        Number of channels that the output of each CG are mixed to (Expanded to list of
        length :obj:`num_cg_levels`)
    num_species : :obj:`int`
        Number of species of atoms included in the input dataset.

    device : :obj:`torch.device`
        Device to initialize the level to
    dtype : :obj:`torch.dtype`
        Data type to initialize the level to level to
    cg_dict : :obj:`nn.cg_lib.CGDict`
    """
    def __init__(self, maxl, max_sh, num_cg_levels, num_channels, num_species,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 weight_init, level_gain, charge_power, basis_set,
                 charge_scale, gaussian_mask,
                 top, input, num_mpnn_layers, activation='leakyrelu',
                 device=None, dtype=None, cg_dict=None):

        logging.info('Initializing network!')
        level_gain = expand_var_list(level_gain, num_cg_levels)

        hard_cut_rad = expand_var_list(hard_cut_rad, num_cg_levels)
        soft_cut_rad = expand_var_list(soft_cut_rad, num_cg_levels)
        soft_cut_width = expand_var_list(soft_cut_width, num_cg_levels)

        maxl = expand_var_list(maxl, num_cg_levels)
        max_sh = expand_var_list(max_sh, num_cg_levels)
        num_channels = expand_var_list(num_channels, num_cg_levels+1)

        logging.info('hard_cut_rad: {}'.format(hard_cut_rad))
        logging.info('soft_cut_rad: {}'.format(soft_cut_rad))
        logging.info('soft_cut_width: {}'.format(soft_cut_width))
        logging.info('maxl: {}'.format(maxl))
        logging.info('max_sh: {}'.format(max_sh))
        logging.info('num_channels: {}'.format(num_channels))

        super().__init__(maxl=max(maxl+max_sh), device=device, dtype=dtype, cg_dict=cg_dict)
        device, dtype, cg_dict = self.device, self.dtype, self.cg_dict

        print('CGDICT', cg_dict.maxl)

        self.num_cg_levels = num_cg_levels
        self.num_channels = num_channels
        self.charge_power = charge_power
        self.charge_scale = charge_scale
        self.num_species = num_species

        # Set up spherical harmonics
        self.sph_harms = SphericalHarmonicsRel(max(max_sh), conj=True,
                                               device=device, dtype=dtype, cg_dict=cg_dict)

        # Set up position functions, now independent of spherical harmonics
        self.rad_funcs = RadialFilters(max_sh, basis_set, num_channels, num_cg_levels,
                                       device=self.device, dtype=self.dtype)
        tau_pos = self.rad_funcs.tau

        num_scalars_in = self.num_species * (self.charge_power + 1)
        num_scalars_out = num_channels[0]

        self.input_func_atom = InputMPNN(num_scalars_in, num_scalars_out, num_mpnn_layers,
                                         soft_cut_rad[0], soft_cut_width[0], hard_cut_rad[0],
                                         activation=activation, device=self.device, dtype=self.dtype)
        self.input_func_edge = NoLayer()

        tau_in_atom = self.input_func_atom.tau
        tau_in_edge = self.input_func_edge.tau

        self.cormorant_cg = CormorantCG(maxl, max_sh, tau_in_atom, tau_in_edge,
                     tau_pos, num_cg_levels, num_channels, level_gain, weight_init,
                     cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                     cat=True, gaussian_mask=False,
                     device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)

        tau_cg_levels_atom = self.cormorant_cg.tau_levels_atom
        tau_cg_levels_edge = self.cormorant_cg.tau_levels_edge

        self.get_scalars_atom = GetScalarsAtom(tau_cg_levels_atom,
                                               device=self.device, dtype=self.dtype)
        self.get_scalars_edge = NoLayer()

        num_scalars_atom = self.get_scalars_atom.num_scalars
        num_scalars_edge = self.get_scalars_edge.num_scalars

        self.output_layer_atom = OutputPMLP(num_scalars_atom, activation=activation,
                                            device=self.device, dtype=self.dtype)
        self.output_layer_edge = NoLayer()

        logging.info('Model initialized. Number of parameters: {}'.format(
            sum([p.nelement() for p in self.parameters()])))

    def forward(self, data, covariance_test=False):
        """
        Runs a forward pass of the network.

        Parameters
        ----------
        data : :obj:`dict`
            Dictionary of data to pass to the network.
        covariance_test : :obj:`bool`, optional
            If true, returns all of the atom-level representations twice.

        Returns
        -------
        prediction : :obj:`torch.Tensor`
            The output of the layer
        """
        # Get and prepare the data
        atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions = self.prepare_input(data)

        # Calculate spherical harmonics and radial functions
        spherical_harmonics, norms = self.sph_harms(atom_positions, atom_positions)
        rad_func_levels = self.rad_funcs(norms, edge_mask * (norms > 0))

        # Prepare the input reps for both the atom and edge network
        atom_reps_in = self.input_func_atom(atom_scalars, atom_mask, edge_scalars, edge_mask, norms)
        edge_net_in = self.input_func_edge(atom_scalars, atom_mask, edge_scalars, edge_mask, norms)

        # Clebsch-Gordan layers central to the network
        atoms_all, edges_all = self.cormorant_cg(atom_reps_in, atom_mask, edge_net_in, edge_mask,
                                                 rad_func_levels, norms, spherical_harmonics)

        # Construct scalars for network output
        atom_scalars = self.get_scalars_atom(atoms_all)
        edge_scalars = self.get_scalars_edge(edges_all)

        # Prediction in this case will depend only on the atom_scalars. Can make
        # it more general here.
        prediction = self.output_layer_atom(atom_scalars, atom_mask)

        # Covariance test
        if covariance_test:
            return prediction, atoms_all, atoms_all
        else:
            return prediction

    def prepare_input(self, data):
        """
        Extracts input from data class

        Parameters
        ----------
        data : ?????
            Information on the state of the system.

        Returns
        -------
        atom_scalars : :obj:`torch.Tensor`
            Tensor of scalars for each atom.
        atom_mask : :obj:`torch.Tensor`
            Mask used for batching data.
        atom_positions: :obj:`torch.Tensor`
            Positions of the atoms
        edge_mask: :obj:`torch.Tensor`
            Mask used for batching data.
        """
        charge_power, charge_scale, device, dtype = self.charge_power, self.charge_scale, self.device, self.dtype

        atom_positions = data['positions'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)

        atom_mask = data['atom_mask'].to(device)
        edge_mask = data['edge_mask'].to(device)

        charge_tensor = (charges.unsqueeze(-1)/charge_scale).pow(torch.arange(charge_power+1., device=device, dtype=dtype))
        charge_tensor = charge_tensor.view(charges.shape + (1, charge_power+1))
        atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))

        edge_scalars = torch.tensor([])

        return atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions

def expand_var_list(var, num_cg_levels):
    if type(var) is list:
        var_list = var + (num_cg_levels-len(var))*[var[-1]]
    elif type(var) in [float, int]:
        var_list = [var] * num_cg_levels
    else:
        raise ValueError('Incorrect type {}'.format(type(var)))
    return var_list

class CormorantMD17(CGModule):
    """
    Basic Cormorant Network used to train MD17 results in Cormorant paper.

    Parameters
    ----------
    maxl : :obj:`int` of :class:`list` of :class:`int`
        Maximum weight in the output of CG products. (Expanded to list of
        length :obj:`num_cg_levels`)
    max_sh : :class:`int` of :class:`list` of :class:`int`
        Maximum weight in the output of the spherical harmonics  (Expanded to list of
        length :obj:`num_cg_levels`)
    num_cg_levels : :class:`int`
        Number of cg levels to use.
    num_channels : :class:`int` of :class:`list` of :class:`int`
        Number of channels that the output of each CG are mixed to (Expanded to list of
        length :obj:`num_cg_levels`)
    num_species : :class:`int`
        Number of species of atoms included in the input dataset.
    device : :class:`torch.device`
        Device to initialize the level to
    dtype : :class:`torch.torch.dtype`
        Data type to initialize the level to level to
    dummy_torch_obj: :class:`torch.Tensor`
        Object created for testing external links.
    cg_dict : :class:`CGDict <cormorant.cg_lib.CGDict>`
        Clebsch-gordan dictionary object.
    """
    def __init__(self, maxl, max_sh, num_cg_levels, num_channels, num_species,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 weight_init, level_gain, charge_power, basis_set,
                 charge_scale, gaussian_mask,
                 top, input, num_mpnn_layers, activation='leakyrelu',
                 device=None, dtype=None, cg_dict=None):

        logging.info('Initializing network!')
        level_gain = expand_var_list(level_gain, num_cg_levels)

        hard_cut_rad = expand_var_list(hard_cut_rad, num_cg_levels)
        soft_cut_rad = expand_var_list(soft_cut_rad, num_cg_levels)
        soft_cut_width = expand_var_list(soft_cut_width, num_cg_levels)

        maxl = expand_var_list(maxl, num_cg_levels)
        max_sh = expand_var_list(max_sh, num_cg_levels)
        num_channels = expand_var_list(num_channels, num_cg_levels+1)

        logging.info('hard_cut_rad: {}'.format(hard_cut_rad))
        logging.info('soft_cut_rad: {}'.format(soft_cut_rad))
        logging.info('soft_cut_width: {}'.format(soft_cut_width))
        logging.info('maxl: {}'.format(maxl))
        logging.info('max_sh: {}'.format(max_sh))
        logging.info('num_channels: {}'.format(num_channels))

        super().__init__(maxl=max(maxl+max_sh), device=device, dtype=dtype, cg_dict=cg_dict)
        device, dtype, cg_dict = self.device, self.dtype, self.cg_dict

        print('CGDICT', cg_dict.maxl)

        self.num_cg_levels = num_cg_levels
        self.num_channels = num_channels
        self.charge_power = charge_power
        self.charge_scale = charge_scale
        self.num_species = num_species

        # Set up spherical harmonics
        self.sph_harms = SphericalHarmonicsRel(max(max_sh), conj=True,
                                               device=device, dtype=dtype, cg_dict=cg_dict)

        # Set up position functions, now independent of spherical harmonics
        self.rad_funcs = RadialFilters(max_sh, basis_set, num_channels, num_cg_levels,
                                       device=self.device, dtype=self.dtype)
        tau_pos = self.rad_funcs.tau

        num_scalars_in = self.num_species * (self.charge_power + 1)
        num_scalars_out = num_channels[0]

        self.input_func_atom = InputLinear(num_scalars_in, num_scalars_out,
                                           device=self.device, dtype=self.dtype)
        self.input_func_edge = NoLayer()

        tau_in_atom = self.input_func_atom.tau
        tau_in_edge = self.input_func_edge.tau

        self.cormorant_cg = CormorantCG(maxl, max_sh, tau_in_atom, tau_in_edge,
                     tau_pos, num_cg_levels, num_channels, level_gain, weight_init,
                     cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                     cat=True, gaussian_mask=False,
                     device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)

        tau_cg_levels_atom = self.cormorant_cg.tau_levels_atom
        tau_cg_levels_edge = self.cormorant_cg.tau_levels_edge

        self.get_scalars_atom = GetScalarsAtom(tau_cg_levels_atom,
                                               device=self.device, dtype=self.dtype)
        self.get_scalars_edge = NoLayer()

        num_scalars_atom = self.get_scalars_atom.num_scalars
        num_scalars_edge = self.get_scalars_edge.num_scalars

        self.output_layer_atom = OutputLinear(num_scalars_atom, bias=True,
                                              device=self.device, dtype=self.dtype)
        self.output_layer_edge = NoLayer()

        logging.info('Model initialized. Number of parameters: {}'.format(
            sum([p.nelement() for p in self.parameters()])))

    def forward(self, data, covariance_test=False):
        """
        Runs a forward pass of the network.

        Parameters
        ----------
        data : :obj:`dict`
            Dictionary of data to pass to the network.
        covariance_test : :obj:`bool`, optional
            If true, returns all of the atom-level representations twice.

        Returns
        -------
        prediction : :obj:`torch.Tensor`
            The output of the layer
        """
        # Get and prepare the data
        atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions = self.prepare_input(data)

        # Calculate spherical harmonics and radial functions
        spherical_harmonics, norms = self.sph_harms(atom_positions, atom_positions)
        rad_func_levels = self.rad_funcs(norms, edge_mask * (norms > 0))

        # Prepare the input reps for both the atom and edge network
        atom_reps_in = self.input_func_atom(atom_scalars, atom_mask, edge_scalars, edge_mask, norms)
        edge_net_in = self.input_func_edge(atom_scalars, atom_mask, edge_scalars, edge_mask, norms)

        # Clebsch-Gordan layers central to the network
        atoms_all, edges_all = self.cormorant_cg(atom_reps_in, atom_mask, edge_net_in, edge_mask,
                                                 rad_func_levels, norms, spherical_harmonics)

        # Construct scalars for network output
        atom_scalars = self.get_scalars_atom(atoms_all)
        edge_scalars = self.get_scalars_edge(edges_all)

        # Prediction in this case will depend only on the atom_scalars. Can make
        # it more general here.
        prediction = self.output_layer_atom(atom_scalars, atom_mask)

        # Covariance test
        if covariance_test:
            return prediction, atoms_all, atoms_all
        else:
            return prediction

    def prepare_input(self, data):
        """
        Extracts input from data class

        Parameters
        ----------
        data : ?????
            Information on the state of the system.

        Returns
        -------
        atom_scalars : :obj:`torch.Tensor`
            Tensor of scalars for each atom.
        atom_mask : :obj:`torch.Tensor`
            Mask used for batching data.
        atom_positions: :obj:`torch.Tensor`
            Positions of the atoms
        edge_mask: :obj:`torch.Tensor`
            Mask used for batching data.
        """
        charge_power, charge_scale, device, dtype = self.charge_power, self.charge_scale, self.device, self.dtype

        atom_positions = data['positions'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)

        atom_mask = data['atom_mask'].to(device)
        edge_mask = data['edge_mask'].to(device)

        charge_tensor = (charges.unsqueeze(-1)/charge_scale).pow(torch.arange(charge_power+1., device=device, dtype=dtype))
        charge_tensor = charge_tensor.view(charges.shape + (1, charge_power+1))
        atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))

        edge_scalars = torch.tensor([])

        return atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions

def expand_var_list(var, num_cg_levels):
    if type(var) is list:
        var_list = var + (num_cg_levels-len(var))*[var[-1]]
    elif type(var) in [float, int]:
        var_list = [var] * num_cg_levels
    else:
        raise ValueError('Incorrect type {}'.format(type(var)))
    return var_list

class CormorantEdgeLevel(CGModule):
    """
    Scalar edge level as part of the Clebsch-Gordan layers are described in the
    Cormorant paper.

    The input takes in an edge network from a previous level,
    the representations from the previous level, and also a set of
    scalar functions of the relative positions between all atoms.

    Parameters
    ----------
    tau_atom : :class:`SO3Tau`
        Multiplicity (tau) of the input atom representations
    tau_edge : :class:`SO3Tau`
        Multiplicity (tau) of the input edge layer representations
    tau_pos : :class:`SO3Tau`
        Multiplicity (tau) of the input set of position functions
    nout : :obj:`int`
        Number of output channels to mix the concatenated scalars to.
    max_sh : :obj:`int`
        Maximum weight of the spherical harmonics.
    cutoff_type : :obj:`str`
        `cutoff_type` to be passed to :class:`cormorant.nn.MaskLevel`
    hard_cut_rad : :obj:`float`
        `hard_cut_rad` to be passed to :class:`cormorant.nn.MaskLevel`
    soft_cut_rad : :obj:`float`
        `soft_cut_rad` to be passed to :class:`cormorant.nn.MaskLevel`
    soft_cut_width : :obj:`float`
        `soft_cut_width` to be passed to :class:`cormorant.nn.MaskLevel`
    gaussian_mask : :obj:`bool`
        `gaussian_mask` to be passed to :class:`cormorant.nn.MaskLevel`
    cat : :obj:`bool`
        Concatenate all the scalars in :class:`cormorant.nn.DotMatrix`
    device : :obj:`torch.device`
        Device to initialize the level to
    dtype : :obj:`torch.dtype`
        Data type to initialize the level to
    """
    def __init__(self, tau_atom, tau_edge, tau_pos, nout, max_sh,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 cat=True, gaussian_mask=False,
                 device=None, dtype=None):
        super().__init__(device=device, dtype=dtype)
        device, dtype = self.device, self.dtype

        # Set up type of edge network depending on specified input operations
        self.dot_matrix = DotMatrix(tau_atom, cat=cat,
                                    device=self.device, dtype=self.dtype)
        tau_dot = self.dot_matrix.tau

        # Set up mixing layer
        edge_taus = [tau for tau in (tau_edge, tau_dot, tau_pos) if tau is not None]
        self.cat_mix = CatMixReps(edge_taus, nout, real=False, maxl=max_sh,
                                  device=self.device, dtype=self.dtype)
        self.tau = self.cat_mix.tau

        # Set up edge mask layer
        self.mask_layer = MaskLevel(nout, hard_cut_rad, soft_cut_rad, soft_cut_width, cutoff_type,
                                    gaussian_mask=gaussian_mask, device=self.device, dtype=self.dtype)

    def forward(self, edge_in, atom_reps, pos_funcs, base_mask, norms):
        # Caculate the dot product matrix.
        edge_dot = self.dot_matrix(atom_reps)

        # Concatenate and mix the three different types of edge features together
        edge_mix = self.cat_mix([edge_in, edge_dot, pos_funcs])

        # Apply mask to layer -- For now, only can be done after mixing.
        edge_net = self.mask_layer(edge_mix, base_mask, norms)

        return edge_net


class CormorantAtomLevel(CGModule):
    """
    Atom level as part of the Clebsch-Gordan layers are described in the
    Cormorant paper.

    The input takes in an edge network from a previous level, along
    with a set of representations that correspond to edges between atoms.
    Applies a masked Clebsh-Gordan operation.

    Parameters
    ----------
    tau_in : :class:`SO3Tau`
        Multiplicity (tau) of the input atom representations
    tau_pos : :class:`SO3Tau`
        Multiplicity (tau) of the input set of position functions
    maxl : :obj:`int`
        Maximum weight of the spherical harmonics.
    num_channels : :obj:`int`
        Number of output channels to mix the concatenated :class:`SO3Vec` to.
    weight_init : :obj:`str`
        Weight initialization function.
    level_gain : :obj:`int`
        Gain for the weights at each level.

    device : :obj:`torch.device`
        Device to initialize the level to
    dtype : :obj:`torch.dtype`
        Data type to initialize the level to
    cg_dict : :obj:`cormorant.cg_lib.CGDict`
        Clebsch-Gordan dictionary for the CG levels.

    """
    def __init__(self, tau_in, tau_pos, maxl, num_channels, level_gain, weight_init,
                 device=None, dtype=None, cg_dict=None):
        super().__init__(maxl=maxl, device=device, dtype=dtype, cg_dict=cg_dict)
        device, dtype, cg_dict = self.device, self.dtype, self.cg_dict

        self.tau_in = tau_in
        self.tau_pos = tau_pos

        # Operations linear in input reps
        self.cg_aggregate = CGProduct(tau_pos, tau_in, maxl=self.maxl, aggregate=True,
                                      device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)
        tau_ag = list(self.cg_aggregate.tau)

        self.cg_power = CGProduct(tau_in, tau_in, maxl=self.maxl,
                                  device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)
        tau_sq = list(self.cg_power.tau)

        self.cat_mix = CatMixReps([tau_ag, tau_in, tau_sq], num_channels,
                                  maxl=self.maxl, weight_init=weight_init, gain=level_gain,
                                  device=self.device, dtype=self.dtype)
        self.tau = self.cat_mix.tau

    def forward(self, atom_reps, edge_reps, mask):
        """
        Runs a forward pass of the network.

        Parameters
        ----------
        atom_reps : SO3Vec
            Representation of the atomic environment.
        edge_reps : SO3Vec
            Representation of the connections between atoms
        mask : pytorch Tensor
            Mask determining which elements of atom_reps are active.

        Returns
        -------
        reps_out : SO3Vec
            Output representation of the atomic environment.
        """
        # Aggregate information based upon edge reps
        reps_ag = self.cg_aggregate(edge_reps, atom_reps)

        # CG non-linearity for each atom
        reps_sq = self.cg_power(atom_reps, atom_reps)

        # Concatenate and mix results
        reps_out = self.cat_mix([reps_ag, atom_reps, reps_sq])

        return reps_out


class CormorantCG(CGModule):
    def __init__(self, maxl, max_sh, tau_in_atom, tau_in_edge, tau_pos,
                 num_cg_levels, num_channels,
                 level_gain, weight_init,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 cat=True, gaussian_mask=False,
                 device=None, dtype=None, cg_dict=None):
        super().__init__(device=device, dtype=dtype, cg_dict=cg_dict)
        device, dtype, cg_dict = self.device, self.dtype, self.cg_dict

        self.max_sh = max_sh

        tau_atom_in = atom_in.tau if type(tau_in_atom) is CGModule else tau_in_atom
        tau_edge_in = edge_in.tau if type(tau_in_edge) is CGModule else tau_in_edge

        logging.info('{} {}'.format(tau_atom_in, tau_edge_in))

        atom_levels = nn.ModuleList()
        edge_levels = nn.ModuleList()

        tau_atom, tau_edge = tau_atom_in, tau_edge_in

        for level in range(num_cg_levels):
            # First add the edge, since the output type determines the next level
            edge_lvl = CormorantEdgeLevel(tau_atom, tau_edge, tau_pos[level], num_channels[level], max_sh[level],
                                          cutoff_type, hard_cut_rad[level], soft_cut_rad[level], soft_cut_width[level],
                                          gaussian_mask=gaussian_mask, device=device, dtype=dtype)
            edge_levels.append(edge_lvl)
            tau_edge = edge_lvl.tau

            # Now add the NBody level
            atom_lvl = CormorantAtomLevel(tau_atom, tau_edge, maxl[level], num_channels[level+1],
                                          level_gain[level], weight_init,
                                          device=device, dtype=dtype, cg_dict=cg_dict)
            atom_levels.append(atom_lvl)
            tau_atom = atom_lvl.tau

            logging.info('{} {}'.format(tau_atom, tau_edge))

        self.atom_levels = atom_levels
        self.edge_levels = edge_levels

        self.tau_levels_atom = [level.tau for level in atom_levels]
        self.tau_levels_edge = [level.tau for level in edge_levels]

    def forward(self, atom_reps, atom_mask, edge_net, edge_mask, rad_funcs, norms, sph_harm):
        """
        Runs a forward pass of the Cormorant CG layers.

        Parameters
        ----------
        atom_reps :  SO3 Vector
            Input atom representations.
        atom_mask : :obj:`torch.Tensor` with data type `torch.byte`
            Batch mask for atom representations. Shape is
            :math:`(N_{batch}, N_{atom})`.
        edge_net : SO3 Scalar or None`
            Input edge scalar features.
        edge_mask : :obj:`torch.Tensor` with data type `torch.byte`
            Batch mask for atom representations. Shape is
            :math:`(N_{batch}, N_{atom}, N_{atom})`.
        rad_funcs : :obj:`list` of SO3 Scalars
            The (possibly learnable) radial filters.
        edge_mask : :obj:`torch.Tensor`
            Matrix of the magnitudes of relative position vectors of pairs of atoms.
            :math:`(N_{batch}, N_{atom}, N_{atom})`.
        sph_harm : SO3 Vector
            Representation of spherical harmonics calculated from the relative
            position vectors of pairs of points.

        Returns
        -------
        atoms_all : list of SO3 Vectors
            The concatenated output of the representations output at each level.
        edges_all : list of SO3 Scalars
            The concatenated output of the scalar edge network output at each level.
        """
        assert len(self.atom_levels) == len(self.edge_levels) == len(rad_funcs)

        # Construct iterated multipoles
        atoms_all = []
        edges_all = []

        for idx, (atom_level, edge_level, max_sh) in enumerate(zip(self.atom_levels, self.edge_levels, self.max_sh)):
            edge_net = edge_level(edge_net, atom_reps, rad_funcs[idx], edge_mask, norms)
            edge_reps = edge_net * sph_harm[:max_sh+1]
            atom_reps = atom_level(atom_reps, edge_reps, atom_mask)

            atoms_all.append(atom_reps)
            edges_all.append(edge_net)

        return atoms_all, edges_all


class CGDict():
    r"""
    A dictionary of Clebsch-Gordan (CG) coefficients to be used in CG operations.

    The CG coefficients

    .. math::
        \langle \ell_1, m_1, l_2, m_2 | l, m \rangle

    are used to decompose the tensor product of two
    irreps of maximum weights :math:`\ell_1` and :math:`\ell_2` into a direct
    sum of irreps with :math:`\ell = |\ell_1 -\ell_2|, \ldots, (\ell_1 + \ell_2)`.

    The coefficients for each :math:`\ell_1` and :math:`\ell_2`
    are stored as a :math:`D \times D` matrix :math:`C_{\ell_1,\ell_2}` ,
    where :math:`D = (2\ell_1+1)\times(2\ell_2+1)`.

    The module has a dict-like interface with keys :math:`(l_1, l_2)` for
    :math:`\ell_1, l_2 \leq l_{\rm max}`. Each value is a matrix of shape
    :math:`D \times D`, where :math:`D = (2l_1+1)\times(2l_2+1)`.
    The matrix has elements.

    Parameters
    ----------
    maxl: :class:`int`
        Maximum weight for which to calculate the Clebsch-Gordan coefficients.
        This refers to the maximum weight for the ``input tensors``, not the
        output tensors.
    transpose: :class:`bool`, optional
        Transpose the CG coefficient matrix for each :math:`(\ell_1, \ell_2)`.
        This cannot be modified after instantiation.
    device: :class:`torch.torch.device`, optional
        Device of CG dictionary.
    dtype: :class:`torch.torch.dtype`, optional
        Data type of CG dictionary.

    """

    def __init__(self, maxl=None, transpose=True, dtype=torch.float, device=torch.device('cpu')):

        self.dtype = dtype
        self.device = device
        self._transpose = transpose
        self._maxl = None
        self._cg_dict = {}

        if maxl is not None:
            self.update_maxl(maxl)

    @property
    def transpose(self):
        """
        Use "transposed" version of CG coefficients.
        """
        return self._transpose

    @property
    def maxl(self):
        """
        Maximum weight for CG coefficients.
        """
        return self._maxl

    def update_maxl(self, new_maxl):
        """
        Update maxl to a new (possibly larger) value. If the new_maxl is
        larger than the current maxl, new CG coefficients should be calculated
        and the cg_dict will be updated.

        Otherwise, do nothing.

        Parameters
        ----------
        new_maxl: :class:`int`
            New maximum weight.

        Return
        ------
        self: :class:`CGDict`
            Returns self with a possibly updated self.cg_dict.
        """
        # If self is already initialized, and maxl is sufficiently large, do nothing
        if self and (self.maxl >= new_maxl):
            return self

        # If self is false, old_maxl = 0 (uninitialized).
        # old_maxl = self.maxl if self else 0

        # Otherwise, update the CG coefficients.
        cg_dict_new = _gen_cg_dict(new_maxl, transpose=self.transpose, existing_keys=self._cg_dict.keys())

        # Ensure elements of new CG dict are on correct device.
        cg_dict_new = {key: val.to(device=self.device, dtype=self.dtype) for key, val in cg_dict_new.items()}

        # Now update the CG dict, and also update maxl
        self._cg_dict.update(cg_dict_new)
        self._maxl = new_maxl

        return self

    def to(self, dtype=None, device=None):
        """
        Convert CGDict() to a new device/dtype.

        Parameters
        ----------
        device : :class:`torch.torch.device`, optional
            Device to move the cg_dict to.
        dtype : :class:`torch.torch.dtype`, optional
            Data type to convert the cg_dict to.
        """
        if dtype is None and device is None:
            pass
        elif dtype is None and device is not None:
            self._cg_dict = {key: val.to(device=device) for key, val in self._cg_dict.items()}
            self.device = device
        elif dtype is not None and device is None:
            self._cg_dict = {key: val.to(dtype=dtype) for key, val in self._cg_dict.items()}
            self.dtype = dtype
        elif dtype is not None and device is not None:
            self._cg_dict = {key: val.to(device=device, dtype=dtype) for key, val in self._cg_dict.items()}
            self.device, self.dtype = device, dtype
        return self

    def keys(self):
        return self._cg_dict.keys()

    def values(self):
        return self._cg_dict.values()

    def items(self):
        return self._cg_dict.items()

    def __getitem__(self, idx):
        if not self:
            raise ValueError('CGDict() not initialized. Either set maxl, or use update_maxl()')
        return self._cg_dict[idx]

    def __bool__(self):
        """
        Check to see if CGDict has been properly initialized, since :maxl=-1: initially.
        """
        return self.maxl is not None


def _gen_cg_dict(maxl, transpose=False, existing_keys={}):
    """
    Generate all Clebsch-Gordan coefficients for a weight up to maxl.

    Parameters
    ----------
    maxl: :class:`int`
        Maximum weight to generate CG coefficients.

    Return
    ------
    cg_dict: :class:`dict`
        Dictionary of CG basis transformation matrices with keys :(l1, l2):,
        and matrices that convert a tensor product of irreps of type :l1: and :l2:
        into a direct sum of irreps :l: from :abs(l1-l2): to :l1+l2:
    """
    cg_dict = {}

    for l1 in range(maxl+1):
        for l2 in range(maxl+1):
            if (l1, l2) in existing_keys:
                continue

            lmin, lmax = abs(l1 - l2), l1 + l2
            N1, N2 = 2*l1+1, 2*l2+1
            N = N1*N2
            cg_mat = torch.zeros((N1, N2, N), dtype=torch.double)
            for l in range(lmin, lmax+1):
                l_off = l*l - lmin*lmin
                for m1 in range(-l1, l1+1):
                    for m2 in range(-l2, l2+1):
                        for m in range(-l, l+1):
                            if m == m1 + m2:
                                cg_mat[l1+m1, l2+m2, l+m+l_off] = _clebsch(l1, l2, l, m1, m2, m)

            cg_mat = cg_mat.view(N, N)
            if transpose:
                cg_mat = cg_mat.transpose(0, 1)
            cg_dict[(l1, l2)] = cg_mat

    return cg_dict


# Taken from http://qutip.org/docs/3.1.0/modules/qutip/utilities.html

# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

def _clebsch(j1, j2, j3, m1, m2, m3):
    """Calculates the Clebsch-Gordon coefficient
    for coupling (j1,m1) and (j2,m2) to give (j3,m3).

    Parameters
    ----------
    j1 : :class:`float`
        Total angular momentum 1.
    j2 : :class:`float`
        Total angular momentum 2.
    j3 : :class:`float`
        Total angular momentum 3.
    m1 : :class:`float`
        z-component of angular momentum 1.
    m2 : :class:`float`
        z-component of angular momentum 2.
    m3 : :class:`float`
        z-component of angular momentum 3.

    Returns
    -------
    cg_coeff : :class:`float`
        Requested Clebsch-Gordan coefficient.

    """
    if m3 != m1 + m2:
        return 0
    vmin = int(np.max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    C = np.sqrt((2.0 * j3 + 1.0) * factorial(j3 + j1 - j2)
                * factorial(j3 - j1 + j2) * factorial(j1 + j2 - j3)
                * factorial(j3 + m3) * factorial(j3 - m3)
                / (factorial(j1 + j2 + j3 + 1)
                * factorial(j1 - m1) * factorial(j1 + m1)
                * factorial(j2 - m2) * factorial(j2 + m2)))
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) / factorial(v) * \
            factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v) / \
            factorial(j3 - j1 + j2 - v) / factorial(j3 + m3 - v) / \
            factorial(v + j1 - j2 - m3)
    C = C * S
    return C


def cg_product_tau(tau1, tau2, maxl=inf):
    """
    Calulate output multiplicity of the CG Product of two SO3 Vectors
    given the multiplicty of two input SO3 Vectors.

    Parameters
    ----------
    tau1 : :class:`list` of :class:`int`, :class:`SO3Tau`.
        Multiplicity of first representation.

    tau2 : :class:`list` of :class:`int`, :class:`SO3Tau`.
        Multiplicity of second representation.

    maxl : :class:`int`
        Largest weight to include in CG Product.

    Return
    ------

    tau : :class:`SO3Tau`
        Multiplicity of output representation.

    """
    tau1 = SO3Tau(tau1)
    tau2 = SO3Tau(tau2)

    L1, L2 = tau1.maxl, tau2.maxl
    L = min(L1 + L2, maxl)

    tau = [0]*(L+1)

    for l1 in range(L1+1):
        for l2 in range(L2+1):
            lmin, lmax = abs(l1-l2), min(l1+l2, maxl)
            for l in range(lmin, lmax+1):
                tau[l] += tau1[l1]

    return SO3Tau(tau)


class CGProduct(CGModule):
    r"""
    Create new CGproduct object. Inherits from CGModule, and has access
    to the CGDict related features.

    Takes two lists of type

    .. math::

        [\tau^1_{\text{min} [l_1]}, \tau^1_{\text{min} [l_1]+1}, ..., \tau^1_{\text{max} [l_1]}],
        [\tau^2_{\text{min}[l_2]}, \tau^2_{\text{min}[l_2]+1}, ..., \tau^2_{\text{max} [l_2]}],

    and outputs a new SO3 vector of type:

    .. math::

        [\tau_{\text{min} [l]}, \tau_{\text{min} [l]+1}, ..., \tau_{\text{max_l}}]

    Each part can have an arbitrary number of batch dimensions. These batch
    dimensions must be broadcastable, unless the option :aggregate=True: is used.


    Parameters
    ----------
    tau : :class:`list` of :class:`int`, :class:`SO3Tau`, or object with `.tau` property.
        Multiplicity of the first SO(3) vector.
    tau : :class:`list` of :class:`int`, :class:`SO3Tau`, or object with `.tau` property.
        Multiplicity of the second SO(3) vector.
    minl : :class:`int`, optional
        Minimum weight to include in CG Product
    maxl : :class:`int`, optional
        Maximum weight to include in CG Product
    aggregate : :class:`bool`, optional
        Apply an "aggregation" operation, or a pointwise convolution
        with a :class:`SO3Vec` as a filter.
    cg_dict : :class:`CGDict`, optional
        Specify a Clebsch-Gordan dictionary. If not specified, one will be
        generated automatically at runtime based upon maxl.
    device : :class:`torch.torch.device`, optional
        Device to initialize the module and Clebsch-Gordan dictionary to.
    dtype : :class:`torch.torch.dtype`, optional
        Data type to initialize the module and Clebsch-Gordan dictionary to.

    """
    def __init__(self, tau1=None, tau2=None,
                 aggregate=False,
                 minl=0, maxl=inf, cg_dict=None, dtype=None, device=None):

        self.aggregate = aggregate

        if (maxl == inf) and cg_dict:
            maxl = cg_dict.maxl
        elif (maxl == inf) and (tau1 and tau2):
            maxl = max(len(tau1), len(tau2))
        elif (maxl == inf):
            raise ValueError('maxl is not defined, and was unable to retrieve get maxl from cg_dict or tau1 and tau2')

        super().__init__(cg_dict=cg_dict, maxl=maxl, device=device, dtype=dtype)

        self.set_taus(tau1, tau2)

        if (minl > 0):
            raise NotImplementedError('minl > 0 not yet implemented!')
        else:
            self.minl = 0

    def forward(self, rep1, rep2):
        """
        Performs the Clebsch-Gordan product.

        Parameters
        ----------
        rep1 : :class:`SO3Vec`
            First :class:`SO3Vec` in the CG product
        rep2 : :class:`SO3Vec`
            Second :class:`SO3Vec` in the CG product
        """
        if self.tau1 and self.tau1 != SO3Tau.from_rep(rep1):
            raise ValueError('Input rep1 does not match predefined tau!')

        if self.tau2 and self.tau2 != SO3Tau.from_rep(rep2):
            raise ValueError('Input rep2 does not match predefined tau!')

        return cg_product(self.cg_dict, rep1, rep2, maxl=self.maxl, minl=self.minl, aggregate=self.aggregate)

    @property
    def tau_out(self):
        if not(self.tau1) or not(self.tau2):
            raise ValueError('Module not intialized with input type!')
        tau_out = cg_product_tau(self.tau1, self.tau2, maxl=self.maxl)
        return tau_out

    tau = tau_out

    @property
    def tau1(self):
        return self._tau1

    @property
    def tau2(self):
        return self._tau2

    def set_taus(self, tau1=None, tau2=None):
        self._tau1 = SO3Tau(tau1) if tau1 else None
        self._tau2 = SO3Tau(tau2) if tau2 else None

        if self._tau1 and self._tau2:
            if not self.tau1.channels or (self.tau1.channels != self.tau2.channels):
                raise ValueError('The number of fragments must be same for each part! '
                                 '{} {}'.format(self.tau1, self.tau2))


def cg_product(cg_dict, rep1, rep2, maxl=inf, minl=0, aggregate=False, ignore_check=False):
    """
    Explicit function to calculate the Clebsch-Gordan product.
    See the documentation for CGProduct for more information.

    rep1 : list of :obj:`torch.Tensors`
        First :obj:`SO3Vector` in the CG product
    rep2 : list of :obj:`torch.Tensors`
        First :obj:`SO3Vector` in the CG product
    minl : :obj:`int`, optional
        Minimum weight to include in CG Product
    maxl : :obj:`int`, optional
        Minimum weight to include in CG Product
    aggregate : :obj:`bool`, optional
        Apply an "aggregation" operation, or a pointwise convolution
        with a :obj:`SO3Vector` as a filter.
    cg_dict : :obj:`CGDict`, optional
        Specify a Clebsch-Gordan dictionary. If not specified, one will be
        generated automatically at runtime based upon maxl.
    ignore_check : :obj:`bool`
        Ignore SO3Vec initialization check. Necessary for current implementation
        of :obj:`spherical_harmonics`. Use with caution.
    """
    tau1 = SO3Tau.from_rep(rep1)
    tau2 = SO3Tau.from_rep(rep2)

    assert tau1.channels and (tau1.channels == tau2.channels), 'The number of fragments must be same for each part! {} {}'.format(tau1, tau2)

    ells1 = rep1.ells if isinstance(rep1, SO3Vec) else [(part.shape[-2] - 1)//2 for part in rep1]
    ells2 = rep2.ells if isinstance(rep2, SO3Vec) else [(part.shape[-2] - 1)//2 for part in rep2]

    L1 = max(ells1)
    L2 = max(ells2)

    if (cg_dict.maxl < maxl) or (cg_dict.maxl < L1) or (cg_dict.maxl < L2):
        raise ValueError('CG Dictionary maxl ({}) not sufficiently large for (maxl, L1, L2) = ({} {} {})'.format(cg_dict.maxl, maxl, L1, L2))
    assert(cg_dict.transpose), 'This operation uses transposed CG coefficients!'

    maxL = min(L1 + L2, maxl)

    new_rep = [[] for _ in range(maxL + 1)]

    for l1, part1 in zip(ells1, rep1):
        for l2, part2 in zip(ells2, rep2):
            lmin, lmax = max(abs(l1 - l2), minl), min(l1 + l2, maxL)
            if lmin > lmax:
                continue

            cg_mat = cg_dict[(l1, l2)][:(lmax+1)**2 - (lmin)**2, :]

            # Loop over atom irreps accumulating each.
            irrep_prod = complex_kron_product(part1, part2, aggregate=aggregate)
            cg_decomp = torch.matmul(cg_mat, irrep_prod)

            split = [2*l+1 for l in range(lmin, lmax+1)]
            cg_decomp = torch.split(cg_decomp, split, dim=-2)

            for idx, l in enumerate(range(lmin, lmax+1)):
                new_rep[l].append(cg_decomp[idx])

    new_rep = [torch.cat(part, dim=-3) for part in new_rep if len(part) > 0]

    # TODO: Rewrite so ignore_check not necessary
    return SO3Vec(new_rep, ignore_check=ignore_check)


def complex_kron_product(z1, z2, aggregate=False):
    """
    Take two complex matrix tensors z1 and z2, and take their tensor product.

    Parameters
    ----------
    z1 : :class:`torch.Tensor`
        Tensor of shape batch1 x M1 x N1 x 2.
        The last dimension is the complex dimension.
    z1 : :class:`torch.Tensor`
        Tensor of shape batch2 x M2 x N2 x 2.
    aggregate: :class:`bool`
        Apply aggregation/point-wise convolutional filter. Must have batch1 = B x A x A, batch2 = B x A

    Returns
    -------
    z1 : :class:`torch.Tensor`
        Tensor of shape batch x (M1 x M2) x (N1 x N2) x 2
    """
    s1 = z1.shape
    s2 = z2.shape
    assert(len(s1) >= 3), 'Must have batch dimension!'
    assert(len(s2) >= 3), 'Must have batch dimension!'

    b1, b2 = s1[:-3], s2[:-3]
    s1, s2 = s1[-3:], s2[-3:]
    if not aggregate:
        assert(b1 == b2), 'Batch sizes must be equal! {} {}'.format(b1, b2)
        b = b1
    else:
        if (len(b1) == 3) and (len(b2) == 2):
            assert(b1[0] == b2[0]), 'Batch sizes must be equal! {} {}'.format(b1, b2)
            assert(b1[2] == b2[1]), 'Neighborhood sizes must be equal! {} {}'.format(b1, b2)

            z2 = z2.unsqueeze(1)
            b2 = z2.shape[:-3]
            b = b1

            agg_sum_dim = 2

        elif (len(b1) == 2) and (len(b2) == 3):
            assert(b2[0] == b1[0]), 'Batch sizes must be equal! {} {}'.format(b1, b2)
            assert(b2[2] == b1[1]), 'Neighborhood sizes must be equal! {} {}'.format(b1, b2)

            z1 = z1.unsqueeze(1)
            b1 = z1.shape[:-3]
            b = b2

            agg_sum_dim = 2

        else:
            raise ValueError('Batch size error! {} {}'.format(b1, b2))

    # Treat the channel index like a "batch index".
    assert(s1[0] == s2[0]), 'Number of channels must match! {} {}'.format(s1[0], s2[0])

    s12 = b + (s1[0], s1[1]*s2[1], s1[2]*s2[2])

    s10 = b1 + (s1[0],) + torch.Size([s1[1], 1, s1[2], 1])
    s20 = b2 + (s1[0],) + torch.Size([1, s2[1], 1, s2[2]])

    z = (z1.view(s10) * z2.view(s20))
    z = z.contiguous().view(s12)

    if aggregate:
        # Aggregation is sum over aggregation sum dimension defined above
        z = z.sum(agg_sum_dim, keepdim=False)

    zrot = torch.tensor([[1, 0], [0, 1], [0, 1], [-1, 0]], dtype=z.dtype, device=z.device)
    z = torch.matmul(z, zrot)

    return z


class SphericalHarmonics(CGModule):
    r"""
    Calculate a list of spherical harmonics :math:`Y^\ell_m(\hat{\bf r})`
    for a :class:`torch.Tensor` of cartesian vectors :math:`{\bf r}`.

    This module subclasses :class:`CGModule`, and maintains similar functionality.

    Parameters
    ----------
    maxl : :class:`int`
        Calculate spherical harmonics from ``l=0, ..., maxl``.
    normalize : :class:`bool`, optional
        Normalize the cartesian vectors used to calculate the spherical harmonics.
    conj : :class:`bool`, optional
        Return the conjugate of the (conventionally defined) spherical harmonics.
    sh_norm : :class:`str`, optional
        Chose the normalization convention for the spherical harmonics.
        The options are:

        - 'qm': Quantum mechanical convention: :math:`\sum_m |Y^\ell_m|^2 = \frac{2\ell+1}{4\pi}`

        - 'unit': Quantum mechanical convention: :math:`\sum_m |Y^\ell_m|^2 = 1`
    cg_dict : :class:`CGDict`, optional
        Specify a Clebsch-Gordan Dictionary
    dtype : :class:`torch.torch.dtype`, optional
        Specify the dtype to initialize the :class:`CGDict`/:class:`CGModule` to
    device : :class:`torch.torch.device`, optional
        Specify the device to initialize the :class:`CGDict`/:class:`CGModule` to
    """
    def __init__(self, maxl, normalize=True, conj=False, sh_norm='unit',
                 cg_dict=None, dtype=None, device=None):

        self.normalize = normalize
        self.sh_norm = sh_norm
        self.conj = conj

        super().__init__(cg_dict=cg_dict, maxl=maxl, device=device, dtype=dtype)

    def forward(self, pos):
        r"""
        Calculate the Spherical Harmonics for a set of cartesian position vectors.

        Parameters
        ----------
        pos : :class:`torch.Tensor`
            Input tensor of cartesian vectors

        Returns
        -------
        sph_harms : :class:`list` of :class:`torch.Tensor`
            Output list of spherical harmonics from :math:`\ell=0` to :math:`\ell=maxl`
        """
        return spherical_harmonics(self.cg_dict, pos, self.maxl,
                                   self.normalize, self.conj, self.sh_norm)


class SphericalHarmonicsRel(CGModule):
    r"""
    Calculate a matrix of spherical harmonics

    .. math::
        \Upsilon_{ij} = Y^\ell_m(\hat{\bf r}_{ij})

    based upon the difference

    .. math::
        {\bf r}_{ij} = {\bf r}^{(1)}_i - {\bf r}^{(2)}_j.

    in two lists of cartesian vectors  :math:`{\bf r}^{(1)}_i`
    and :math:`{\bf r}^{(2)}_j`.


    This module subclasses :class:`CGModule`, and maintains similar functionality.

    Parameters
    ----------
    maxl : :class:`int`
        Calculate spherical harmonics from ``l=0, ..., maxl``.
    normalize : :class:`bool`, optional
        Normalize the cartesian vectors used to calculate the spherical harmonics.
    conj : :class:`bool`, optional
        Return the conjugate of the (conventionally defined) spherical harmonics.
    sh_norm : :class:`str`, optional
        Chose the normalization convention for the spherical harmonics.
        The options are:

        - 'qm': Quantum mechanical convention: :math:`\sum_m |Y^\ell_m|^2 = \frac{2\ell+1}{4\pi}`

        - 'unit': Quantum mechanical convention: :math:`\sum_m |Y^\ell_m|^2 = 1`
    cg_dict : :class:`CGDict` or None, optional
        Specify a Clebsch-Gordan Dictionary
    dtype : :class:`torch.torch.dtype`, optional
        Specify the dtype to initialize the :class:`CGDict`/:class:`CGModule` to
    device : :class:`torch.torch.device`, optional
        Specify the device to initialize the :class:`CGDict`/:class:`CGModule` to
    """
    def __init__(self, maxl, normalize=False, conj=False, sh_norm='unit',
                 cg_dict=None, dtype=None, device=None):

        self.normalize = normalize
        self.sh_norm = sh_norm
        self.conj = conj

        super().__init__(cg_dict=cg_dict, maxl=maxl, device=device, dtype=dtype)

    def forward(self, pos1, pos2):
        r"""
        Calculate the Spherical Harmonics for a matrix of differences of cartesian
        position vectors `pos1` and `pos2`.

        Note that `pos1` and `pos2` must agree in every dimension except for
        the second-to-last one.

        Parameters
        ----------
        pos1 : :class:`torch.Tensor`
            First tensor of cartesian vectors :math:`{\bf r}^{(1)}_i`.
        pos2 : :class:`torch.Tensor`
            Second tensor of cartesian vectors :math:`{\bf r}^{(2)}_j`.

        Returns
        -------
        sph_harms : :class:`list` of :class:`torch.Tensor`
            Output matrix of spherical harmonics from :math:`\ell=0` to :math:`\ell=maxl`
        """
        return spherical_harmonics_rel(self.cg_dict, pos1, pos2, self.maxl,
                                       self.normalize, self.conj, self.sh_norm)


def spherical_harmonics(cg_dict, pos, maxsh, normalize=True, conj=False, sh_norm='unit'):
    r"""
    Functional form of the Spherical Harmonics. See documentation of
    :class:`SphericalHarmonics` for details.
    """
    s = pos.shape[:-1]

    pos = pos.view(-1, 3)

    if normalize:
        norm = pos.norm(dim=-1, keepdim=True)
        mask = (norm > 0)
        # pos /= norm
        # pos[pos == inf] = 0
        pos = torch.where(mask, pos / norm, torch.zeros_like(pos))

    psi0 = torch.full(s + (1,), sqrt(1/(4*pi)), dtype=pos.dtype, device=pos.device)
    psi0 = torch.stack([psi0, torch.zeros_like(psi0)], -1)
    psi0 = psi0.view(-1, 1, 1, 2)

    sph_harms = [psi0]
    if maxsh >= 1:
        psi1 = pos_to_rep(pos, conj=conj)
        psi1 *= sqrt(3/(4*pi))
        sph_harms.append(psi1)

    if maxsh >= 2:
        new_psi = psi1
        for l in range(2, maxsh+1):
            new_psi = cg_product(cg_dict, [new_psi], [psi1], minl=0, maxl=l, ignore_check=True)[-1]
            # Use equation Y^{m1}_{l1} \otimes Y^{m2}_{l2} = \sqrt((2*l1+1)(2*l2+1)/4*\pi*(2*l3+1)) <l1 0 l2 0|l3 0> <l1 m1 l2 m2|l3 m3> Y^{m3}_{l3}
            # cg_coeff = CGcoeffs[1*(CGmaxL+1) + l-1][5*(l-1)+1, 3*(l-1)+1] # 5*l-4 = (l)^2 -(l-2)^2 + (l-1) + 1, notice indexing starts at l=2
            cg_coeff = cg_dict[(1, l-1)][5*(l-1)+1, 3*(l-1)+1]  # 5*l-4 = (l)^2 -(l-2)^2 + (l-1) + 1, notice indexing starts at l=2
            new_psi *= sqrt((4*pi*(2*l+1))/(3*(2*l-1))) / cg_coeff
            sph_harms.append(new_psi)
    sph_harms = [part.view(s + part.shape[1:]) for part in sph_harms]

    if sh_norm == 'qm':
        pass
    elif sh_norm == 'unit':
        sph_harms = [part*sqrt((4*pi)/(2*ell+1)) for ell, part in enumerate(sph_harms)]
    else:
        raise ValueError('Incorrect choice of spherial harmonic normalization!')

    return SO3Vec(sph_harms)


def spherical_harmonics_rel(cg_dict, pos1, pos2, maxsh, normalize=True, conj=False, sh_norm='unit'):
    r"""
    Functional form of the relative Spherical Harmonics. See documentation of
    :class:`SphericalHarmonicsRel` for details.
    """
    rel_pos = pos1.unsqueeze(-2) - pos2.unsqueeze(-3)
    rel_norms = rel_pos.norm(dim=-1, keepdim=True)

    rel_sph_harm = spherical_harmonics(cg_dict, rel_pos, maxsh, normalize=normalize,
                                       conj=conj, sh_norm=sh_norm)

    return rel_sph_harm, rel_norms.squeeze(-1)


def pos_to_rep(pos, conj=False):
    r"""
    Convert a tensor of cartesian position vectors to an l=1 spherical tensor.

    Parameters
    ----------
    pos : :class:`torch.Tensor`
        A set of input cartesian vectors. Can have arbitrary batch dimensions
         as long as the last dimension has length three, for x, y, z.
    conj : :class:`bool`, optional
        Return the complex conjugated representation.


    Returns
    -------
    psi1 : :class:`torch.Tensor`
        The input cartesian vectors converted to a l=1 spherical tensor.

    """
    pos_x, pos_y, pos_z = pos.unbind(-1)

    # Only the y coordinates get mapped to imaginary terms
    if conj:
        pos_m = torch.stack([pos_x, pos_y], -1)/sqrt(2.)
        pos_p = torch.stack([-pos_x, pos_y], -1)/sqrt(2.)
    else:
        pos_m = torch.stack([pos_x, -pos_y], -1)/sqrt(2.)
        pos_p = torch.stack([-pos_x, -pos_y], -1)/sqrt(2.)
    pos_0 = torch.stack([pos_z, torch.zeros_like(pos_z)], -1)

    psi1 = torch.stack([pos_m, pos_0, pos_p], dim=-2).unsqueeze(-3)

    return psi1


def rep_to_pos(rep):
    r"""
    Convert a tensor of l=1 spherical tensors to cartesian position vectors.

    Warning
    -------
    The input spherical tensor must satisfy :math:`F_{-m} = (-1)^m F_{m}^*`,
    so the output cartesian tensor is explicitly real. If this is not satisfied
    an error will be thrown.

    Parameters
    ----------
    rep : :class:`torch.Tensor`
        A set of input l=1 spherical tensors.
        Can have arbitrary batch dimensions as long
        as the last dimension has length three, for m = -1, 0, +1.

    Returns
    -------
    pos : :class:`torch.Tensor`
        The input l=1 spherical tensors converted to cartesian vectors.

    """
    rep_m, rep_0, rep_p = rep.unbind(-2)

    pos_x = (-rep_p + rep_m)/sqrt(2.)
    pos_y = (-rep_p - rep_m)/sqrt(2.)
    pos_z = rep_0

    imag_part = [pos_x[..., 1].abs().mean(), pos_y[..., 0].abs().mean(), pos_z[..., 1].abs().mean()]
    if (any([p > 1e-6 for p in imag_part])):
        raise ValueError('Imaginary part not zero! {}'.format(imag_part))

    pos = torch.stack([pos_x[..., 0], pos_y[..., 1], pos_z[..., 0]], dim=-1)

    return pos




#########  Weight mixing  ###########

def mix_zweight_zvec(weight, part, zdim=-1):
    """
    Apply the linear matrix in :obj:`SO3Weight` and a part of a :obj:`SO3Vec`.

    Parameters
    ----------
    weight : :obj:`torch.Tensor`
        A tensor of mixing weights to apply to `part`.
    part : :obj:`torch.Tensor`
        Part of :obj:`SO3Vec` to multiply by scalars.

    """
    weight_r, weight_i = weight.unbind(zdim)
    part_r, part_i = part.unbind(zdim)

    return torch.stack([weight_r@part_r - weight_i@part_i,
                        weight_i@part_r + weight_r@part_i], dim=zdim)


def mix_zweight_zscalar(weight, part, zdim=-1):
    """
    Apply the linear matrix in :obj:`SO3Weight` and a part of a :obj:`SO3Scalar`.

    Parameters
    ----------
    scalar : :obj:`torch.Tensor`
        A tensor of mixing weights to apply to `part`.
    part : :obj:`torch.Tensor`
        Part of :obj:`SO3Scalar` to multiply by scalars.

    """
    # Must permute first two dimensions
    weight_r, weight_i = weight.transpose(0, 1).unbind(zdim)
    part_r, part_i = part.unbind(zdim)

    # # Since the dimension to be mixed in part is the right-most,
    return torch.stack([part_r@weight_r - part_i@weight_i,
                        part_r@weight_i + part_i@weight_r], dim=zdim)


#########  Multiply  ###########

def mul_zscalar_zirrep(scalar, part, rdim=-2, zdim=-1):
    """
    Multiply the part of a :obj:`SO3Scalar` and a part of a :obj:`SO3Vec`.

    Parameters
    ----------
    scalar : :obj:`torch.Tensor`
        A tensor of scalars to apply to `part`.
    part : :obj:`torch.Tensor`
        Part of :obj:`SO3Vec` to multiply by scalars.

    """
    scalar_r, scalar_i = scalar.unsqueeze(rdim).unbind(zdim)
    part_r, part_i = part.unbind(zdim)

    return torch.stack([part_r*scalar_r - part_i*scalar_i, part_r*scalar_i + part_i*scalar_r], dim=zdim)


def mul_zscalar_zscalar(scalar1, scalar2, zdim=-1):
    """
    Complex multiply the part of a :obj:`SO3Scalar` and a part of a
    different :obj:`SO3Scalar`.

    Parameters
    ----------
    scalar1 : :obj:`torch.Tensor`
        First tensor of scalars to multiply.
    scalar2 : :obj:`torch.Tensor`
        Second tensor of scalars to multiply.
    zdim : :obj:`int`
        Dimension for which complex multiplication is defined.


    """
    scalar1_r, scalar1_i = scalar1.unbind(zdim)
    scalar2_r, scalar2_i = scalar2.unbind(zdim)

    return torch.stack([scalar1_r*scalar2_r - scalar1_i*scalar2_i,
                        scalar1_r*scalar2_i + scalar1_i*scalar2_r], dim=zdim)


import torch

#########  Weight mixing  ###########

def mix_zweight_zvec(weight, part, zdim=-1):
    """
    Apply the linear matrix in :obj:`SO3Weight` and a part of a :obj:`SO3Vec`.

    Parameters
    ----------
    weight : :obj:`torch.Tensor`
        A tensor of mixing weights to apply to `part`.
    part : :obj:`torch.Tensor`
        Part of :obj:`SO3Vec` to multiply by scalars.

    """
    weight_r, weight_i = weight.unbind(zdim)
    part_r, part_i = part.unbind(zdim)

    return torch.stack([weight_r@part_r - weight_i@part_i,
                        weight_i@part_r + weight_r@part_i], dim=zdim)


def mix_zweight_zscalar(weight, part, zdim=-1):
    """
    Apply the linear matrix in :obj:`SO3Weight` and a part of a :obj:`SO3Scalar`.

    Parameters
    ----------
    scalar : :obj:`torch.Tensor`
        A tensor of mixing weights to apply to `part`.
    part : :obj:`torch.Tensor`
        Part of :obj:`SO3Scalar` to multiply by scalars.

    """
    # Must permute first two dimensions
    weight_r, weight_i = weight.transpose(0, 1).unbind(zdim)
    part_r, part_i = part.unbind(zdim)

    # # Since the dimension to be mixed in part is the right-most,
    return torch.stack([part_r@weight_r - part_i@weight_i,
                        part_r@weight_i + part_i@weight_r], dim=zdim)


#########  Multiply  ###########

def mul_zscalar_zirrep(scalar, part, rdim=-2, zdim=-1):
    """
    Multiply the part of a :obj:`SO3Scalar` and a part of a :obj:`SO3Vec`.

    Parameters
    ----------
    scalar : :obj:`torch.Tensor`
        A tensor of scalars to apply to `part`.
    part : :obj:`torch.Tensor`
        Part of :obj:`SO3Vec` to multiply by scalars.

    """
    scalar_r, scalar_i = scalar.unsqueeze(rdim).unbind(zdim)
    part_r, part_i = part.unbind(zdim)

    return torch.stack([part_r*scalar_r - part_i*scalar_i, part_r*scalar_i + part_i*scalar_r], dim=zdim)


def mul_zscalar_zscalar(scalar1, scalar2, zdim=-1):
    """
    Complex multiply the part of a :obj:`SO3Scalar` and a part of a
    different :obj:`SO3Scalar`.

    Parameters
    ----------
    scalar1 : :obj:`torch.Tensor`
        First tensor of scalars to multiply.
    scalar2 : :obj:`torch.Tensor`
        Second tensor of scalars to multiply.
    zdim : :obj:`int`
        Dimension for which complex multiplication is defined.


    """
    scalar1_r, scalar1_i = scalar1.unbind(zdim)
    scalar2_r, scalar2_i = scalar2.unbind(zdim)

    return torch.stack([scalar1_r*scalar2_r - scalar1_i*scalar2_i,
                        scalar1_r*scalar2_i + scalar1_i*scalar2_r], dim=zdim)

# from cormorant.so3_lib import so3_wigner_d
#
# SO3WignerD = so3_wigner_d.SO3WignerD

# TODO: Update legacy code to use SO3Vec/SO3WignerD interfaces
# TODO: Convert to PyTorch objects to allow for GPU parallelism and autograd support

# Explicitly construct functions for the 3D cartesian rotation matrices


# def Rx(theta):
#     """
#     Rotation Matrix for rotations on the x axis.
#
#     Parameters
#     ----------
#     theta : double
#         Angle over which to rotate.
#
#     Returns
#     -------
#     Rmat : :obj:`torch.Tensor`
#         The associated rotation matrix.
#     """
#     return torch.tensor([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]], dtype=torch.double)


# def Ry(theta, device=None, dtype=None):
def Ry(theta):
    """
    Rotation Matrix for rotations on the y axis.

    Parameters
    ----------
    theta : double
        Angle over which to rotate.

    Returns
    -------
    Rmat : :obj:`torch.Tensor`
        The associated rotation matrix.
    """
    return torch.tensor([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]], dtype=torch.double)


def Rz(theta):
    """
    Rotation Matrix for rotations on the z axis. Syntax is the same as with Ry.
    """
    return torch.tensor([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=torch.double)


def EulerRot(alpha, beta, gamma):
    """
    Constructs a Rotation Matrix from Euler angles.

    Parameters
    ----------
    alpha : double
        First Euler angle
    beta : double
        Second Euler angle
    gamma : double
        Third Euler angle

    Returns
    -------
    Rmat : :obj:`torch.Tensor`
        The associated rotation matrix.
    """
    return Rz(alpha) @ Ry(beta) @ Rz(gamma)


def gen_rot(maxl, angles=None, device=None, dtype=None):
    """
    Generate a rotation matrix corresponding to a Cartesian and also a Wigner-D
    representation of a specific  rotation. If `angles` is :obj:`None`, will
    generate random rotation.

    Parameters
    ----------
    maxl : :obj:`int`
        Maximum weight to include in the Wigner D-matrix list
    angles : :obj:`list` of :obj:`float` or compatible, optional
        Three Euler angles (alpha, beta, gamma) to parametrize the rotation.
    device : :obj:`torch.device`, optional
        Device of the output tensor
    dtype : :obj:`torch.dtype`, optional
        Data dype of the output tensor

    Returns
    -------
    D : :obj:`list` of :obj:`torch.Tensor`
        List of Wigner D-matrices from `l=0` to `l=maxl`
    R : :obj:`torch.Tensor`
        Rotation matrix that will perform the same cartesian rotation
    angles : :obj:`tuple`
        Euler angles that defines the input rotation
    """
    # TODO : Output D as SO3WignerD
    if angles is None:
        alpha, beta, gamma = np.random.rand(3) * 2*np.pi
        beta = beta / 2
        angles = alpha, beta, gamma
    else:
        assert len(angles) == 3
        alpha, beta, gamma = angles
    D = WignerD_list(maxl, alpha, beta, gamma, device=device, dtype=dtype)
    # R = EulerRot(alpha, beta, gamma, device=device, dtype=dtype)
    R = EulerRot(alpha, beta, gamma).to(device=device, dtype=dtype)

    return D, R, angles


def rotate_cart_vec(R, vec):
    """ Rotate a Cartesian vector by a Euler rotation matrix. """
    return torch.einsum('ij,...j->...i', R, vec)  # Broadcast multiplication along last axis.


def rotate_part(D, z, dir='left'):
    """ Apply a WignerD matrix using complex broadcast matrix multiplication. """
    Dr, Di = D.unbind(-1)
    zr, zi = z.unbind(-1)

    if dir == 'left':
        matmul = lambda D, z: torch.einsum('ij,...kj->...ki', D, z)
    elif dir == 'right':
        matmul = lambda D, z: torch.einsum('ji,...kj->...ki', D, z)
    else:
        raise ValueError('Must apply Wigner rotation from dir=left/right! got dir={}'.format(dir))

    return torch.stack((matmul(Dr, zr) - matmul(Di, zi),
                        matmul(Di, zr) + matmul(Dr, zi)), -1)


def rotate_rep(D_list, rep, dir='left'):
    """ Apply a WignerD rotation part-wise to a representation. """
    ls = [(part.shape[-2]-1)//2 for part in rep]
    D_maxls = (D_list[-1].shape[-2]-1)//2
    assert((D_maxls >= max(ls))), 'Must have at least one D matrix for each rep! {} {}'.format(D_maxls, len(rep))

    D_list = [D_list[l] for l in ls]
    return [rotate_part(D, part, dir=dir) for (D, part) in zip(D_list, rep)]


def dagger(D):
    conj = torch.tensor([1, -1], dtype=D.dtype, device=D.device).view(1, 1, 2)
    D = (D*conj).permute((1, 0, 2))
    return D


def create_J(j):
    mrange = -np.arange(-j, j)
    jp_diag = np.sqrt((j+mrange)*(j-mrange+1))
    Jp = np.diag(jp_diag, k=1)
    Jm = np.diag(jp_diag, k=-1)

    # Jx = (Jp + Jm) / complex(2, 0)
    # Jy = -(Jp - Jm) / complex(0, 2)
    Jz = np.diag(-np.arange(-j, j+1))
    Id = np.eye(2*j+1)

    return Jp, Jm, Jz, Id


def create_Jy(j):
    mrange = -np.arange(-j, j)
    jp_diag = np.sqrt((j+mrange)*(j-mrange+1))
    Jp = np.diag(jp_diag, k=1)
    Jm = np.diag(jp_diag, k=-1)

    Jy = -(Jp - Jm) / complex(0, 2)

    return Jy


def create_Jx(j):
    mrange = -np.arange(-j, j)
    jp_diag = np.sqrt((j+mrange)*(j-mrange+1))
    Jp = np.diag(jp_diag, k=1)
    Jm = np.diag(jp_diag, k=-1)

    Jx = (Jp + Jm) / complex(2, 0)

    return Jx


def littled(j, beta):
    Jy = create_Jy(j)

    evals, evecs = np.linalg.eigh(Jy)
    evecsh = evecs.conj().T
    evals_exp = np.diag(np.exp(complex(0, -beta)*evals))

    d = np.matmul(np.matmul(evecs, evals_exp), evecsh)

    return d


def WignerD(j, alpha, beta, gamma, numpy_test=False, dtype=torch.float, device=torch.device('cpu')):
    """
    Calculates the Wigner D matrix for a given degree and Euler Angle.

    Parameters
    ----------
    j : int
        Degree of the representation.
    alpha : double
        First Euler angle
    beta : double
        Second Euler angle
    gamma : double
        Third Euler angle
    numpy_test : bool, optional
        ?????
    device : :obj:`torch.device`, optional
        Device of the output tensor
    dtype : :obj:`torch.dtype`, optional
        Data dype of the output tensor

    Returns
    -------
    D =


    """
    d = littled(j, beta)

    Jz = np.arange(-j, j+1)
    Jzl = np.expand_dims(Jz, 1)

    # np.multiply() broadcasts, so this isn't actually matrix multiplication, and 'left'/'right' are lies
    left = np.exp(complex(0, -alpha)*Jzl)
    right = np.exp(complex(0, -gamma)*Jz)

    D = left * d * right

    if not numpy_test:
        D = complex_from_numpy(D, dtype=dtype, device=device)

    return D


def WignerD_list(jmax, alpha, beta, gamma, numpy_test=False, dtype=torch.float, device=torch.device('cpu')):
    """

    """
    return [WignerD(j, alpha, beta, gamma, numpy_test=numpy_test, dtype=dtype, device=device) for j in range(jmax+1)]


def complex_from_numpy(z, dtype=torch.float, device=torch.device('cpu')):
    """ Take a numpy array and output a complex array of the same size. """
    zr = torch.from_numpy(z.real).to(dtype=dtype, device=device)
    zi = torch.from_numpy(z.imag).to(dtype=dtype, device=device)

    return torch.stack((zr, zi), -1)

SO3Tensor = so3_tensor.SO3Tensor
SO3Tau = so3_tau.SO3Tau

class SO3Scalar(SO3Tensor):
    """
    Core class for creating and tracking SO(3) Scalars that
    are used to part-wise multiply :obj:`SO3Vec`.

    At the core of each :obj:`SO3Scalar` is a list of :obj:`torch.Tensors` with
    shape `(B, C, 2)`, where:

    * `B` is some number of batch dimensions.
    * `C` is the channels/multiplicity (tau) of each irrep.
    * `2` corresponds to the real/imaginary parts of the complex dimension.

    Parameters
    ----------

    data : List of of `torch.Tensor` with appropriate shape
        Input of a SO(3) Scalar.
    """

    @property
    def bdim(self):
        return slice(0, -2)

    @property
    def cdim(self):
        return -2

    @property
    def rdim(self):
        return None

    @property
    def zdim(self):
        return -1

    @staticmethod
    def _get_shape(batch, weight, channels):
        return tuple(batch) + (channels, 2)

    def check_data(self, data):
        if any(part.numel() == 0 for part in data):
            raise NotImplementedError('Non-zero parts in SO3Scalars not currrently enabled!')

        shapes = [part.shape[self.bdim] for part in data]
        if len(set(shapes)) > 1:
            raise ValueError('Batch dimensions are not identical!')

        if any(part.shape[self.zdim] != 2 for part in data):
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, shapes[self.zdim]))


class SO3Tau():
    """
    Class for keeping track of multiplicity (number of channels) of a SO(3)
    vector.

    Parameters
    ----------
    tau : :class:`list` of :class:`int`, :class:`SO3Tau`, or class with `.tau` property.
        Multiplicity of an SO(3) vector.
    """
    def __init__(self, tau):
        if type(tau) in [list, tuple]:
            if not all(type(t) == int for t in tau):
                raise ValueError('Input must be list or tuple of ints! {} {}'.format(type(tau), [type(t) for t in tau]))
        else:
            try:
                tau = tau.tau
            except AttributeError:
                raise AttributeError('Input is of type %s does not have a defined .tau property!' % type(tau))

        self._tau = tuple(tau)

    @property
    def maxl(self):
        return len(self._tau) - 1

    def keys(self):
        return range(len(self))

    def values(self):
        return self._tau

    def items(self):
        return zip(self._tau, range(len(self)))

    def __iter__(self):
        """
        Loop over SO3Tau
        """
        for t in self._tau:
            yield t

    def __getitem__(self, idx):
        """
        Get item of SO3Tau.
        """
        if type(idx) is slice:
            return SO3Tau(self._tau[idx])
        else:
            return self._tau[idx]

    def __len__(self):
        """
        Length of SO3Tau
        """
        return len(self._tau)

    def __setitem__(self, idx, val):
        """
        Set index of SO3Tau
        """
        self._tau[idx] = val

    def __eq__(self, other):
        """
        Check equality of two :math:`SO3Tau` objects or :math:`SO3Tau` and
        a list.
        """
        self_tau = tuple(self._tau)
        other_tau = tuple(other)

        return self_tau == other_tau

    @staticmethod
    def cat(tau_list):
        """
        Return the multiplicity :class:`SO3Tau` corresponding to the concatenation
        (direct sum) of a list of objects of type :class:`SO3Tensor`.
        
        Parameters
        ----------
        tau_list : :class:`list` of :class:`SO3Tau` or :class:`list` of :class:`int`s
            List of multiplicites of input :class:`SO3Tensor`

        Return
        ------

        tau : :class:`SO3Tau`
            Output tau of direct sum of input :class:`SO3Tensor`.
        """
        return SO3Tau([sum(taus) for taus in zip_longest(*tau_list, fillvalue=0)])


    def __and__(self, other):
        return SO3Tau.cat([self, other])

    def __rand__(self, other):
        return SO3Tau.cat([self, other])

    def __str__(self):
        return str(list(self._tau))

    __repr__ = __str__

    def __add__(self, other):
        return SO3Tau(list(self) + list(other))

    def __radd__(self, other):
        """
        Reverse add, includes type checker to deal with sum([])
        """
        if type(other) is int:
            return self
        return SO3Tau(list(other) + list(self))

    @staticmethod
    def from_rep(rep):
        """
        Construct SO3Tau object from an SO3Vector representation.

        Parameters
        ----------
        rep : :obj:`SO3Tensor` :obj:`list` of :obj:`torch.Tensors`
            Input representation.

        """
        
        if rep is None:
            return SO3Tau([])

        if isinstance(rep, SO3Tensor):
            return rep.tau

        if torch.is_tensor(rep):
            raise ValueError('Input not compatible with SO3Tensor')
        elif type(rep) in [list, tuple] and any(type(irrep) != torch.Tensor for irrep in rep):
            raise ValueError('Input not compatible with SO3Tensor')

        ells = [(irrep[0].shape[-2] - 1) // 2 for irrep in rep]

        minl, maxl = ells[0], ells[-1]

        assert ells == list(range(minl, maxl+1)), 'Rep must be continuous from minl to maxl'

        tau = [irrep.shape[-3] for irrep in rep]

        return SO3Tau(tau)

    @property
    def tau(self):
        return self._tau

    @property
    def channels(self):
        channels = set(self._tau)
        if len(channels) == 1:
            return channels.pop()
        else:
            return None


SO3Tau = so3_tau.SO3Tau


class SO3Tensor(ABC):
    """
    Core class for creating and tracking SO3 Vectors (aka SO3 representations).

    Parameters
    ----------
    data : iterable of of `torch.Tensor` with appropriate shape
        Input of a SO(3) vector.
    """
    def __init__(self, data, ignore_check=False):
        if isinstance(data, type(self)):
            data = data.data

        if not ignore_check:
            self.check_data(data)

        self._data = data

    @abstractmethod
    def check_data(self, data):
        """
        Implement a data checking method.
        """
        pass

    @property
    @abstractmethod
    def bdim(self):
        """
        Define the batch dimension for each part.
        """
        pass

    @property
    @abstractmethod
    def cdim(self):
        """
        Define the tau (channels) dimension for each part.
        """
        pass

    @property
    @abstractmethod
    def rdim(self):
        """
        Define the representation (2*l+1) dimension for each part. Should be None
        if it is not applicable for this type of SO3Tensor.
        """
        pass

    @property
    @abstractmethod
    def zdim(self):
        """
        Define the complex dimension for each part.
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_shape(batch, weight, channels):
        """
        Generate the shape of part based upon a batch size, multiplicity/number
        of channels, and weight.
        """
        pass

    def __len__(self):
        """
        Length of SO3Vec.
        """
        return len(self._data)

    @property
    def maxl(self):
        """
        Maximum weight (maxl) of SO3 object.

        Returns
        -------
        int
        """
        return len(self._data) - 1

    def truncate(self, maxl):
        """
        Update the maximum weight (`maxl`) by truncating parts of the
        :class:`SO3Tensor` if they correspond to weights greater than `maxl`.

        Parameters
        ----------
        maxl : :obj:`int`
            Maximum weight to truncate the representation to.

        Returns
        -------
        :class:`SO3Tensor` subclass
            Truncated :class:`SO3Tensor`
        """
        return self[:maxl+1]

    @property
    def tau(self):
        """
        Multiplicity of each weight if SO3 object.

        Returns
        -------
        :obj:`SO3Tau`
        """
        return SO3Tau([part.shape[self.cdim] for part in self])

    @property
    def bshape(self):
        """
        Get a list of shapes of each :obj:`torch.Tensor`
        """
        bshapes = [p.shape[self.bdim] for p in self]

        if len(set(bshapes)) != 1:
            raise ValueError('Every part must have the same shape! {}'.format(bshapes))

        return bshapes

    @property
    def shapes(self):
        """
        Get a list of shapes of each :obj:`torch.Tensor`
        """
        return [p.shape for p in self]

    @property
    def channels(self):
        """
        Constructs :obj:`SO3Tau`, and then gets the corresponding `SO3Tau.channels`
        method.
        """
        return self.tau.channels

    @property
    def device(self):
        if any(self._data[0].device != part.device for part in self._data):
            raise ValueError('Not all parts on same device!')

        return self._data[0].device

    @property
    def dtype(self):
        if any(self._data[0].dtype != part.dtype for part in self._data):
            raise ValueError('Not all parts using same data type!')

        return self._data[0].dtype

    def keys(self):
        return range(len(self))

    def values(self):
        return iter(self._data)

    def items(self):
        return zip(range(len(self)), self._data)

    def __iter__(self):
        """
        Loop over SO3Vec
        """
        for t in self._data:
            yield t

    def __getitem__(self, idx):
        """
        Get item of SO3Vec.
        """
        if type(idx) is slice:
            return self.__class__(self._data[idx])
        else:
            return self._data[idx]

    def __setitem__(self, idx, val):
        """
        Set index of SO3Vec.
        """
        self._data[idx] = val

    def __eq__(self, other):
        """
        Check equality of two :math:`SO3Vec` compatible objects.
        """
        if len(self) != len(other):
            return False
        return all((part1 == part2).all() for part1, part2 in zip(self, other))

    @staticmethod
    def allclose(rep1, rep2, **kwargs):
        """
        Check equality of two :obj:`SO3Tensor` compatible objects.
        """
        if len(rep1) != len(rep2):
            raise ValueError('')
        return all(torch.allclose(part1, part2, **kwargs) for part1, part2 in zip(rep1, rep2))

    def __and__(self, other):
        return self.cat([self, other])

    def __rand__(self, other):
        return self.cat([other, self])

    def __str__(self):
        return str(list(self._data))

    __datar__ = __str__

    @classmethod
    def requires_grad(cls):
        return cls([t.requires_grad() for t in cls._data])

    def requires_grad_(self, requires_grad=True):
        self._data = [t.requires_grad_(requires_grad) for t in self._data]
        return self

    def to(self, *args, **kwargs):
        self._data = [t.to(*args, **kwargs) for t in self._data]
        return self

    def cpu(self):
        self._data = [t.cpu() for t in self._data]
        return self

    def cuda(self, **kwargs):
        self._data = [t.cuda(**kwargs) for t in self._data]
        return self

    def long(self):
        self._data = [t.long() for t in self._data]
        return self

    def byte(self):
        self._data = [t.byte() for t in self._data]
        return self

    def bool(self):
        self._data = [t.bool() for t in self._data]
        return self

    def half(self):
        self._data = [t.half() for t in self._data]
        return self

    def float(self):
        self._data = [t.float() for t in self._data]
        return self

    def double(self):
        self._data = [t.double() for t in self._data]
        return self

    def clone(self):
        return type(self)([t.clone() for t in self])

    def detach(self):
        return type(self)([t.detach() for t in self])

    @property
    def data(self):
        return self._data

    @property
    def grad(self):
        return type(self)([t.grad for t in self])

    def add(self, other):
        return so3_torch.add(self, other)

    def __add__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        return so3_torch.add(self, other)

    __radd__ = __add__

    def sub(self, other):
        return so3_torch.sub(self, other)

    def __sub__(self, other):
        """
        Subtract element wise `torch.Tensors`
        """
        return so3_torch.sub(self, other)

    __rsub__ = __sub__

    def mul(self, other):
        return so3_torch.mul(self, other)

    def complex_mul(self, other):
        return so3_torch.mul(self, other)

    def __mul__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        return so3_torch.mul(self, other)

    __rmul__ = __mul__

    def div(self, other):
        return so3_torch.div(self, other)

    def __truediv__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        return so3_torch.div(self, other)

    __rtruediv__ = __truediv__

    def abs(self):
        """
        Calculate the element-wise absolute value of the :class:`torch.SO3Tensor`.

        Warning
        -------
        Will break covariance!
        """

        return type(self)([part.abs() for part in self])

    __abs__ = abs

    def max(self):
        """
        Returns a list of maximum values of each part in the
        :class:`torch.SO3Tensor`.
        """

        return [part.max() for part in self]

    def min(self):
        """
        Returns a list of minimum values of each part in the
        :class:`torch.SO3Tensor`.
        """

        return [part.min() for part in self]

    @classmethod
    def rand(cls, batch, tau, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`SO3Vec`.
        """

        shapes = [cls._get_shape(batch, l, t) for l, t in enumerate(tau)]

        return cls([torch.rand(shape, device=device, dtype=dtype,
                               requires_grad=requires_grad) for shape in shapes])

    @classmethod
    def randn(cls, tau, batch, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`SO3Vec`.
        """

        shapes = [cls._get_shape(batch, l, t) for l, t in enumerate(tau)]

        return cls([torch.randn(shape, device=device, dtype=dtype,
                                requires_grad=requires_grad) for shape in shapes])

    @classmethod
    def zeros(cls, tau, batch, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`SO3Vec`.
        """

        shapes = [cls._get_shape(batch, l, t) for l, t in enumerate(tau)]

        return cls([torch.zeros(shape, device=device, dtype=dtype,
                                requires_grad=requires_grad) for shape in shapes])

    @classmethod
    def ones(cls, tau, batch, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`SO3Vec`.
        """

        shapes = [cls._get_shape(batch, l, t) for l, t in enumerate(tau)]

        return cls([torch.ones(shape, device=device, dtype=dtype,
                               requires_grad=requires_grad) for shape in shapes])


SO3Tau = so3_tau.SO3Tau
SO3Tensor = so3_tensor.SO3Tensor
SO3Vec = so3_vec.SO3Vec
SO3Scalar = so3_scalar.SO3Scalar
SO3Weight = so3_weight.SO3Weight
SO3WignerD = so3_wigner_d.SO3WignerD


def _check_maxl(val1, val2):
    if len(val1) != len(val2):
        raise ValueError('Two SO3Tensor subclasses have different maxl values '
                         '({} {})!'.format(len(val1)-1, len(val2)-1))


def _check_mult_compatible(val1, val2):
    """
    Function to check that two SO3Tensors are compatible with regards
    to a specific binary operation.
    """
    val1_has_rdim = (val1.rdim is not None)
    val2_has_rdim = (val2.rdim is not None)

    if val1_has_rdim and val2_has_rdim:
        # raise ValueError('Cannot multiply two SO3Vecs together!')
        warnings.warn("Both Inputs have representation dimensions. "
                      "Multiplying them together may break covariance.",
                      RuntimeWarning)


def _dispatch_op(op, val1, val2):
    """
    Used to dispatch a binary operator where at least one of the two inputs is a
    SO3Tensor.
    """

    # Hack to make SO3Vec/SO3Scalar multiplication work
    # TODO: Figure out better way of doing this?
    if isinstance(val1, SO3Scalar) and isinstance(val2, SO3Vec):
        _check_maxl(val1, val2)
        applied_op = [op(part1.unsqueeze(val2.rdim), part2)
                      for part1, part2 in zip(val1, val2)]
        output_class = SO3Vec
    elif isinstance(val1, SO3Vec) and isinstance(val2, SO3Scalar):
        _check_maxl(val1, val2)
        applied_op = [op(part1, part2.unsqueeze(val1.rdim))
                      for part1, part2 in zip(val1, val2)]
        output_class = SO3Vec
    # Both va1 and val2 are other instances of SO3Tensor
    elif isinstance(val1, SO3Tensor) and isinstance(val2, SO3Tensor):
        _check_maxl(val1, val2)
        applied_op = [op(part1, part2) for part1, part2 in zip(val1, val2)]
        output_class = type(val2)
    # Multiply val1 with a list/tuple
    elif isinstance(val1, SO3Tensor) and type(val2) in [list, tuple]:
        _check_maxl(val1, val2)
        applied_op = [op(part1, part2) for part1, part2 in zip(val1, val2)]
        output_class = type(val1)
    # Multiply val1 with something else
    elif isinstance(val1, SO3Tensor) and not isinstance(val2, SO3Tensor):
        applied_op = [op(val2, part1) for part1 in val1]
        output_class =  type(val1)
    # Multiply val2 with a list/tuple
    elif not isinstance(val1, SO3Tensor) and type(val1) in [list, tuple]:
        _check_maxl(val1, val2)
        applied_op = [op(part1, part2) for part1, part2 in zip(val1, val2)]
        output_class = type(val1)
    # Multiply val2 with something else
    elif not isinstance(val1, SO3Tensor) and isinstance(val2, SO3Tensor):
        applied_op = [op(val1, part2) for part2 in val2]
        output_class = type(val2)
    else:
        raise ValueError('Neither class inherits from SO3Tensor!')

    return output_class(applied_op)


def _dispatch_mul(val1, val2):
    """
    Used to dispatch a binary operator where at least one of the two inputs is a
    SO3Tensor.
    """

    # Hack to make SO3Vec/SO3Scalar multiplication work
    # TODO: Figure out better way of doing this?
    if isinstance(val1, SO3Scalar) and isinstance(val2, SO3Vec):
        _check_maxl(val1, val2)
        applied_op = [mul_zscalar_zirrep(part1, part2, rdim=val2.rdim)
                      for part1, part2 in zip(val1, val2)]
        output_class = SO3Vec
    elif isinstance(val1, SO3Vec) and isinstance(val2, SO3Scalar):
        _check_maxl(val1, val2)
        applied_op = [mul_zscalar_zirrep(part2, part1, rdim=val1.rdim)
                      for part1, part2 in zip(val1, val2)]
        output_class = SO3Vec
    elif isinstance(val1, SO3Scalar) and isinstance(val2, SO3Scalar):
        _check_maxl(val1, val2)
        applied_op = [mul_zscalar_zscalar(part1, part2)
                      for part1, part2 in zip(val1, val2)]
        output_class = SO3Scalar
    # Both va1 and val2 are other instances of SO3Tensor
    elif isinstance(val1, SO3Tensor) and isinstance(val2, SO3Tensor):
        _check_maxl(val1, val2)
        _check_mult_compatible(val1, val2)
        applied_op = [mul_zscalar_zscalar(part1, part2)
                      for part1, part2 in zip(val1, val2)]
        output_class = type(val2)
    # Multiply val1 with a list/tuple
    elif isinstance(val1, SO3Tensor) and type(val2) in [list, tuple]:
        _check_maxl(val1, val2)
        applied_op = [torch.mul(part1, part2) for part1, part2 in zip(val1, val2)]
        output_class = type(val1)
    # Multiply val1 with something else
    elif isinstance(val1, SO3Tensor) and not isinstance(val2, SO3Tensor):
        applied_op = [torch.mul(val2, part1) for part1 in val1]
        output_class = type(val1)
    # Multiply val2 with a list/tuple
    elif not isinstance(val1, SO3Tensor) and type(val1) in [list, tuple]:
        _check_maxl(val1, val2)
        applied_op = [torch.mul(part1, part2) for part1, part2 in zip(val1, val2)]
        output_class = type(val1)
    # Multiply val2 with something else
    elif not isinstance(val1, SO3Tensor) and isinstance(val2, SO3Tensor):
        applied_op = [torch.mul(val1, part2) for part2 in val2]
        output_class = type(val2)
    else:
        raise ValueError('Neither class inherits from SO3Tensor!')

    return output_class(applied_op)


def mul(val1, val2):
    return _dispatch_mul(val1, val2)


def add(val1, val2):
    return _dispatch_op(torch.add, val1, val2)


def sub(val1, val2):
    return _dispatch_op(torch.sub, val1, val2)


def div(val1, val2):
    raise NotImplementedError('Complex Division has not been implemented yet')
    # return __dispatch_divtype(torch.div, val1, val2)


def cat(reps_list):
    """
    Concatenate (direct sum) a :obj:`list` of :obj:`SO3Tensor` representations.

    Parameters
    ----------
    reps_list : :obj:`list` of :obj:`SO3Tensor`

    Return
    ------
    rep_cat : :obj:`SO3Tensor`
        Direct sum of all :obj:`SO3Tensor` in `reps_list`
    """
    reps_cat = [list(filter(lambda x: x is not None, reps)) for reps in zip_longest(*reps_list, fillvalue=None)]
    reps_cat = [torch.cat(reps, dim=reps_list[0].cdim) for reps in reps_cat]

    return reps_list[0].__class__(reps_cat)

def mix(weights, rep):
    """
    Linearly mix representation.

    Parameters
    ----------
    rep : :obj:`SO3Vec` or compatible
    weights : :obj:`SO3Weights` or compatible

    Return
    ------
    :obj:`SO3Vec`
        Mixed direct sum of all :obj:`SO3Vec` in `reps_list`
    """
    if len(rep) != len(weights):
        raise ValueError('Must have one mixing weight for each part of SO3Vec!')

    if isinstance(rep, SO3Vec):
        rep_mix = SO3Vec([mix_zweight_zvec(weight, part) for weight, part in zip(weights, rep)])
    elif isinstance(rep, SO3Scalar):
        rep_mix = SO3Scalar([mix_zweight_zscalar(weight, part) for weight, part in zip(weights, rep)])
    elif isinstance(rep, SO3Weight):
        rep_mix = SO3Weight([mix_zweight_zvec(weight, part) for weight, part in zip(weights, rep)])
    elif isinstance(rep, SO3Tensor):
        raise NotImplementedError('Mixing for object {} not yet implemented!'.format(type(rep)))
    else:
        raise ValueError('Mixing only implemented for SO3Tensor subclasses!')

    return rep_mix


def cat_mix(weights, reps_list):
    """
    First concatenate (direct sum) and then linearly mix a :obj:`list` of
    :obj:`SO3Vec` objects with :obj:`SO3Weights` weights.

    Parameters
    ----------
    reps_list : :obj:`list` of :obj:`SO3Vec` or compatible
    weights : :obj:`SO3Weights` or compatible

    Return
    ------
    :obj:`SO3Vec`
        Mixed direct sum of all :obj:`SO3Vec` in `reps_list`
    """

    return mix(weights, cat(reps_list))


def apply_wigner(wigner_d, rep, dir='left'):
    """
    Apply a Wigner-D rotation to a :obj:`SO3Vec` representation
    """
    return SO3Vec(rot.rotate_rep(wigner_d, rep, dir=dir))


SO3Tau = so3_tau.SO3Tau
SO3Tensor = so3_tensor.SO3Tensor
SO3Scalar = so3_scalar.SO3Scalar
SO3WignerD = so3_wigner_d.SO3WignerD

class SO3Vec(SO3Tensor):
    """
    Core class for creating and tracking SO3 Vectors (aka SO3 representations).

    At the core of each :obj:`SO3Vec` is a list of :obj:`torch.Tensors` with
    shape `(B, C, 2*l+1, 2)`, where:

    * `B` is some number of batch dimensions.
    * `C` is the channels/multiplicity (tau) of each irrep.
    * `2*l+1` is the size of an irrep of weight `l`.
    * `2` corresponds to the real/imaginary parts of the complex dimension.

    Parameters
    ----------

    data : iterable of of `torch.Tensor` with appropriate shape
        Input of a SO(3) vector.
    """

    @property
    def bdim(self):
        return slice(0, -3)

    @property
    def cdim(self):
        return -3

    @property
    def rdim(self):
        return -2

    @property
    def zdim(self):
        return -1

    @property
    def ells(self):
        return [(shape[self.rdim] - 1)//2 for shape in self.shapes]

    @staticmethod
    def _get_shape(batch, l, channels):
        return tuple(batch) + (channels, 2*l+1, 2)

    def check_data(self, data):
        if any(part.numel() == 0 for part in data):
            raise NotImplementedError('Non-zero parts in SO3Vec not currrently enabled!')

        bdims = set(part.shape[self.bdim] for part in data)
        if len(bdims) > 1:
            raise ValueError('All parts (torch.Tensors) must have same number of'
                             'batch  dimensions! {}'.format(part.shape[self.bdim] for part in data))

        shapes = [part.shape for part in data]

        cdims = [shape[self.cdim] for shape in shapes]
        rdims = [shape[self.rdim] for shape in shapes]
        zdims = [shape[self.zdim] for shape in shapes]

        if not all([rdim == 2*l+1 for l, rdim in enumerate(rdims)]):
            raise ValueError('Irrep dimension (dim={}) of each tensor should have shape 2*l+1! Found: {}'.format(self.rdim, list(enumerate(rdims))))

        if not all([zdim == 2 for zdim in zdims]):
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, zdims))


    def apply_wigner(self, wigner_d, dir='left'):
        """
        Apply a WignerD matrix to `self`

        Parameters
        ----------
        wigner_d : :class:`SO3WignerD`
            The Wigner D matrix rotation to apply to `self`
        dir : :obj:`str`
            The direction to apply the Wigner-D matrices. Options are left/right.

        Returns
        -------
        :class:`SO3Vec`
            The current :class:`SO3Vec` rotated by :class:`SO3Vec`
        """

        return so3_torch.apply_wigner(wigner_d, self, dir=dir)

class SO3Weight(SO3Tensor):
    """
    Core class for creating and tracking SO(3) Weights that
    are used to part-wise mix a :obj:`SO3Vec`.

    At the core of each :obj:`SO3Weight` is a list of :obj:`torch.Tensors` with
    shape `(C_{out}, C_{in}, 2)`, where:

    * `C_{in}` is the channels/multiplicity (tau) of the input :obj:`SO3Vec`.
    * `C_{out}` is the channels/multiplicity (tau) of the output :obj:`SO3Vec`.
    * `2` corresponds to the real/imaginary parts of the complex dimension.

    Parameters
    ----------

    data : List of of `torch.Tensor` with appropriate shape
        Input of a SO(3) Weight object.
    """

    @property
    def bdim(self):
        return None

    @property
    def cdim(self):
        return None

    @property
    def rdim(self):
        return None

    @property
    def zdim(self):
        return 2

    @staticmethod
    def _get_shape(batch, t_out, t_in):
        return (t_out, t_in, 2)

    @property
    def tau_in(self):
        return SO3Tau([part.shape[1] for part in self])

    @property
    def tau_out(self):
        return SO3Tau([part.shape[0] for part in self])

    tau = tau_out

    def check_data(self, data):
        if any(part.numel() == 0 for part in data):
            raise NotImplementedError('Non-zero parts in SO3Weights not currrently enabled!')

        shapes = set(part.shape for part in data)
        shapes = shapes.pop()

        if not shapes[self.zdim] == 2:
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, shapes[self.zdim]))

    def as_parameter(self):
        """
        Return the weight as a :obj:`ParameterList` of :obj:`Parameter` so
        the weights can be added as parameters to a :obj:`torch.Module`.
        """
        return ParameterList([Parameter(weight) for weight in self._data])

    @staticmethod
    def rand(tau_in, tau_out, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`SO3Weight`.
        """

        shapes = [(t2, t1, 2) for t1, t2 in zip(tau_in, tau_out)]

        return SO3Weight([torch.rand(shape, device=device, dtype=dtype,
                          requires_grad=requires_grad) for shape in shapes])

    @staticmethod
    def randn(tau_in, tau_out, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random-normal :obj:`SO3Weight`.
        """

        shapes = [(t2, t1, 2) for t1, t2 in zip(tau_in, tau_out)]

        return SO3Weight([torch.randn(shape, device=device, dtype=dtype,
                          requires_grad=requires_grad) for shape in shapes])

    @staticmethod
    def zeros(tau_in, tau_out, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`SO3Weight`.
        """

        shapes = [(t2, t1, 2) for t1, t2 in zip(tau_in, tau_out)]

        return SO3Weight([torch.randn(shape, device=device, dtype=dtype,
                          requires_grad=requires_grad) for shape in shapes])

    @staticmethod
    def zeros(tau_in, tau_out, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new all-zeros :obj:`SO3Weight`.
        """

        shapes = [(t2, t1, 2) for t1, t2 in zip(tau_in, tau_out)]

        return SO3Weight([torch.zeros(shape, device=device, dtype=dtype,
                          requires_grad=requires_grad) for shape in shapes])

    @staticmethod
    def ones(tau_in, tau_out, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new all-ones :obj:`SO3Weight`.
        """

        shapes = [(t2, t1, 2) for t1, t2 in zip(tau_in, tau_out)]

        return SO3Weight([torch.ones(shape, device=device, dtype=dtype,
                          requires_grad=requires_grad) for shape in shapes])
