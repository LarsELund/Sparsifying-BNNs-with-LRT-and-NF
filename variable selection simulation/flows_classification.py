
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parameter_init(low, high, size):
    random_init = (low - high) * torch.rand(size, device=DEVICE) + high
    return random_init


class PropagateFlow(nn.Module):
    def __init__(self, transform, dim, num_transforms):
        super().__init__()
        if transform == 'Planar':
            self.transforms = nn.ModuleList([PlanarTransform(dim) for i in range(num_transforms)])
        elif transform == 'Radial':
            self.transforms = nn.ModuleList([RadialTransform(dim) for i in range(num_transforms)])
        elif transform == 'Sylvester':
            self.transforms = nn.ModuleList([SylvesterTransform(dim) for i in range(num_transforms)])
        elif transform == 'Householder':
            self.transforms = nn.ModuleList([HouseholderTransform(dim) for i in range(num_transforms)])

        elif transform == 'RNVP':
            self.transforms = nn.ModuleList([RNVP(dim) for i in range(num_transforms)])
        elif transform == 'IAF':
            self.transforms = nn.ModuleList([IAF(dim) for i in range(num_transforms)])

        elif transform == 'MNF':
            self.transforms = nn.ModuleList([MNF(dim) for i in range(num_transforms)])
        elif transform == 'mixed':
            self.transforms = nn.ModuleList([HouseholderTransform(dim), PlanarTransform(dim),
                                             HouseholderTransform(dim), PlanarTransform(dim),
                                             HouseholderTransform(dim), PlanarTransform(dim),
                                             HouseholderTransform(dim), PlanarTransform(dim),
                                             HouseholderTransform(dim), PlanarTransform(dim),
                                             ])

        else:
            print('Transform not implemented')

    def forward(self, z):
        logdet = 0
        for f in self.transforms:
            z = f(z)
            logdet += f.log_det()
        return z, logdet


class RadialTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.z_0 = nn.Parameter(parameter_init(-0.1, 0.1, dim))
        self.log_alpha = nn.Parameter(parameter_init(-4, 5, 1))
        self.beta = nn.Parameter(parameter_init(-0.1, 0.1, 1))
        self.d = dim
        self.softplus = nn.Softplus()

    def forward(self, z):
        alpha = self.softplus(self.log_alpha)
        diff = z - self.z_0

        r = torch.norm(diff, dim=list(range(1, self.z_0.dim())))
        self.H1 = self.beta / (alpha + r)
        self.H2 = - self.beta * r * (alpha + r) ** (-2)
        z_new = z + self.H1 + self.H2
        return z_new

    def log_det(self):
        logdet = (self.d - 1) * torch.log(1 + self.H1) + torch.log(1 + self.H1 + self.H2)
        return logdet


class PlanarTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # flow parameters
        self.u = nn.Parameter(parameter_init(-0.01, 0.01, dim))
        self.w = nn.Parameter(parameter_init(-0.01, 0.01, dim))
        self.bias = nn.Parameter(parameter_init(-0.01, 0.01, 1))

    def h(self, x):  # tanh activation function
        return torch.tanh(x)

    def h_derivative(self, x):
        return 1 - torch.tanh(x) ** 2

    def forward(self, z):
        inner = torch.dot(self.w, z) + self.bias
        z_new = z + self.u * (self.h(inner))
        self.psi = self.h_derivative(inner) * self.w

        return z_new

    def log_det(self):
        return torch.log(torch.abs(1 + torch.dot(self.u, self.psi)))


class SylvesterTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.M = 5
        self.A = nn.Parameter(parameter_init(-0.01, 0.01, (dim, self.M)))
        self.B = nn.Parameter(parameter_init(-0.01, 0.01, (self.M, dim)))
        self.b = nn.Parameter(parameter_init(-0.01, 0.01, self.M))

    def h(self, x):  # tanh activation function
        return torch.tanh(x)

    def h_derivative(self, x):
        return 1 - torch.tanh(x) ** 2

    def forward(self, z):
        self.linear = torch.matmul(self.B, z) + self.b
        return z + torch.matmul(self.A, self.h(self.linear))

    def log_det(self):
        I = torch.diag(torch.ones(self.M, device=DEVICE))
        diag = torch.diag(self.h_derivative(self.linear.flatten()))
        BA = torch.matmul(self.B, self.A)
        return torch.log(torch.det(I + torch.matmul(diag, BA)))


class HouseholderTransform(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.v = nn.Parameter(parameter_init(-0.01, 0.01, dim))

    def forward(self, z):
        vtz = torch.dot(self.v, z)
        vvtz = self.v * vtz
        z_new = z - 2 * vvtz / torch.sum(self.v ** 2)
        return z_new

    def log_det(self):
        return 0


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.round()  # convert to (0,1)

    @staticmethod
    def backward(ctx, grad_output):
        # return F.hardtanh(grad_output)
        return grad_output


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


import math
import torch.nn as nn
from torch.nn.init import xavier_normal


class ARLinear(nn.Module):
    def __init__(self, in_size, out_size, bias=True, ):
        super(ARLinear, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.weight = nn.Parameter(torch.Tensor(self.in_size, self.out_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, ):
        stdv = (2 / self.in_size)
        self.weight = nn.Parameter(stdv * torch.randn(self.in_size, self.out_size))

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.dim() == 2 and self.bias is not None:
            return torch.addmm(self.bias, input, self.weight.tril(-1))

        output = input @ self.weight.tril(-1)
        if self.bias is not None:
            output += self.bias
        return output


class MLP(nn.Sequential):
    """Multilayer perceptron"""

    def __init__(self, *layer_sizes, leaky_a=0.0):
        layers = []
        for s1, s2 in zip(layer_sizes, layer_sizes[1:]):
            layers.append(ARLinear(s1, s2))
            #  layers.append(nn.BatchNorm1d(s2))
            layers.append(nn.LeakyReLU(leaky_a))
        super().__init__(*layers[:-1])  # drop last ReLU

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)



### code for the autoregressive NN is taken from Andrej Karpathy
### https://github.com/karpathy/pytorch-made

class MADE(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.LeakyReLU(0.001) #should be 0.0001
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)
        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.num_masks == 1: return  # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.net(x)


class IAF(nn.Module):
    

    def __init__(self, dim, h_sizes=[250,250]):
        super().__init__()
        self.net = MADE(nin=dim, hidden_sizes=h_sizes, nout =2 * dim)
    def forward(self, z):  # z -> x

        out = self.net(z)
        first_half = int(out.shape[-1] / 2)
        shift = out[:first_half]
        scale = out[first_half:]
        self.gate = torch.sigmoid(scale)
        x = z * self.gate + (1 - self.gate) * shift
        
        return x

    def log_det(self):
        return (torch.log(self.gate)).sum(-1)



