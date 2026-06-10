#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


class NoisyModule(nn.Module):
    """
    Base class to collect some functions that will be used by all noisy ANN
    implementations.

    Can have noise injected at some intermediate layer when `noisy` is True.
    The shape of the noise injected is specified by `cov`.

    The layer at which noise is injected is inferred from params.
    Layers are zero-indexed. The 0-th layer is the first hidden layer (the
    input layer is not counted) and the last layer is the output layer.

    Parameters
    ----------

    `params`: param_utils.Params
        used to get `params.activn`, `params.noisy_layer` and
        `params.cov_filename()`.

    `noisy`: False, True, 'zero', 'identity', or 'diagonal'
        If set to False, the noiseless version of the network is instantiated.
        If `noisy` is 'zero', don't add noise, but train only post-noise layers.
        If `noisy` is 'identity', use an identity covariance matrix.
        If `noisy` is True, use the saved covariance matrix; and
        if `noisy` is 'diagonal', set its non-diagonal entries to zero.
    """

    def __init__(self, params, device=None):
        # `noisy` is set separately from params because different files will
        # initialize the network differently.
        super().__init__()
        self.activn = F.relu if params.activn == 'relu' else torch.tanh
        self.noisy = params.args.noisy
        self.pre_gen = params.args.pre_gen
        self.alpha = params.args.alpha
        self.beta = params.args.beta

        # Set the layer at which to add noise
        # This value is required even if noisy is False, e.g., when extracting
        # activations of the noisy layer from the base model to compute the covariance
        self.noisy_layer = params.noisy_layer
        self.trace_scale = params.args.trace_scale

        # List to collect outputs from different layers - will typically
        # exclude the output of convolutional layers before max-pooling
        self.outputs = []

        if device:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if self.noisy:
            # Initialize the covariance matrix
            if self.noisy == 'identity':
                self.cov = torch.eye(self._noise_dim())
            elif self.noisy == 'zero':
                self.cov = torch.zeros((self._noise_dim(), self._noise_dim()))
            else:
                self.cov = torch.from_numpy(np.load(params.cov_filename())).to(dtype=torch.float32)
                assert self._noise_dim() == self.cov.shape[0]
            if self.noisy == 'diagonal': 
                self.cov *= torch.eye(self.cov.shape[0])

            # Normalize the trace of the noise covariance matrix if trace_scale is specified 
            # Scales the matrix such that the trace is t*n for an nxn covariance matrix 
            # with a trace_scale t
            if self.trace_scale is not None and self.noisy is not None and self.noisy != 'zero':
                current_trace = np.trace(self.cov)
                goal_trace = self.trace_scale * self.cov.shape[0]
                C = goal_trace / current_trace
                self.cov = C * self.cov   
                
                assert np.isclose(np.trace(self.cov), goal_trace)

            # Used to create a linear combination of the covariance matrix with the identity matrix
            # if alpha and/or beta are specified 
            if self.alpha is not None:
                if self.beta is None:
                    self.beta = 1 - self.alpha
                self.cov = (self.alpha * self.cov) + ((self.beta * torch.eye(self._noise_dim())))

            self.cov.to(device)

            # Initialize a random number generator
            self.rng = np.random.default_rng()
            if self.device.type == 'cuda' and self.pre_gen:     # Pre-generate the noise
                self.generate_noise_tensor(size=params.num_epochs*params.num_train)
            else:
                if self.noisy == 'zero': # Cannot take cholesky decomposition of zero matrix 
                    self.cov_sqrt = None    
                else:
                    self.cov_sqrt = torch.linalg.cholesky(self.cov, upper=True).to(self.device)
                    # Add small number to the diagonal for numerical stability
                    self.cov_sqrt = self.cov_sqrt + (torch.eye(self.cov.shape[0]).to(self.device) * 1e-6) 


    @staticmethod
    def _layer_num(name):
        """Extract layer number from a layer name."""
        return int(name.split('.')[0][-1])


    def _noise_dim(self):
        """Return the size of the noisy layer."""
        return np.prod(self._layer_shapes[self.noisy_layer])


    def _add_noise(self, layer, x):
        """
        Add noise to the activations of the noisy layer.
        """
        if not self.noisy or layer != self.noisy_layer or self.noisy=="zero":
            return x

        if self.device.type == 'cuda' and self.pre_gen:
            # Work around for running attacks that run the model for more than num_epochs * num_train iterations
            # Regenerate noise when we are reaching the end of the noise
            if self.i + x.shape[0] >= self.noise.shape[0]:
                self.generate_noise_tensor(self.noise.shape[0])

            noise = self.noise[self.i : self.i + x.shape[0]]
            self.i += x.shape[0]
        else:
            # Add noise with the same covariance as self.cov
            n = self.cov.shape[0]
            G = torch.randn(size=(x.shape[0], n)).to(self.device)
            noise = G @ self.cov_sqrt
        return x + noise.view(x.shape)

    def generate_noise_tensor(self, size):
        """ Only used when pre_gen == True and cuda is available.
        Instead of dynamically generating noise, generate a large tensor of noise at the start
        and iterate through it during training"""
        noise = self.rng.multivariate_normal(
                    np.zeros(self.cov.shape[0]), self.cov,
                    size=(size)
        )
        self.noise = torch.tensor(noise.astype(np.float32)).to(self.device)
        self.i = 0

    def noisy_layer_output(self):
        """Return the activations of the noisy layer after a forward pass."""
        try:
            return self.outputs[self.noisy_layer]
        except:
            raise ValueError('Must run a forward pass before calling '
                             'noisy_layer_output')

    def freeze_layers(self):
        """
        Freezes layers upto and including `noisy_layer`. When noise is added to
        the hidden neurons of `noisy_layer`, only subsequent layers need
        retraining. Assumes that layer names contain the layer number as the
        last character.
        """
        for name, parameter in self.named_parameters():
            if self._layer_num(name) <= self.noisy_layer:
                parameter.requires_grad = False


    def post_noise_reinit(self):
        """Reset the weights of the post-noise layers."""
        for name, child in self.named_children():
            if self._layer_num(name) > self.noisy_layer:
                child.reset_parameters()



class Mnist_v1_1C5F(NoisyModule):
    """
    6-layer feedforward neural network with one convolutional layer.
    """

    _layer_shapes = [(32, 14, 14), (128,), (64,), (32,), (16,), (10,)]
    # Last layer is the output layer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*14*14, 128)  # 32 chans, image 14x14 after pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)


    def forward(self, x):
        conv0_out = self.activn(self.conv0(x))
        maxpool0_out = self._add_noise(0, F.max_pool2d(conv0_out, kernel_size=2,
                                                       stride=2)
                                      ).view(x.shape[0], -1)
        fc1_out = self._add_noise(1, self.activn(self.fc1(maxpool0_out)))
        fc2_out = self._add_noise(2, self.activn(self.fc2(fc1_out)))
        fc3_out = self._add_noise(3, self.activn(self.fc3(fc2_out)))
        fc4_out = self._add_noise(4, self.activn(self.fc4(fc3_out)))
        fc5_out = self.fc5(fc4_out)

        # Save outputs
        self.outputs = [maxpool0_out, fc1_out, fc2_out, fc3_out, fc4_out, fc5_out]
        return fc5_out



class Mnist_v2_3C3F(NoisyModule):
    """
    6-layer feedforward neural network with three convolutional layers.
    """

    _layer_shapes = [(6, 14, 14), (16, 14, 14), (16, 5, 5), (32,), (16,), (10,)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv0 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # Image remains 28x28 after this
        # max pool down to 14x14 after this
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=1)  # Image remains 14x14
        # No max pooling here
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=0) # Image becomes 10x10
        # max pool down to 5x5
        self.fc3 = nn.Linear(16*5*5, 32)  # 16 channels, image size 5x5 after max pooling
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)


    def forward(self, x):
        conv0_out = self.activn(self.conv0(x))
        maxpool0_out = self._add_noise(0, F.max_pool2d(conv0_out, kernel_size=2,
                                                       stride=2))
        conv1_out = self._add_noise(1, self.activn(self.conv1(maxpool0_out)))
        conv2_out = self.activn(self.conv2(conv1_out))
        maxpool2_out = self._add_noise(2, F.max_pool2d(conv2_out, kernel_size=2,
                                                       stride=2)
                                      ).view(x.shape[0], -1)
        fc3_out = self._add_noise(3, self.activn(self.fc3(maxpool2_out)))
        fc4_out = self._add_noise(4, self.activn(self.fc4(fc3_out)))
        fc5_out = self.fc5(fc4_out)

        # Save outputs
        self.outputs = [maxpool0_out.view(x.shape[0], -1),
                        conv1_out.view(x.shape[0], -1),
                        maxpool2_out, fc3_out, fc4_out, fc5_out]
        return fc5_out



class Mnist_v3_2C3F(NoisyModule):
    """
    5-layer feedforward neural network with two convolutional layers.
    """

    _layer_shapes = [(6, 14, 14), (16, 5, 5), (32,), (16,), (10,)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv0 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # Image remains 28x28 after this
        # max pool down to 14x14 after this
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, padding=0) # Image becomes 10x10
        # max pool down to 5x5
        self.fc2 = nn.Linear(16*5*5, 32)  # 16 channels, image size 5x5 after max pooling
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 10)

    def forward(self, x):
        conv0_out = self.activn(self.conv0(x))
        maxpool0_out = self._add_noise(0, F.max_pool2d(conv0_out, kernel_size=2,
                                                       stride=2))
        conv1_out = self.activn(self.conv1(maxpool0_out))
        maxpool1_out = self._add_noise(1, F.max_pool2d(conv1_out, kernel_size=2,
                                                       stride=2)
                                      ).view(x.shape[0], -1)
        fc2_out = self._add_noise(2, self.activn(self.fc2(maxpool1_out)))
        fc3_out = self._add_noise(3, self.activn(self.fc3(fc2_out)))
        fc4_out = self.fc4(fc3_out)

        # Save outputs
        self.outputs = [maxpool0_out.view(x.shape[0], -1), maxpool1_out,
                        fc2_out, fc3_out, fc4_out]
        return fc4_out
    