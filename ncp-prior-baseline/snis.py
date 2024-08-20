import os
import random

from shutil import copyfile

import datetime, time
import logging
import sys

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse

class get_independent_normal():
     def __init__(self, data_dim, variance=1.0):
          """Returns an independent normal with event size the size of data_dim.

          Args:
          data_dim: List of data dimensions.
          variance: A scalar that is used as the diagonal entries of the covariance
               matrix.

          Returns:
          Independent normal distribution.
          """
          self.data_dim = data_dim

     def sample(self, num_samples):
          return torch.randn(num_samples, self.data_dim[0])

     

class AbstractNIS():
     """Self-normalized Importance Sampling distribution."""
     def __init__(self,  # pylint: disable=invalid-name
               K,
               data_dim,
               energy_fn,
               proposal=None,
               data_mean=None,
               reparameterize_proposal_samples=True,
               dtype=torch.float):


          """Creates a NIS model.

          Args:
          K: The number of proposal samples to take.
          data_dim: The dimension of the data.
          energy_fn: Energy function.
          proposal: A distribution over the data space of this model. Must support
          sample() and log_prob() although log_prob only needs to return a lower
          bound on the true log probability. If not supplied, then defaults to
          Gaussian.
          data_mean: Mean of the data used to center the input.
          reparameterize_proposal_samples: Whether to allow gradients to pass
          through the proposal samples.
          dtype: Type of the tensors.
          """
          self.K = K   # pylint: disable=invalid-name
          self.data_dim = data_dim  # self.data_dim is always a list
          self.reparameterize_proposal_samples = reparameterize_proposal_samples
          if data_mean is not None:
               self.data_mean = data_mean
          else:
               self.data_mean = torch.zeros((), dtype=dtype)  # centre of the proposal distn
          self.energy_fn = energy_fn
          if proposal is None:
               self.proposal = get_independent_normal(self.data_dim)
          else:
               self.proposal = proposal

     def sample(self, num_samples=1):
          """Sample from the model."""
          num_samples = 1
          flat_proposal_samples = self.proposal.sample(num_samples * self.K).cuda() # (num_samples * self.K, data_dim)
          proposal_samples = torch.reshape(flat_proposal_samples,
                                  [num_samples, self.K] + self.data_dim)
          with torch.no_grad():
               log_energy = torch.reshape(
                    torch.squeeze(self.energy_fn(flat_proposal_samples), axis=-1),
                    [num_samples, self.K])  
               indexes = td.categorical.Categorical(logits=log_energy).sample()
               #Todo: Check the below step. Works for only one sample now. 
               #samples = torch.gather(proposal_samples, 1, indexes.unsqueeze(-1))
               samples = flat_proposal_samples[indexes]
          return torch.squeeze(samples, dim=1)

class NIS(AbstractNIS):
     def __init__(self,
               K,
               data_dim,
               energy_fn,
               energy_hidden_sizes=None,
               proposal=None,
               data_mean=None,
               reparameterize_proposal_samples=False,
               dtype=torch.float,
               name="nis"):
          """Creates a NIS model.

          Args:
               energy_function: the energy funciton model : f: data_dim --> R
               K: The number of proposal samples to take.
               data_dim: The dimension of the data.
               energy_hidden_sizes: The sizes of the hidden layers for the MLP that
               parameterizes the energy function.
               proposal: A distribution over the data space of this model. Must support
               sample() and log_prob() although log_prob only needs to return a lower
               bound on the true log probability. If not supplied, then defaults to
               Gaussian.
               data_mean: Mean of the data used to center the input.
               reparameterize_proposal_samples: Whether to allow gradients to pass
               through the proposal samples.
               dtype: Type of the tensors.
               name: Name to use for ops.
          """
          if data_mean is None:
               data_mean = torch.zeros((), dtype=dtype)
          super(NIS, self).__init__(K, data_dim, energy_fn, proposal, data_mean,
                              reparameterize_proposal_samples, dtype)