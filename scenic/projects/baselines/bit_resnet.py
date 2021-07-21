# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of ResNetV1 with group norm and weight standardization.

Ported from:
https://github.com/google-research/big_transfer/blob/master/bit_jax/models.py
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
from scenic.model_lib.base_models.classification_model import ClassificationModel
from scenic.model_lib.base_models.multilabel_classification_model import MultiLabelClassificationModel
from scenic.model_lib.layers import nn_layers


def weight_standardize(w: jnp.ndarray,
                       axis: Union[Sequence[int], int],
                       eps: float):
  """Standardize (mean=0, var=1) a weight."""
  w = w - jnp.mean(w, axis=axis, keepdims=True)
  w = w / jnp.sqrt(jnp.mean(jnp.square(w), axis=axis, keepdims=True) + eps)
  return w


class StdConv(nn.Conv):
  """Convolution with weight standardized kernel."""

  def param(self,
            name: str,
            initializer: Callable[..., Any],
            shape: Tuple[int, ...]):
    param = super().param(name, initializer, shape)
    if name == 'kernel':
      param = weight_standardize(param, axis=[0, 1, 2], eps=1e-10)
    return param


class ResidualUnit(nn.Module):
  """Bottleneck ResNet block.

  Attributes:
    nout: Number of output features.
    strides: Downsampling stride.
    dilation: Kernel dilation.
    bottleneck: If True, the block is a bottleneck block.
  """
  nout: int
  strides: Tuple[int, ...] = (1, 1)
  dilation: Tuple[int, ...] = (1, 1)
  bottleneck: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    features = self.nout
    nout = self.nout * 4 if self.bottleneck else self.nout
    needs_projection = x.shape[-1] != nout or self.strides != (1, 1)
    residual = x
    if needs_projection:
      residual = StdConv(nout,
                         (1, 1),
                         self.strides,
                         use_bias=False,
                         name='conv_proj')(residual)
      residual = nn.GroupNorm(epsilon=1e-4, name='gn_proj')(residual)

    if self.bottleneck:
      x = StdConv(features, (1, 1), use_bias=False, name='conv1')(x)
      x = nn.GroupNorm(epsilon=1e-4, name='gn1')(x)
      x = nn.relu(x)

    x = StdConv(features, (3, 3), self.strides, kernel_dilation=self.dilation,
                use_bias=False, name='conv2')(x)
    x = nn.GroupNorm(epsilon=1e-4, name='gn2')(x)
    x = nn.relu(x)

    last_kernel = (1, 1) if self.bottleneck else (3, 3)
    x = StdConv(nout, last_kernel, use_bias=False, name='conv3')(x)
    x = nn.GroupNorm(
        epsilon=1e-4, name='gn3', scale_init=nn.initializers.zeros)(x)
    x = nn.relu(residual + x)

    return x


class ResNetStage(nn.Module):
  """ResNet Stage: one or more stacked ResNet blocks.

  Attributes:
    block_size: Number of ResNet blocks to stack.
    nout: Number of features.
    first_stride: Downsampling stride.
    first_dilation: Kernel dilation.
    bottleneck: If True, the bottleneck block is used.
  """

  block_size: int
  nout: int
  first_stride: Tuple[int, ...]
  first_dilation: Tuple[int, ...] = (1, 1)
  bottleneck: bool = True

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x = ResidualUnit(self.nout,
                     strides=self.first_stride,
                     dilation=self.first_dilation,
                     bottleneck=self.bottleneck,
                     name='unit1')(x)
    for i in range(1, self.block_size):
      x = ResidualUnit(self.nout,
                       strides=(1, 1),
                       bottleneck=self.bottleneck,
                       name=f'unit{i + 1}')(x)
    return x


class BitResNet(nn.Module):
  """Bit ResNetV1.

  Attributes:
    num_outputs: Num output classes. If None, a dict of intermediate feature
      maps is returned.
    width_factor: Width multiplier for each of the ResNet stages.
    num_layers: Number of layers (see `BLOCK_SIZE_OPTIONS` for stage
      configurations).
    max_output_stride: Defines the maximum output stride of the resnet.
      Typically, resnets output feature maps have sride 32. We can, however,
      lower that number by swapping strides with dilation in later stages. This
      is common in cases where stride 32 is too large, e.g., in dense prediciton
      tasks.
  """

  num_outputs: Optional[int] = 1000
  width_factor: int = 1
  num_layers: int = 50
  max_output_stride: int = 32

  @nn.compact
  def __call__(self,
               x: jnp.ndarray,
               train: bool = True,
               debug: bool = False
               ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Applies the Bit ResNet model to the inputs.

    Args:
      x: Inputs to the model.
      train: Unused.
      debug: Unused.

    Returns:
       Un-normalized logits if `num_outputs` is provided, a dictionary with
       representations otherwise.
    """
    del train
    del debug
    if self.max_output_stride not in [4, 8, 16, 32]:
      raise ValueError('Only supports output strides of [4, 8, 16, 32]')

    blocks, bottleneck = BLOCK_SIZE_OPTIONS[self.num_layers]

    width = int(64 * self.width_factor)

    # Root block.
    x = StdConv(width, (7, 7), (2, 2), use_bias=False, name='conv_root')(x)
    x = nn.GroupNorm(epsilon=1e-4, name='gn_root')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    representations = {'stem': x}

    # Stages.
    x = ResNetStage(
        blocks[0],
        width,
        first_stride=(1, 1),
        bottleneck=bottleneck,
        name='block1')(x)
    stride = 4
    for i, block_size in enumerate(blocks[1:], 1):
      max_stride_reached = self.max_output_stride <= stride
      x = ResNetStage(
          block_size,
          width * 2**i,
          first_stride=(2, 2) if not max_stride_reached else (1, 1),
          first_dilation=(2, 2) if max_stride_reached else (1, 1),
          bottleneck=bottleneck,
          name=f'block{i + 1}')(x)
      if not max_stride_reached:
        stride *= 2
      representations[f'stage_{i + 1}'] = x

    if self.num_outputs:
      # Head.
      x = jnp.mean(x, axis=(1, 2))
      x = nn_layers.IdentityLayer(name='pre_logits')(x)
      x = nn.Dense(
          self.num_outputs,
          kernel_init=nn.initializers.zeros,
          name='output_projection')(x)
      return x
    return representations


# A dictionary mapping the number of layers in a resnet to the number of
# blocks in each stage of the model. The second argument indicates whether we
# use bottleneck layers or not.
BLOCK_SIZE_OPTIONS = {
    5: ([1], True),  # Only strided blocks. Total stride 4.
    8: ([1, 1], True),  # Only strided blocks. Total stride 8.
    11: ([1, 1, 1], True),  # Only strided blocks. Total stride 16.
    14: ([1, 1, 1, 1], True),  # Only strided blocks. Total stride 32.
    9: ([1, 1, 1, 1], False),  # Only strided blocks. Total stride 32.
    18: ([2, 2, 2, 2], False),
    26: ([2, 2, 2, 2], True),
    34: ([3, 4, 6, 3], False),
    50: ([3, 4, 6, 3], True),
    101: ([3, 4, 23, 3], True),
    152: ([3, 8, 36, 3], True),
    200: ([3, 24, 36, 3], True)
}


class BitResNetClassificationModel(ClassificationModel):
  """Implements the Bit ResNet model for classification."""

  def build_flax_model(self) -> nn.Module:
    return BitResNet(
        num_outputs=self.dataset_meta_data['num_classes'],
        width_factor=self.config.get('width_factor', 1),
        num_layers=self.config.num_layers)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return _get_default_configs_for_testing()


class BitResNetMultiLabelClassificationModel(MultiLabelClassificationModel):
  """Implements the Bit ResNet model for multi-label classification."""

  def build_flax_model(self) -> nn.Module:
    return BitResNet(
        num_outputs=self.dataset_meta_data['num_classes'],
        width_factor=self.config.get('width_factor', 1),
        num_layers=self.config.num_layers)

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    return _get_default_configs_for_testing()


def _get_default_configs_for_testing() -> ml_collections.ConfigDict:
  return ml_collections.ConfigDict(dict(
      width_factor=1,
      num_layers=5,
  ))