# Copyright 2022 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base class for all classification models."""

import functools
from typing import Dict, Optional, Tuple, Union

from flax.training import common_utils
from immutabledict import immutabledict
import jax.numpy as jnp
from scenic.model_lib.base_models import base_model
from scenic.model_lib.base_models import model_utils


# Standard default metrics for the classification models.
_CLASSIFICATION_METRICS = immutabledict({
    'accuracy':
        (model_utils.weighted_correctly_classified, model_utils.num_examples),
    'loss': (model_utils.weighted_unnormalized_softmax_cross_entropy,
             model_utils.num_examples)
})


def classification_metrics_function(
    logits: jnp.ndarray,
    batch: base_model.Batch,
    target_is_onehot: bool = False,
    metrics: base_model.MetricNormalizerFnDict = _CLASSIFICATION_METRICS,
    axis_name: Union[str, Tuple[str, ...]] = 'batch',
) -> Dict[str, Tuple[float, int]]:
  """Calculates metrics for the classification task.


  Currently we assume each metric_fn has the API:
    ```metric_fn(logits, targets, weights)```
  and returns an array of shape [batch_size]. We also assume that to compute
  the aggregate metric, one should sum across all batches, then divide by the
  total samples seen. In this way we currently only support metrics of the 1/N
  sum f(inputs, targets). Note, the caller is responsible for dividing by
  the normalizer when computing the mean of each metric.

  Args:
   logits: Output of model in shape [batch, length, num_classes].
   batch: Batch of data that has 'label' and optionally 'batch_mask'.
   target_is_onehot: If the target is a one-hot vector.
   metrics: The classification metrics to evaluate. The key is the name of the
     metric, and the value is the metrics function.
   axis_name: List of axes on which we run the pmsum.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  if target_is_onehot:
    one_hot_targets = batch['label']
  else:
    one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])
  weights = batch.get('batch_mask')  # batch_mask might not be defined

  return {
      key: model_utils.psum_metric_normalizer(
          (
              val[0](logits, one_hot_targets, weights),
              val[1](logits, one_hot_targets, weights),
          ),
          axis_name=axis_name,
      )
      for key, val in metrics.items()
  }


class ClassificationModel(base_model.BaseModel):
  """Defines commonalities between all classification models.

  A model is class with three members: get_metrics_fn, loss_fn, & a flax_model.

  get_metrics_fn returns a callable function, metric_fn, that calculates the
  metrics and returns a dictionary. The metric function computes f(x_i, y_i) on
  a minibatch, it has API:
    ```metric_fn(logits, label, weights).```

  The trainer will then aggregate and compute the mean across all samples
  evaluated.

  loss_fn is a function of API
    loss = loss_fn(logits, batch, model_params=None).

  This model class defines a softmax_cross_entropy_loss with weight decay,
  where the weight decay factor is determined by config.l2_decay_factor.

  flax_model is returned from the build_flax_model function. A typical
  usage pattern will be:
    ```
    model_cls = model_lib.models.get_model_cls('fully_connected_classification')
    model = model_cls(config, dataset.meta_data)
    flax_model = model.build_flax_model
    dummy_input = jnp.zeros(input_shape, model_input_dtype)
    model_state, params = flax_model.init(
        rng, dummy_input, train=False).pop('params')
    ```
  And this is how to call the model:
    variables = {'params': params, **model_state}
    logits, new_model_state = flax_model.apply(variables, inputs, ...)
    ```
  """

  def get_metrics_fn(self, split: Optional[str] = None) -> base_model.MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    del split  # For all splits, we return the same metric functions.
    return functools.partial(
        classification_metrics_function,
        target_is_onehot=self.dataset_meta_data.get('target_is_onehot', False),
        metrics=_CLASSIFICATION_METRICS)

  def loss_function(self,
                    logits: jnp.ndarray,
                    batch: base_model.Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns softmax cross entropy loss with an L2 penalty on the weights.

    Args:
      logits: Output of model in shape [batch, length, num_classes].
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    weights = batch.get('batch_mask')

    if self.dataset_meta_data.get('target_is_onehot', False):
      one_hot_targets = batch['label']
    else:
      one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])

    sof_ce_loss = model_utils.weighted_softmax_cross_entropy(
        logits,
        one_hot_targets,
        weights,
        label_smoothing=self.config.get('label_smoothing'))
    if self.config.get('l2_decay_factor') is None:
      return sof_ce_loss
    l2_loss = model_utils.l2_regularization(model_params)
    return sof_ce_loss + 0.5 * self.config.l2_decay_factor * l2_loss

  def build_flax_model(self):
    raise NotImplementedError('Subclasses must implement build_flax_model().')

  def default_flax_model_config(self):
    """Default config for the flax model that is built in `build_flax_model`.

    This function in particular serves the testing functions and supposed to
    provide config tha are passed to the flax_model when it's build in
    `build_flax_model` function, e.g., `model_dtype_str`.
    """
    raise NotImplementedError(
        'Subclasses must implement default_flax_model_config().')
