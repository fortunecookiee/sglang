# adapted from
# https://github.com/vllm-project/vllm/blob/82a1b1a82b1fbb454c82a9ef95730b929c9b270c/vllm/model_executor/layers/pooler.py

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.layers.activation import get_cross_encoder_activation_function
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class PoolingType(IntEnum):
    LAST = 0
    CLS = 1


@dataclass
class EmbeddingPoolerOutput:
    # Pooler can return list[tensor] instead of tensor if the dimension of each tensor in the batch is different
    # due to different per-request matryoshka dim truncation
    embeddings: torch.Tensor | list[torch.Tensor]


def pool_at_delimiter_positions(
    data: torch.Tensor,
    forward_batch: ForwardBatch,
    device: torch.device,
) -> List[torch.Tensor]:
    """Pool a tensor at the position before each MIS delimiter for every request.

    Uses pre-computed delimiter indices from ForwardBatch (CPU tensors),
    moves to GPU with non_blocking=True to avoid CUDA syncs.

    Args:
        data: 2-D tensor [total_tokens, dim] — hidden states or logits.
        forward_batch: Forward batch with extend_seq_lens_cpu and
                       multi_item_delimiter_indices populated.
        device: Device for the index tensor.

    Returns:
        One tensor per request, shaped [num_delimiters, dim].
    """
    all_index_tensors: List[torch.Tensor] = []
    delim_counts: List[int] = []
    offset = 0
    for req_idx, req_seq_len in enumerate(forward_batch.extend_seq_lens_cpu):
        indices_tensor = forward_batch.multi_item_delimiter_indices[req_idx]
        n = len(indices_tensor)
        if n > 0:
            # Note: if the first delimiter is at position 0 (empty query),
            # indices - 1 wraps to -1. This is harmless — the first delimiter
            # entry is always discarded by _process_multi_item_scoring_results.
            all_index_tensors.append(
                indices_tensor.to(device, non_blocking=True) + offset - 1
            )
        delim_counts.append(n)
        offset += req_seq_len

    if all_index_tensors:
        index_tensor = torch.cat(all_index_tensors)
    else:
        index_tensor = torch.tensor([], dtype=torch.long, device=device)
    return list(data[index_tensor].split(delim_counts))


def score_and_pool(
    score_head: nn.Module,
    pooler: "Pooler",
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    input_ids: torch.Tensor,
) -> EmbeddingPoolerOutput:
    """Apply a classification/score head with MIS support.

    MIS path: scores all tokens first, then pools logits at delimiter positions.
    Single-item path: pools hidden states via the Pooler, then scores.
    """
    if forward_batch.multi_item_delimiter_indices is not None:
        logits = score_head(hidden_states)
        scores = pool_at_delimiter_positions(logits, forward_batch, input_ids.device)
    else:
        pooled_hidden = pooler(hidden_states, forward_batch).embeddings
        scores = score_head(pooled_hidden)

    return EmbeddingPoolerOutput(embeddings=scores)


class Pooler(nn.Module):
    """A layer that pools specific information from hidden states.
    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.
    Attributes:
        pooling_type: The type of pooling to use (LAST, AVERAGE, MAX).
        normalize: Whether to normalize the pooled data.
    """

    def __init__(self, pooling_type: PoolingType, normalize: bool):
        super().__init__()
        self.pooling_type = pooling_type
        self.normalize = normalize

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> EmbeddingPoolerOutput:

        if self.pooling_type == PoolingType.LAST:
            last_token_indices = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            pooled_data = hidden_states[last_token_indices]
        elif self.pooling_type == PoolingType.CLS:
            prompt_lens = forward_batch.extend_seq_lens
            first_token_flat_indices = torch.zeros_like(prompt_lens)
            first_token_flat_indices[1:] += torch.cumsum(prompt_lens, dim=0)[:-1]
            pooled_data = hidden_states[first_token_flat_indices]
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        if forward_batch.dimensions is not None:
            all_same_dimensions = len(set(forward_batch.dimensions)) == 1
            if all_same_dimensions:
                pooled_data = pooled_data[..., : forward_batch.dimensions[0]]
            else:
                pooled_data = [
                    tensor[..., :dim]
                    for tensor, dim in zip(pooled_data, forward_batch.dimensions)
                ]

        if self.normalize:
            if isinstance(pooled_data, list):
                pooled_data = [
                    nn.functional.normalize(tensor, p=2, dim=-1)
                    for tensor in pooled_data
                ]
            else:
                pooled_data = nn.functional.normalize(pooled_data, p=2, dim=-1)

        return EmbeddingPoolerOutput(embeddings=pooled_data)


class CrossEncodingPooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `EmbeddingPoolerOutput`.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        classifier: nn.Module,
        pooler: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.classifier = classifier
        self.pooler = pooler
        self.default_activation_function = get_cross_encoder_activation_function(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> EmbeddingPoolerOutput:
        """Pools sentence pair scores from the hidden_states."""

        prompt_lens = forward_batch.extend_seq_lens

        offset = 0
        pooled_data_lst = []
        for prompt_len in prompt_lens:
            pooled_data_i = hidden_states[offset : offset + prompt_len]

            if self.pooler is not None:
                final_shape_tensor = self.pooler(pooled_data_i, forward_batch)
            else:
                final_shape_tensor = self.classifier(pooled_data_i)

            pooled_data_lst.append(final_shape_tensor)
            offset += prompt_len

        pooled_output = torch.stack(pooled_data_lst)

        if self.pooler is not None:
            # apply classifier once on the full batch if possible
            pooled_output = self.classifier(pooled_output)

        scores = self.default_activation_function(pooled_output).squeeze(-1)

        return EmbeddingPoolerOutput(embeddings=scores)
