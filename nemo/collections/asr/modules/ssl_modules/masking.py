# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


from typing import Optional, Union

import torch
import torch.nn as nn

from nemo.core.classes import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType


class RandomBlockMasking(NeuralModule):
    """
    Performs random block masking on sequence of features.
    Args:
        mask_prob (float): percentage of sequence to mask
        block_size (int): size of each block to mask
        mask_value (Optional[float]): value to use for masking, if None, use random values
        feat_in (Optional[int]): size of input features, required if mask_value is None
        freeze (bool): if True, mask embedding is not trainable
        allow_overlap (bool): if True, masked blocks can overlap
    """

    def __init__(
        self,
        feat_in: int,
        mask_prob: float = 0.5,
        block_size: int = 48,
        mask_value: Optional[float] = None,
        freeze: bool = True,
        allow_overlap: bool = False,
        max_mask_ratio: float = 0.8,
    ):
        super().__init__()
        self.block_size = block_size
        self.mask_prob = mask_prob
        self.allow_overlap = allow_overlap
        self.max_mask_ratio = max_mask_ratio

        if mask_value is None:
            self.mask_embedding = nn.Parameter(torch.FloatTensor(feat_in))
            nn.init.normal_(self.mask_embedding, mean=0.0, std=0.1)
        else:
            self.mask_embedding = nn.Parameter(torch.ones(feat_in) * mask_value, requires_grad=False)
        if freeze:
            self.freeze()

    @property
    def input_types(self):
        """Returns definitions of module input types"""
        return {
            "input_feats": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "input_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output types"""
        return {
            "maksed_feats": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "masks": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
        }

    def forward(self, input_feats: torch.Tensor, input_lengths: torch.Tensor):
        """
        Args:
            input_feats (Tensor): input sequence features, shape=(batch, features, time)
            input_length (Tensor): length of each sequence in the batch, shape=(batch)
        Returns:
            masked_feats (Tensor): masked features, shape=(batch, features, time)
            masks (Tensor): the generated masks, shape=(batch, features, time)
        """
        if self.allow_overlap:
            return self.forward_with_overlap(input_feats, input_lengths)
        else:
            return self.forward_without_overlap(input_feats, input_lengths)

    def forward_without_overlap(self, input_feats: torch.Tensor, input_lengths: torch.Tensor):
        """
        Args:
            input_feats (Tensor): input sequence features, shape=(batch, features, time)
            input_length (Tensor): length of each sequence in the batch, shape=(batch)
        Returns:
            masked_feats (Tensor): masked features, shape=(batch, features, time)
            masks (Tensor): the generated masks, shape=(batch, features, time)
        """
        batch_size = input_feats.size(0)
        masks = torch.zeros_like(input_feats)
        masked_feats = input_feats
        indices = []
        for i in range(batch_size):
            if self.block_size >= input_lengths[i] * self.max_mask_ratio:
                # handle case where audio is too short
                block_size = 8
                num_patches = 1
                patch_indices = torch.tensor([0])
                offset = 0
            else:
                num_patches = torch.ceil(input_lengths[i] * self.mask_prob / self.block_size).int()
                offset = torch.randint(0, self.block_size, (1,))[0]
                block_size = self.block_size
                if (num_patches + 1) * self.block_size > input_lengths[i]:
                    block_size = torch.div(input_lengths[i], (num_patches + 1), rounding_mode='trunc')
                max_num_patches = torch.div(input_lengths[i], block_size, rounding_mode='trunc')
                patch_indices = torch.randperm(max_num_patches - 1)[:num_patches]

            if num_patches:
                starts = patch_indices * block_size + offset
                ends = starts + block_size
                positions = torch.cat([torch.arange(s, e) for s, e in zip(starts, ends)]).reshape(-1, 1)
                batch_index = torch.full((positions.shape[0], 1), i, dtype=positions.dtype)
                positions = torch.cat([batch_index, positions], dim=1)
                indices.append(positions.unique(dim=0))

        if indices:
            indices = torch.cat(indices, dim=0).unbind(1)
            masks = masks.permute(0, 2, 1)
            masked_feats = masked_feats.permute(0, 2, 1)

            masks = masks.index_put(indices, values=torch.tensor(1.0)).permute(0, 2, 1)
            masked_feats = masked_feats.index_put(indices, values=self.mask_embedding).permute(0, 2, 1)

        return masked_feats, masks

    def forward_with_overlap(self, input_feats: torch.Tensor, input_lengths: torch.Tensor):
        """
        Args:
            input_feats (Tensor): input sequence features, shape=(batch, features, time)
            input_length (Tensor): length of each sequence in the batch, shape=(batch)
        Returns:
            masked_feats (Tensor): masked features, shape=(batch, features, time)
            masks (Tensor): the generated masks, shape=(batch, features, time)
        """
        batch_size = input_feats.size(0)
        masks = torch.zeros_like(input_feats)
        masked_feats = input_feats
        mask_prob = torch.tensor(self.mask_prob)
        indices = []
        for i in range(batch_size):
            input_length = input_lengths[i].item()
            if self.block_size >= input_length * self.max_mask_ratio:
                # handle case where audio is too short
                block_size = 8
                num_patches = 1
                patch_indices = torch.tensor([0])
            else:
                block_size = self.block_size
                count = max(0, input_length - self.block_size)
                num_patches = torch.binomial(torch.tensor(count).float(), mask_prob).long()
                patch_indices = torch.randperm(count)
                patch_indices = patch_indices[:num_patches]
            if num_patches:
                ends = torch.clamp(patch_indices + block_size, max=input_length)
                positions = torch.cat([torch.arange(s, e) for s, e in zip(patch_indices, ends)]).reshape(-1, 1)
                batch_index = torch.full((positions.shape[0], 1), i, dtype=positions.dtype)
                positions = torch.cat([batch_index, positions], dim=1)
                indices.append(positions.unique(dim=0))

        if indices:
            indices = torch.cat(indices, dim=0).unbind(1)
            masks = masks.permute(0, 2, 1)
            masked_feats = masked_feats.permute(0, 2, 1)

            masks = masks.index_put(indices, values=torch.tensor(1.0)).permute(0, 2, 1)
            masked_feats = masked_feats.index_put(indices, values=self.mask_embedding).permute(0, 2, 1)

        return masked_feats, masks


class ConvFeatureMaksingWrapper(NeuralModule):
    """
    A wrapper module that applies masking to the features after subsampling layer of ConformerEncoder.
    """

    def __init__(self, pre_encode_module: nn.Module, masking_module: Union[nn.Module, NeuralModule]) -> None:
        """
        Args:
            pre_encode_module: the pre_encode module of the ConformerEncoder instance
            masking_module: the module that performs masking on the extracted features
        """
        super().__init__()
        self.pre_encode = pre_encode_module
        self.masking = masking_module
        self.curr_mask = None
        self.curr_feat = None
        self.apply_mask = False

    def forward(self, x, lengths):
        """
        Same interface as ConformerEncoder.pre_encode
        """
        feats, lengths = self.pre_encode(x=x, lengths=lengths)
        self.curr_feat = feats.detach()
        if self.apply_mask:
            feats = feats.transpose(1, 2)
            masked_feats, self.curr_mask = self.masking(input_feats=feats, input_lengths=lengths)
            masked_feats = masked_feats.transpose(1, 2).detach()
        else:
            masked_feats = feats
            self.curr_mask = torch.zeros_like(feats)
        return masked_feats, lengths

    def set_masking_enabled(self, apply_mask: bool):
        self.apply_mask = apply_mask

    def get_current_mask(self):
        return self.curr_mask

    def get_current_feat(self):
        return self.curr_feat
