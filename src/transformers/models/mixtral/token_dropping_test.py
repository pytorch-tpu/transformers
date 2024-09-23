#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


# python src/transformers/models/mixtral/token_dropping_test.py


import torch
import unittest
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

class TokenDroppingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    config = MixtralConfig(
        capacity_factor=2,
        num_experts=8,
        static=True,
        gmm=False,
        gmm_stack=False,
    )

    self.moe_block = MixtralSparseMoeBlock(config)


  def test_generate_masks(self):
    # expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
    # expert_capacity_in_batch = (4 * 2 / 8) * 2 = 2

    softmax_probs = torch.tensor(
        [
            [
                [0.20, 0, 0, 0, 0, 0.80, 0, 0],
                [0.68, 0, 0, 0, 0.32, 0, 0, 0],
                [0.22, 0.78, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0.32, 0, 0.68, 0, 0],
            ],
            [
                [0, 0.26, 0.74, 0, 0, 0, 0, 0],
                [0, 0.79, 0, 0, 0.21, 0, 0, 0],
                [0.89, 0, 0, 0, 0, 0.11, 0, 0],
                [0, 0.11, 0, 0, 0, 0, 0, 0.89],
            ],
            [
                [0, 0, 0.26, 0, 0, 0, 0.74, 0],
                [0, 0, 0.88, 0.12, 0, 0, 0, 0],
                [0, 0, 0.17, 0, 0.83, 0, 0, 0],
                [0, 0.35, 0.65, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0.47, 0, 0, 0.53, 0, 0, 0],
                [0.36, 0, 0, 0, 0, 0, 0, 0.64],
                [0.15, 0, 0, 0, 0, 0.85, 0, 0],
                [0, 0, 0, 0, 0.18, 0, 0, 0.82],
            ],
        ],
        dtype=torch.float32,
    )

    routing_weights = torch.tensor(
        [
            [[0.80, 0.20], [0.68, 0.32], [0.78, 0.22], [0.68, 0.32]],
            [[0.74, 0.26], [0.79, 0.21], [0.89, 0.11], [0.89, 0.11]],
            [[0.74, 0.26], [0.88, 0.12], [0.83, 0.17], [0.65, 0.35]],
            [[0.53, 0.47], [0.64, 0.36], [0.85, 0.15], [0.82, 0.18]],
        ],
        dtype=torch.float32,
    )

    top_k_indices = torch.tensor(
        [
            [[5, 0], [0, 4], [1, 0], [5, 3]],
            [[2, 1], [1, 4], [0, 5], [7, 1]],
            [[6, 2], [2, 3], [4, 2], [2, 1]],
            [[4, 1], [7, 0], [5, 0], [4, 7]],
        ],
        dtype=torch.int64,
    )

    expected_top_k_weight_mask = torch.tensor(
        [
            [[1, 1], [1, 1], [1, 0], [1, 1]],  # drop [0,2,1] expert_index 0 got overflow
            [[1, 1], [1, 1], [1, 1], [1, 0]],  # drop [1,3,1] expert_index 1 got overflow
            [[1, 1], [1, 1], [1, 0], [0, 1]],  # drop [2,2,1] [2,3,0] expert_index 2 got overflow
            [[1, 1], [1, 1], [1, 1], [1, 1]],  # no dropping
        ],
        dtype=torch.bool,
    )

    # shape of dispatch_mask & combine_mask: (batch_size, seq_len, num_experts, expert_capacity_per_batch)
    expected_combine_mask = torch.tensor(
        [
            [
                [[0.2, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.8, 0], [0, 0], [0, 0]],
                [[0, 0.68], [0, 0], [0, 0], [0, 0], [0.32, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0.78, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0.32, 0], [0, 0], [0, 0.68], [0, 0], [0, 0]],
            ],
            [
                [[0, 0], [0.26, 0], [0.74, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0.79], [0, 0], [0, 0], [0.21, 0], [0, 0], [0, 0], [0, 0]],
                [[0.89, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.11, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.89, 0]],
            ],
            [
                [[0, 0], [0, 0], [0.26, 0], [0, 0], [0, 0], [0, 0], [0.74, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0.88], [0.12, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0.83, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0.35, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            ],
            [
                [[0, 0], [0.47, 0], [0, 0], [0, 0], [0.53, 0], [0, 0], [0, 0], [0, 0]],
                [[0.36, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0.64, 0]],
                [[0, 0.15], [0, 0], [0, 0], [0, 0], [0, 0], [0.85, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0.18], [0, 0], [0, 0], [0, 0.82]],
            ],
        ],
        dtype=torch.float32,
    )

    expected_softmax_probs_mask = torch.tensor(
        [
            [
                [1, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],  # 0th token dropped
                [0, 0, 0, 1, 0, 1, 0, 0],
            ],
            [
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],  # 1st token dropped
            ],
            [
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],  # 2nd token dropped
                [0, 1, 0, 0, 0, 0, 0, 0],  # 2nd token dropped
            ],
            [
                [0, 1, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1],
            ],
        ],
        dtype=torch.float32,
    )

    expected_dispatch_mask = expected_combine_mask.bool()
    actual_dispatch_mask, actual_token_mask, actual_top_k_weight_mask = self.moe_block.generate_masks(top_k_indices)

    self.assertTrue(torch.allclose(expected_dispatch_mask, actual_dispatch_mask))
    self.assertTrue(torch.allclose(expected_top_k_weight_mask, actual_top_k_weight_mask))
    self.assertTrue(torch.allclose(expected_softmax_probs_mask.sum(-1).bool(), actual_token_mask))


if __name__ == "__main__":
    unittest.main()
