"""
Environment for training with PrometheusEvalPreferenceModel.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Sequence

import chz
import numpy as np
import tinker
from tinker_cookbook import renderers
from rubric_preference_types import (
    PrometheusEvalComparison,
    PrometheusEvalPreferenceModel,
    PrometheusEvalPreferenceModelFromChatRenderer,
)
from tinker_cookbook.rl.preference_envs import PreferenceEnv, TournamentPattern, get_pairs
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer, Tokenizer
from tinker_cookbook.utils.misc_utils import safezip
from tinker_cookbook.rl.preference_envs import TournamentPattern, get_pairs

logger = logging.getLogger(__name__)


@chz.chz
class HFPrometheusEvalDatasetBuilder():
    """
    Builds HF datasets for training with PrometheusEvalPreferenceModel.

    It expects rows in the dataset to have the following fields:
        prompt_conversation: list[dict]
        rubric: str
        reference: str | None
    """
    
    model_name_for_tokenizer: str
    renderer_name: str

    @property
    def tokenizer(self) -> Tokenizer:
        return get_tokenizer(self.model_name_for_tokenizer)

    @property
    def renderer(self) -> renderers.Renderer:
        return renderers.get_renderer(self.renderer_name, self.tokenizer)


class PrometheusEvalPairwisePreferenceDataset(RLDataset):
    def __init__(
        self,
        dataset_builder: HFPrometheusEvalDatasetBuilder,
        batch_size: int,
        preference_model: PrometheusEvalPreferenceModel,
        tournament_pattern: TournamentPattern = TournamentPattern.ALL_PAIRS_BOTH_WAYS,
        group_size: int = 4,
    ):
        self.dataset_builder = dataset_builder
        self.batch_size = batch_size
        self.preference_model = preference_model
        self.train_dataset, _ = self.dataset_builder.get_train_and_test_datasets()
        self.tournament_pattern = tournament_pattern
        self.group_size = group_size

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        """Get a batch of EnvGroupBuilders for RL training."""
        start_index = index * self.batch_size
        # ensure the end index does not go past the dataset length
        end_index = min(start_index + self.batch_size, len(self.train_dataset))

        env_group_builders = []
        # iterate by index to correctly access each row from the Hugging Face dataset
        for i in range(start_index, end_index):
            item = self.train_dataset[i]
            builder = PrometheusEvalPairwisePreferenceGroupBuilder(
                convo_prefix=item["prompt_conversation"],
                policy_renderer=self.dataset_builder.renderer,
                tournament_pattern=self.tournament_pattern,
                preference_model=self.preference_model,
                rubric=item["rubric"],
                reference=item["reference"],
                num_envs=self.group_size,
            )
            env_group_builders.append(builder)
        return env_group_builders

    def __len__(self) -> int:
        return len(self.train_dataset) // self.batch_size


@chz.chz
class PrometheusEvalPairwisePreferenceRLDatasetBuilder(RLDatasetBuilder):
    comparison_dataset_builder: HFPrometheusEvalDatasetBuilder
    rm_renderer_name: str
    rm_model_name_for_tokenizer: str
    batch_size: int
    tournament_pattern: TournamentPattern = TournamentPattern.ALL_PAIRS_BOTH_WAYS
    rm_model_path: str
    group_size: int
    base_url: str | None = None

    async def __call__(self) -> tuple[PrometheusEvalPairwisePreferenceDataset, None]:
        tokenizer = get_tokenizer(self.rm_model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.rm_renderer_name, tokenizer=tokenizer)

        return PrometheusEvalPairwisePreferenceDataset(
            dataset_builder=self.comparison_dataset_builder,
            batch_size=self.batch_size,
            preference_model=PrometheusEvalPreferenceModelFromChatRenderer(
                convo_renderer=renderer,
                sampling_client=tinker.ServiceClient(base_url=self.base_url).create_sampling_client(
                    model_path=self.rm_model_path
                ),
            ),
            tournament_pattern=self.tournament_pattern,
            group_size=self.group_size,
        ), None



@dataclass(frozen=True)
class PrometheusEvalPairwisePreferenceGroupBuilder(EnvGroupBuilder):
    convo_prefix: list[renderers.Message]
    policy_renderer: renderers.Renderer
    tournament_pattern: TournamentPattern
    preference_model: PrometheusEvalPreferenceModel
    rubric: str
    reference: str | None
    num_envs: int

    async def make_envs(self) -> Sequence[Env]:
        return [
            PreferenceEnv(self.convo_prefix, self.policy_renderer) for _ in range(self.num_envs)
        ]

    async def compute_group_rewards(self, trajectory_group: list[Trajectory]) -> list[tuple[float, Metrics]]:
        """Compute rewards for a group of trajectories using pairwise preference model.
        
        The reward for each trajectory is computed as the sum of:
        1. Tournament reward based on pairwise comparisons with other trajectories in the group.
        2. Formatting reward based on whether the response is valid (1.0 if valid else -1.0).
        """
        # TODO: add your code here
        # Note that for using the rubric RM to compute the reward for a pair of responses,
        # you are calling Tinker API to get the model response. You are suggested to make
        # the calls asynchronously for better efficiency. You can use asyncio.gather to
        # achieve that.

        # 1. reconstruct completions from trajectories
        completions = []
        for traj in trajectory_group:
            # a trajectory is a list of transitions, each transition has an action with tokens
            # concatenate tokens from all transitions to get the full sequence
            tokens = [token for transition in traj.transitions for token in transition.ac.tokens]
            decoded = self.policy_renderer.tokenizer.decode(tokens).strip()
            completions.append(decoded)

        # calculate formatting reward
        formatting_rewards = np.array([1.0 if text else -1.0 for text in completions])

        # create tournament pairs
        pairs = get_pairs(self.num_envs, self.tournament_pattern)
        
        # asynchronously compare all pairs, thanks prometheus!
        comparison_tasks = []
        for i, j in pairs:
            comparison = PrometheusEvalComparison(
                prompt_conversation=self.convo_prefix,
                completion_A=[{"role": "assistant", "content": completions[i]}],
                completion_B=[{"role": "assistant", "content": completions[j]}],
                rubric=self.rubric,
                reference=self.reference,
            )
            comparison_tasks.append(self.preference_model(comparison))
        
        preference_scores = await asyncio.gather(*comparison_tasks)
        
        # add up tournament rewards from pairwise results
        tournament_rewards = np.zeros(self.num_envs)
        for (i, j), score in zip(pairs, preference_scores):
            if score < 0: # completion A wins
                tournament_rewards[i] += 1.0
                tournament_rewards[j] -= 1.0
            elif score > 0: # completion B wins
                tournament_rewards[j] += 1.0
                tournament_rewards[i] -= 1.0
                
        # combine rewards and return
        total_rewards = tournament_rewards + formatting_rewards
        return [(reward, {}) for reward in total_rewards]