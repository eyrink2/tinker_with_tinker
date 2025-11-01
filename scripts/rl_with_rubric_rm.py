import asyncio
from dotenv import load_dotenv

import chz
from datasets import Dataset, load_dataset
import numpy as np
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook import cli_utils, model_info
from rubric_preference_env import (
    PrometheusEvalPreferenceModelFromChatRenderer,
    PrometheusEvalComparison,
    HFPrometheusEvalDatasetBuilder,
    PrometheusEvalPairwisePreferenceRLDatasetBuilder,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.rl import train

from tinker_cookbook.rl.preference_envs import TournamentPattern  # Add import


class WinRateVsBaseOnTestEvaluator(SamplingClientEvaluator):
    """
    Computes win rate of the current policy (from training client) vs the initial policy,
    evaluated on the test split.

    Uses the PrometheusEvalPreferenceModel to judge between the two completions.
    """

    def __init__(
        self,
        comparison_dataset_builder: HFPrometheusEvalDatasetBuilder,
        policy_renderer_name: str,
        policy_tokenizer_model_name: str,
        reward_renderer_name: str,
        reward_tokenizer_model_name: str,
        reward_model_path: str,
        base_model_name: str,
        max_tokens: int,
        base_url: str | None = None,
    ):
        _train_ds, test_ds = comparison_dataset_builder.get_train_and_test_datasets()
        if test_ds is None:
            raise ValueError("Test dataset is not available for win-rate evaluation")
        self.test_ds = test_ds

        self.policy_renderer = renderers.get_renderer(
            policy_renderer_name, tokenizer=get_tokenizer(policy_tokenizer_model_name)
        )
        reward_renderer = renderers.get_renderer(
            reward_renderer_name, tokenizer=get_tokenizer(reward_tokenizer_model_name)
        )

        # Preference model (reward model)
        service_client = tinker.ServiceClient(base_url=base_url)
        reward_sampling_client = service_client.create_sampling_client(model_path=reward_model_path)
        self.preference_model = PrometheusEvalPreferenceModelFromChatRenderer(
            convo_renderer=reward_renderer,
            sampling_client=reward_sampling_client,
        )

        self.base_sampling_client = service_client.create_sampling_client(
            base_model=base_model_name
        )
        self.max_tokens = max_tokens

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """Evaluate the current policy against the base policy.
        If the current policy wins, it gets a score of 1.0; if the base policy wins, it gets a score of 0.0.
        
        Args:
            sampling_client: The sampling client for the current policy.
        
        Returns:
            A dictionary with keys "win_rate_vs_base" and "stderr_vs_base".
        """
        
        async def get_completion(client: tinker.SamplingClient, prompt: list) -> str:
            """Generate a single completion from a client."""
            model_input = self.policy_renderer.build_generation_prompt(prompt)
            response = await client.sample_async(
                model_input,
                num_samples=1,
                sampling_params=tinker.types.SamplingParams(
                    max_tokens=self.max_tokens, 
                    temperature=0.0
                ),
            )
            return self.policy_renderer.tokenizer.decode(response.sequences[0].tokens).strip()

        # generate completions from both policies at the same time
        policy_tasks = [get_completion(sampling_client, ex["prompt_conversation"]) for ex in self.test_ds]
        base_tasks = [get_completion(self.base_sampling_client, ex["prompt_conversation"]) for ex in self.test_ds]
        
        policy_responses = await asyncio.gather(*policy_tasks)
        base_responses = await asyncio.gather(*base_tasks)

        # run pairwise comparisons
        comparison_tasks = []
        for i, example in enumerate(self.test_ds):
            # format responses as message dictionaries for the preference model
            comparison = PrometheusEvalComparison(
                prompt_conversation=example["prompt_conversation"],
                completion_A=[{"role": "assistant", "content": policy_responses[i]}],
                completion_B=[{"role": "assistant", "content": base_responses[i]}],
                rubric=example["rubric"],
                reference=example.get("reference"),
            )
            comparison_tasks.append(self.preference_model(comparison))
        
        preference_scores = await asyncio.gather(*comparison_tasks)
        
        # calculate win rate (negative score means policy wins)
        wins = [1.0 if score < 0 else 0.0 for score in preference_scores]
        
        if len(wins) == 0:
            return {"win_rate_vs_base": 0.0, "stderr_vs_base": 0.0}

        win_rate = np.mean(wins)
        stderr = np.std(wins) / np.sqrt(len(wins))
        
        return {"win_rate_vs_base": win_rate, "stderr_vs_base": stderr}


@chz.chz
class PrometheusEvalDatasetBuilderFromJSONL(HFPrometheusEvalDatasetBuilder):
    train_data_path: str
    test_data_path: str | None = None

    def get_train_and_test_datasets(self) -> tuple[Dataset, Dataset | None]:
        
        train_ds = load_dataset("json", data_files=self.train_data_path, split="train")
        test_ds = load_dataset("json", data_files=self.test_data_path, split="train") if self.test_data_path else None

        def _prep(example: dict) -> dict:
            prompt_conversation = example["prompt_conversation"]
            reference = example.get("reference", None)
            rubric = example["rubric"]
            return {
                "prompt_conversation": prompt_conversation,
                "reference": reference,
                "rubric": rubric,
            }

        train_dataset = train_ds.map(_prep)
        test_dataset = test_ds.map(_prep)

        return train_dataset, test_dataset


def build_config(
    model_name: str,
    reward_model_name: str,
    reward_model_path: str,
    train_data_path: str,
    test_data_path: str,
    log_path: str,
    max_length: int,
    learning_rate: float,
    batch_size: int,
    group_size: int,
    eval_every: int,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> train.Config:
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    comparison_dataset_builder = PrometheusEvalDatasetBuilderFromJSONL(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
    )
    builder = PrometheusEvalPairwisePreferenceRLDatasetBuilder(
        comparison_dataset_builder=comparison_dataset_builder,
        batch_size=batch_size,
        rm_renderer_name=model_info.get_recommended_renderer_name(reward_model_name),
        rm_model_name_for_tokenizer=reward_model_name,
        rm_model_path=reward_model_path,
        group_size=group_size,
        tournament_pattern=TournamentPattern.ALL_PAIRS_BOTH_WAYS,
    )

    def winrate_eval_builder():
        return WinRateVsBaseOnTestEvaluator(
            comparison_dataset_builder=comparison_dataset_builder,
            policy_renderer_name=renderer_name,
            policy_tokenizer_model_name=model_name,
            reward_renderer_name=model_info.get_recommended_renderer_name(reward_model_name),
            reward_tokenizer_model_name=reward_model_name,
            reward_model_path=reward_model_path,
            base_model_name=model_name,
            max_tokens=max_length,
        )

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=learning_rate,
        max_tokens=max_length,
        eval_every=eval_every,
        evaluator_builders=[winrate_eval_builder],
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )


def main(
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    reward_model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    reward_model_path: str = "tinker://4002e868-80df-4ad7-bab6-c37606d18439/sampler_weights/final",  # TODO: add your model path here
    train_data_path: str = "results/rl_train_data.jsonl",
    test_data_path: str = "results/rl_dev_data.jsonl",
    log_path: str = "results/rl_with_rubric_rm",
    max_length: int = 1024,
    learning_rate: float = 4e-5,
    batch_size: int = 128,
    group_size: int = 4,
    eval_every: int = 5,
    wandb_project: str = "tinker_personalization",
    wandb_name: str = "rl_with_rubric_rm_qwen3_4b",
):
    load_dotenv()
    config = build_config(
        model_name=model_name,
        reward_model_name=reward_model_name,
        reward_model_path=reward_model_path,
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        log_path=log_path,
        max_length=max_length,
        learning_rate=learning_rate,
        batch_size=batch_size,
        group_size=group_size,
        eval_every=eval_every,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    chz.entrypoint(main, allow_hyphens=True)
