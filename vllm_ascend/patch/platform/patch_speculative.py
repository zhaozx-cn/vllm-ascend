
from vllm.config.speculative import SpeculativeConfig


def _verify_args(self) -> Self:
    if self.num_speculative_tokens is None:
        raise ValueError(
            "num_speculative_tokens must be provided with "
            "speculative model unless the draft model config contains an "
            "n_predict parameter."
        )

    if self.num_speculative_tokens <= 0:
        raise ValueError(
            "Expected num_speculative_tokens to be greater "
            f"than zero ({self.num_speculative_tokens})."
        )

    if self.draft_model_config:
        self.draft_model_config.verify_with_parallel_config(
            self.draft_parallel_config
        )

    if self.disable_by_batch_size is not None and self.disable_by_batch_size < 2:
        raise ValueError(
            "Expect the batch size threshold of disabling "
            "speculative decoding is > 1, but got "
            f"{self.disable_by_batch_size=}"
        )
    
    eagle3_target_supported = ["llama", "qwen", "minicpm", "gpt_oss", "deepseek_v3"]
    if (
        self.method == "eagle3"
        and self.target_model_config
        and not any(
            supported_model in self.target_model_config.hf_text_config.model_type
            for supported_model in eagle3_target_supported
        )
    ):
        raise ValueError(
            f"Eagle3 is only supported for {eagle3_target_supported} models. "  # noqa: E501
            f"Got {self.target_model_config.hf_text_config.model_type=}"
        )

    return self

SpeculativeConfig._verify_args = _verify_args