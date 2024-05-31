from typing import Tuple

import torch
from transformers import AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig


class Actor(torch.nn.Module):
    """
    Actor model base class.

    Args:
        model_name (str): model name on huggingface model hub
        use_flash_attention_2 (bool): use flash attention 2 or not
        dtype (torch.dtype): torch data type, default is torch.bfloat16
        deepspeed_config (dict): deepspeed config dictionary

    References:
        OpenRLHF:
            https://github.com/OpenLLMAI/OpenRLHF/blob/main/openrlhf/models/actor.py
    """

    def __init__(
        self,
        model_name: str,
        use_flash_attention_2: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        deepspeed_config: dict = None,
    ):
        super(Actor, self).__init__()
        self.model_name = model_name
        self.use_flash_attention_2 = use_flash_attention_2
        self.dtype = dtype
        self.deepspeed_config = deepspeed_config

        attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

        # When we create hf_deepspeed_config, it affects the global variable in hf transformers.
        # We create this config even if we don't use this in the future because if the global variable changes,
        # it will affect the behavior of the model initialization.
        if deepspeed_config is not None and deepspeed_config["zero_optimization"]["stage"] == 3:
            hf_deepspeed_config = HfDeepSpeedConfig(deepspeed_config)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int,
        pad_token_id: int,
        do_sample: bool=True,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate text from the model.

        Args:
            input_ids (torch.Tensor): input token ids
            eos_token_id (int): end of sentence token id
            pad_token_id (int): padding token id
            do_sample (bool): do sampling or not

        Returns:
            sequences: torch.Tensor: generated token ids
            attention_mask: torch.Tensor: attention mask
            action_mask: torch.Tensor: action mask
        """

        min_new_tokens = kwargs.get("min_new_tokens", 1)
        sequences = self.model.generate(
            input_ids,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )
        sequences, attention_mask, action_mask = self.postprocess_sequences(
            sequences=sequences,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        return sequences, attention_mask, action_mask

    def postprocess_sequences(
        self,
        sequences: torch.Tensor,
        eos_token_id: int,
        pad_token_id: int,
        padding_side: str = "left",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Postprocess generated sequences.

        Args:
            sequences (torch.Tensor): generated token ids
            eos_token_id (int): end of sentence token id
            pad_token_id (int): padding token id

        Returns:
            sequences: torch.Tensor: generated token ids
            attention_mask: torch.Tensor: attention mask
            action_mask: torch.Tensor: action mask
        """

        assert padding_side in ["left", "right"], "padding_side should be either left or right"
        # TODO check if this is correct

