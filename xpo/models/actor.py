from typing import Tuple, Optional

import torch
from transformers import AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig

from .utils import log_probs_from_logits


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
        do_sample: bool = True,
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
            input_len=input_ids.size(1),
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        return sequences, attention_mask, action_mask

    def postprocess_sequences(
        self,
        sequences: torch.Tensor,
        input_len: int,
        eos_token_id: int,
        pad_token_id: int,
        padding_side: str = "right",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Postprocess generated sequences.

        Args:
            sequences (torch.Tensor): generated token ids
            input_len (int): input sequence length
            eos_token_id (int): end of sentence token id
            pad_token_id (int): padding token id
            padding_side (str): padding side, only supports "right"

        Returns:
            sequences: torch.Tensor: generated token ids
            attention_mask: torch.Tensor: attention mask
            action_mask: torch.Tensor: action mask

        Notes:
            If input sequence is 'Hello, my name is Kevin<|endoftext|>' and the model generates 'Hi. I am an AI assistant.'.
            Then, the total sequence will be 'Hello, my name is Kevin<|endoftext|>Hi. I am an AI assistant.<|endoftext|><|endoftext|><|endoftext|>'.
            The example above is the case when padding_side is 'right' and the 3 padding tokens are added to the right side.

            In this case, the original attention mask was [1, 1, ..., 0, 0, 0] and the postprocessed attention mask should be [1, 1, ..., 1, 0, 0].
            The firstly appeared padding token should be transformed to 1 and the rest of the padding tokens should be 0.
            This means this will train the firstly appeared padding token and ignore the rest of the padding tokens.

        References:
            https://github.com/OpenLLMAI/OpenRLHF/blob/2cbc9416731a09880df1d5ae8de71e23c4d272b3/openrlhf/models/actor.py#L138
        """

        assert padding_side == "right", "Only right padding is supported for now."

        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        for i in range(attention_mask.size(0)):
            for t in reversed(range(seq_length)):
                if attention_mask[i][t] > 0.5:
                    attention_mask[i][min(t + 1, seq_length - 1)] = True
                    sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
                    break

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1: -1]
        # we only calculate the loss of state_i != eos | pad
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        return sequences, attention_mask, action_mask

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
    ) -> torch.Tensor:
        """

        Args:
            sequences (torch.LongTensor): generated token ids
            num_actions (int): number of actions
            attention_mask (torch.Tensor): attention mask
            return_output (bool): whether to return an output tensor

        Returns:
            torch.Tensor: output tokens, if return_output is True
            torch.Tensor: log probabilities, if return_output is False

        Notes:
            Position token ids may be different by using padding side as "left" or "right".
            So we need to adjust this accurately regardless of padding side.

            if padding side is "right":
                attention_mask: [1, 1, 1, 1, 0, 0, 0, 0]
                position_ids before masked_fill_: [0, 1, 2, 3, 3, 3, 3, 3]
                position_ids after masked_fill_: [0, 1, 2, 3, 1, 1, 1, 1]

            if padding side is "left":
                attention_mask: [0, 0, 0, 0, 1, 2, 3, 4]
                position_ids before masked_fill_: [-1, -1, -1, -1, 0, 1, 2, 3]
                position_ids after masked_fill_: [1, 1, 1, 1, 0, 1, 2, 3]

            Since we don't train tokens which have attention_mask are 0,
            the position ids of such tokens (=1) are ignored.

        References:
            https://github.com/OpenLLMAI/OpenRLHF/issues/217
        """
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)
        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if return_output:
            return output if num_actions is None else (log_probs[:, -num_actions:], output)
        else:
            return log_probs[:, -num_actions:]

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
