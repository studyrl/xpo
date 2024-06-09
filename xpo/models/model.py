import importlib
from functools import partial
from logging import getLogger
from typing import Optional

import deepspeed
import torch
import torch.distributed as dist
from transformers import AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES

logger = getLogger(__name__)


def create_sequence_classification_model(
    pretrained_model_name_or_path: str,
    model_type: str,
    dtype: torch.dtype,
    normalize_reward: bool = False,
    use_flash_attention_2: bool = False,
    deepspeed_config: dict = None,
    init_value_head: bool = False,
    device_map: dict = None,
    **kwargs,
):
    assert model_type in ["reward", "critic"], "'model_type' should be either 'reward' or 'critic'."
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
    )
    config.normalize_rewards = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    if deepspeed_config is not None and deepspeed_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(deepspeed_config)
    else:
        dschf = None

    model_architecture = config.model_type
    if model_architecture in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES:
        model_class_name = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES[model_architecture]
    else:
        raise ValueError(f"Can not create {model_type} model for '{model_architecture}'.")

    model_class_module_name = f"transformers.models.{model_architecture}.modeling_{model_architecture}"
    model_class_module = importlib.import_module(model_class_module_name)
    model_class_obj = getattr(model_class_module, model_class_name)
    model = model_class_obj.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
        num_labels=1,
        **kwargs,
    )

    if normalize_reward:
        model.register_buffer("mean", torch.zeros(1, dtype=dtype), persistent=False)
        model.register_buffer("std", torch.ones(1, dtype=dtype), persistent=False)
        setattr(model, "normalize_reward", normalize_reward)
        if hasattr(config, "mean"):
            model.mean[0] = config.mean
            model.std[0] = config.std

    if init_value_head:
        value_head = get_value_head(model)

        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                if dist.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
    
    if model_type == "reward":
        model.forward = partial(forward_reward_model, self=model)
    else:
        model.forward = partial(forward_critic_model, self=model)


def get_value_head(model):
    for key, value in model.named_children():
        if isinstance(value, torch.nn.Linear) and value.out_features == 1:
            return value
    return None


def forward_reward_model(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    return_output: bool = False,
):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    base_model = getattr(self, self.base_model_prefix)
    value_head = get_value_head(self)

    if value_head is None:
        raise ValueError("Value head not found in the model.")

    outputs = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    last_hidden_states = outputs["last_hidden_state"]
    # [bsz, seq_len, hidden_size]
    values = self.value_head(last_hidden_states).squeeze(-1)
    # [bsz, seq_len]

    # left padding in training mode
    if self.training:
        # use only the last token's value as reward including eos token
        reward = values[:, -1]
    else:
        # use only the last token's value as reward excluding eos token
        eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
        reward = values.gather(dim=1, index=eos_indices).squeeze(1)

        if self.normalize_reward:
            # normalize reward in eval mode
            reward = (reward - self.mean) / self.std

    if return_output:
        return reward, outputs
    else:
        return reward


def forward_critic_model(
    self,
    input_ids: torch.LongTensor = None,
    action_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    return_output=False,
):
    pass
