import os
from typing import Callable, Dict, List, Optional, Union

import torch
from huggingface_hub.utils import validate_hf_hub_args

from diffusers.utils import (
    USE_PEFT_BACKEND,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    convert_unet_state_dict_to_peft,
    deprecate,
    get_adapter_name,
    get_peft_kwargs,
    is_peft_available,
    is_peft_version,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
    logging,
    scale_lora_layers,
)
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.loaders.lora_conversion_utils import (
    _convert_kohya_flux_lora_to_diffusers,
    _convert_non_diffusers_lora_to_diffusers,
    _convert_xlabs_flux_lora_to_diffusers,
    _maybe_map_sgm_blocks_to_diffusers,
)


_LOW_CPU_MEM_USAGE_DEFAULT_LORA = False
if is_torch_version(">=", "1.9.0"):
    if (
        is_peft_available()
        and is_peft_version(">=", "0.13.1")
        and is_transformers_available()
        and is_transformers_version(">", "4.45.2")
    ):
        _LOW_CPU_MEM_USAGE_DEFAULT_LORA = True


if is_transformers_available():
    from diffusers.models.lora import text_encoder_attn_modules, text_encoder_mlp_modules

logger = logging.get_logger(__name__)

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"
TRANSFORMER_NAME = "transformer"

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

def lora_state_dict(
    cls,
    pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
    return_alphas: bool = False,
    **kwargs,
):

    # Load the main state dict first which has the LoRA layers for either of
    # transformer and text encoder or both.
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", None)
    token = kwargs.pop("token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", None)
    weight_name = kwargs.pop("weight_name", None)
    use_safetensors = kwargs.pop("use_safetensors", None)

    allow_pickle = False
    if use_safetensors is None:
        use_safetensors = True
        allow_pickle = True

    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }

    state_dict = cls._fetch_state_dict(
        pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
        weight_name=weight_name,
        use_safetensors=use_safetensors,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        token=token,
        revision=revision,
        subfolder=subfolder,
        user_agent=user_agent,
        allow_pickle=allow_pickle,
    )
    is_dora_scale_present = any("dora_scale" in k for k in state_dict)
    if is_dora_scale_present:
        warn_msg = "It seems like you are using a DoRA checkpoint that is not compatible in Diffusers at the moment. So, we are going to filter out the keys associated to 'dora_scale` from the state dict. If you think this is a mistake please open an issue https://github.com/huggingface/diffusers/issues/new."
        logger.warning(warn_msg)
        state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}

    # TODO (sayakpaul): to a follow-up to clean and try to unify the conditions.
    is_kohya = any(".lora_down.weight" in k for k in state_dict)
    if is_kohya:
        state_dict = _convert_kohya_flux_lora_to_diffusers(state_dict)
        # Kohya already takes care of scaling the LoRA parameters with alpha.
        return (state_dict, None) if return_alphas else state_dict

    is_xlabs = any("processor" in k for k in state_dict)
    if is_xlabs:
        state_dict = _convert_xlabs_flux_lora_to_diffusers(state_dict)
        # xlabs doesn't use `alpha`.
        return (state_dict, None) if return_alphas else state_dict

    # For state dicts like
    # https://huggingface.co/TheLastBen/Jon_Snow_Flux_LoRA
    keys = list(state_dict.keys())
    network_alphas = {}
    for k in keys:
        if "alpha" in k:
            alpha_value = state_dict.get(k)
            if (torch.is_tensor(alpha_value) and torch.is_floating_point(alpha_value)) or isinstance(
                alpha_value, float
            ):
                network_alphas[k] = state_dict.pop(k)
            else:
                raise ValueError(
                    f"The alpha key ({k}) seems to be incorrect. If you think this error is unexpected, please open as issue."
                )

    if return_alphas:
        return state_dict, network_alphas
    else:
        return state_dict
    
pretrained_model_name_or_path_or_dict = '/ytech_m2v2_hdd/fuxiao/CogVideo/finetune/cogvideox5b-lora-single-node-r32/checkpoint-2000/pytorch_lora_weights.safetensors'
state_dict, network_alphas = lora_state_dict(pretrained_model_name_or_path_or_dict, return_alphas=True, **kwargs)

import ipdb; ipdb.set_trace()