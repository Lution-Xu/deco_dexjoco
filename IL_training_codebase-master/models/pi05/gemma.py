
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaConfig,
    GemmaForCausalLM,
    GemmaMLP,
    GemmaModel,
)
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaModel,
)


def _gated_residual(x, y, gate):
    """Gated residual: x + y when gate is None, else x + y * gate."""
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


def layernorm_forward(layernorm, x, cond=None):
    if cond is not None:
        return layernorm(x, cond=cond)
    else:
        return layernorm(x)


class PiGemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None

    def _norm(self, x):
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    def forward(self, x, cond=None):
        dtype = x.dtype
        normed = self._norm(x)
        if cond is None or self.dense is None:
            normed = normed * (1.0 + self.weight.float())
            return normed.type_as(x), None
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dim {self.cond_dim}, got {cond.shape[-1]}")
        modulation = self.dense(cond)
        if len(x.shape) == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)

    def extra_repr(self) -> str:
        if self.dense is not None:
            return f"dim={self.dim}, eps={self.eps}, adaptive=True, cond_dim={self.cond_dim}"
        return f"dim={self.dim}, eps={self.eps}"


class _PiGemmaDecoderLayerBase(GradientCheckpointingLayer):
    """Decoder layer that uses PiGemmaRMSNorm and _gated_residual, compatible with v5 Gemma."""

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        cond_dim = (
            getattr(config, "adarms_cond_dim", None) if getattr(config, "use_adarms", False) else None
        )
        self.input_layernorm = PiGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
        )
        self.post_attention_layernorm = PiGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, cond=adarms_cond)
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = _gated_residual(residual, hidden_states, gate)

        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, cond=adarms_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _gated_residual(residual, hidden_states, gate)
        return hidden_states



class PiGemmaModel(GemmaModel):  # type: ignore[misc]
    """
    GemmaModel extended with AdaRMS (adaptive RMSNorm) and gated residuals when config.use_adarms is True.
    """

    def __init__(self, config: GemmaConfig, **kwargs):
        super().__init__(config, **kwargs)
        # if not getattr(config, "use_adarms", False):
        #     return
        cond_dim = getattr(config, "adarms_cond_dim", None)
        self.layers = nn.ModuleList(
            [_PiGemmaDecoderLayerBase(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PiGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """
        adarms_cond (`torch.Tensor` of shape `(batch_size, cond_dim)`, *optional*):
            Condition for ADARMS.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            import logging

            logging.warning(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # embed positions
        hidden_states = inputs_embeds
        # Convert to bfloat16 if the first layer uses bfloat16
        if len(self.layers) > 0 and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # normalized
        # Gemma downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
                **kwargs,
            )

            hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states, _ = self.norm(hidden_states, adarms_cond)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class PiGemmaForCausalLM(GemmaForCausalLM):  # type: ignore[misc]
    def __init__(self, config: GemmaConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = PiGemmaModel(config)


class PaliGemmaModelWithPiGemma(PaliGemmaModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = PiGemmaModel(config.text_config)


class PaliGemmaForConditionalGenerationWithPiGemma(PaliGemmaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = PaliGemmaModelWithPiGemma(config)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model



def _fix_pytorch_state_dict_keys(model, state_dict):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
    """Fix state dict keys to match current model architecture."""
    import re

    fixed_state_dict = {}

    for key, value in state_dict.items():
        new_key = key
        # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
        # For gemma expert layers
        if re.match(r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight", key):
            # Check if the model actually has adaRMS enabled for the expert
            expert_uses_adarms = getattr(model.paligemma_with_expert.gemma_expert.config, "use_adarms", False)
            if expert_uses_adarms:
                print(f"Skipping layer norm key (adaRMS mismatch): {key}")
                continue

        if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
            # Check if the model actually has adaRMS enabled for the expert
            expert_uses_adarms = getattr(model.paligemma_with_expert.gemma_expert.config, "use_adarms", False)
            if expert_uses_adarms:
                print(f"Skipping norm key (adaRMS mismatch): {key}")
                continue

        # Handle MLP naming changes for pi05
        # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
        if key.startswith("action_time_mlp_in."):
            new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
        elif key.startswith("action_time_mlp_out."):
            new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
        # Also handle state_proj which shouldn't exist in pi05
        if key.startswith("state_proj."):
            print(f"Skipping state_proj key in pi05 mode: {key}")
            continue

        # Handle vision tower embedding layer potential differences
        if "patch_embedding" in key:
            # Some checkpoints might have this, but current model expects different structure
            print(f"Vision embedding key might need handling: {key}")

        if (key == "model.paligemma_with_expert.paligemma.lm_head.weight" or key == "paligemma_with_expert.paligemma.lm_head.weight"):
            fixed_state_dict["paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = value.clone()

        fixed_state_dict[new_key] = value

    return fixed_state_dict


def remap_state_dict_keys(fix_model_dict):
    remapped_state_dict = {}
    remap_count = 0

    for key, value in fix_model_dict.items():
        if key.startswith("model."):
            new_key = key[6:]
            remapped_state_dict[new_key] = value
            remap_count += 1
        else:
            remapped_state_dict[key] = value

    if remap_count > 0:
        print(f"Remapped {remap_count} state dict keys")

    return remapped_state_dict