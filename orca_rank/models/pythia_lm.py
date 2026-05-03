from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class InputAdapter(nn.Module):
    """Lightweight mapper f on token embeddings."""

    def __init__(self, hidden_size: int, bottleneck: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PythiaFrontendCausalLM(nn.Module):
    """Wraps GPT-NeoX (Pythia) with optional adapter before transformer stack."""

    def __init__(
        self,
        model_name: str,
        use_adapter: bool,
        adapter_bottleneck: int = 128,
        torch_dtype=None,
        device_map: str | None = None,
    ):
        super().__init__()
        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            kwargs["device_map"] = device_map
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
        hidden = self.lm.config.hidden_size
        self.adapter = (
            InputAdapter(hidden, bottleneck=adapter_bottleneck) if use_adapter else None
        )
        if self.adapter is not None:
            try:
                lm_dtype = next(self.lm.parameters()).dtype
            except StopIteration:
                lm_dtype = torch.float32
            self.adapter.to(dtype=lm_dtype)

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if self.adapter is not None:
                inputs_embeds = self.adapter(inputs_embeds)

        out = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out

    def generate(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kw):
        if inputs_embeds is None and input_ids is not None:
            e = self.get_input_embeddings()(input_ids)
            if self.adapter is not None:
                e = self.adapter(e)
            inputs_embeds = e
            input_ids = None
        try:
            return self.lm.generate(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kw,
            )
        except TypeError:
            return self.lm.generate(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kw,
            )

    def pooled_masked_mean(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pool token embeddings produced after optional adapter."""

        inputs_embeds = self.get_input_embeddings()(input_ids)
        if self.adapter is not None:
            inputs_embeds = self.adapter(inputs_embeds)

        hs = inputs_embeds
        m = attention_mask.unsqueeze(-1).to(dtype=hs.dtype)
        summed = (hs * m).sum(dim=1)
        denom = m.sum(dim=1).clamp(min=1e-6)
        return summed / denom
