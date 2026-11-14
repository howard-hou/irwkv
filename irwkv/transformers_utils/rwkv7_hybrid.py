# SPDX-License-Identifier: Apache-2.0
"""Configuration for the fixed bux timeline 3:1 hybrid family."""

from transformers import PreTrainedConfig


class Rwkv7HybridConfig(PreTrainedConfig):
    model_type = "rwkv7_hybrid"

    def __init__(
        self,
        hidden_size=3328,
        num_hidden_layers=None,
        vocab_size=50304,
        max_position_embeddings=32768,
        **kwargs,
    ):
        layers = (
            4 * (hidden_size // 256) if num_hidden_layers is None else num_hidden_layers
        )
        if hidden_size <= 0 or hidden_size % 64:
            raise ValueError("hidden_size must be a positive multiple of 64")
        if layers <= 0 or layers % 4:
            raise ValueError("num_hidden_layers must be a positive multiple of four")
        if vocab_size <= 0 or max_position_embeddings <= 0:
            raise ValueError("vocabulary and context sizes must be positive")
        expected = {
            "tie_word_embeddings": True,
            "head_size": 64,
            "num_attention_heads": hidden_size // 64,
            "num_key_value_heads": hidden_size // 64,
            "intermediate_size": 4 * hidden_size,
            "rope_theta": 10000.0,
            "rope_scaling": None,
            "hidden_act": "gelu",
            "layer_norm_epsilon": 1e-5,
            "layers_block_type": [
                "attention" if i % 4 == 3 else "mamba" for i in range(layers)
            ],
        }
        for key, value in expected.items():
            if key in kwargs and kwargs.pop(key) != value:
                raise ValueError(f"{key} must equal {value!r} for the bux hybrid")
        kwargs.setdefault("architectures", ["Rwkv7HybridForCausalLM"])
        super().__init__(tie_word_embeddings=True, **kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        for key, value in expected.items():
            setattr(self, key, value)

    def parameter_count(self):
        width, layers = self.hidden_size, self.num_hidden_layers
        recurrent = layers * 3 // 4
        attention = layers // 4
        active = layers * 12 * width**2 + (recurrent * 534 + attention * 13 + 2) * width
        return {"total": active + self.vocab_size * width, "active": active}
