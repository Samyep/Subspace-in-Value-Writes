from __future__ import annotations

import pytest
from torch import nn

from vw_rag.model_loading import truncate_decoder_layers


class DummyConfig:
    def __init__(self, count: int):
        self.num_hidden_layers = count


class DummyDecoder(nn.Module):
    def __init__(self, count: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(1, 1) for _ in range(count))
        self.config = DummyConfig(count)


class DummyModel(nn.Module):
    def __init__(self, count: int):
        super().__init__()
        self.model = DummyDecoder(count)
        self.config = DummyConfig(count)


def test_truncate_decoder_layers_keeps_exact_prefix_and_updates_configs():
    model = DummyModel(7)
    original_layers = list(model.model.layers)

    original_count = truncate_decoder_layers(model, 4)

    assert original_count == 7
    assert list(model.model.layers) == original_layers[:4]
    assert model.config.num_hidden_layers == 4
    assert model.model.config.num_hidden_layers == 4


@pytest.mark.parametrize("keep_layers", [0, 8])
def test_truncate_decoder_layers_rejects_invalid_prefix(keep_layers: int):
    with pytest.raises(ValueError, match="keep_layers"):
        truncate_decoder_layers(DummyModel(7), keep_layers)
