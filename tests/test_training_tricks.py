import torch
from torch import nn

from opinion_mining.training_tricks import EMAModel, FGM


class TinyEncoder(nn.Module):
    def __init__(self, vocab: int = 8, hidden: int = 4):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.word_embeddings = nn.Embedding(vocab, hidden)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.head = nn.Linear(4, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.embeddings.word_embeddings(tokens))


def test_ema_shadow_tracks_parameters_and_restores_after_apply():
    model = TinyModel()
    parameters = list(model.parameters())
    ema = EMAModel(parameters, decay=0.5)
    index = next(i for i, parameter in enumerate(parameters) if parameter is model.head.weight)
    before = model.head.weight.detach().clone()

    with torch.no_grad():
        model.head.weight.add_(1.0)
    ema.update()
    shifted = model.head.weight.detach().clone()
    assert torch.allclose(ema.shadow[index], before * 0.5 + shifted * 0.5)

    ema.apply()
    assert torch.allclose(model.head.weight, ema.shadow[index])
    ema.restore()
    assert torch.allclose(model.head.weight, shifted)


def test_fgm_perturbs_embeddings_along_gradient_and_restores():
    torch.manual_seed(0)
    model = TinyModel()
    fgm = FGM(model, epsilon=1.0)
    tokens = torch.tensor([1, 2, 3])
    loss = model(tokens).sum()
    loss.backward()
    original = model.encoder.embeddings.word_embeddings.weight.detach().clone()

    assert fgm.attack() is True
    perturbed = model.encoder.embeddings.word_embeddings.weight.detach().clone()
    assert not torch.allclose(perturbed, original)

    assert fgm.restore() is True
    assert torch.allclose(model.encoder.embeddings.word_embeddings.weight, original)


def test_fgm_without_gradients_is_a_noop():
    model = TinyModel()
    fgm = FGM(model, epsilon=1.0)

    assert fgm.attack() is False
    assert fgm.restore() is False
