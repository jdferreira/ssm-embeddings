from __future__ import annotations

import torch
import torch.nn.functional as F


class Embedder(torch.nn.Module):
    def __init__(
        self,
        n_concepts: int,
        embedding_dimension: int,
        embedding_layers: int,
        mixer_layers: int,
        output_dimension: int,
        dropout: float,
    ):
        super().__init__()

        self.first = torch.nn.Linear(n_concepts, embedding_dimension)

        self.embedding_layers: list[torch.nn.Linear] = []

        for i in range(embedding_layers):
            layer = torch.nn.Linear(embedding_dimension, embedding_dimension)

            self.embedding_layers.append(layer)

            setattr(self, f'embedding_layer_{i}', layer)

        self.mixer_layers: list[torch.nn.Linear] = []

        for i in range(mixer_layers):
            layer = torch.nn.Linear(2 * embedding_dimension, 2 * embedding_dimension)

            self.mixer_layers.append(layer)

            setattr(self, f'mixer_layer_{i}', layer)

        self.last = torch.nn.Linear(2 * embedding_dimension, output_dimension)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, one: torch.Tensor, two: torch.Tensor) -> torch.Tensor:
        one = self.embed(one)
        two = self.embed(two)

        mixed = self.mix(one, two)

        return self.last(mixed)

    def embed(self, t: torch.Tensor, return_intermediate=False) -> torch.Tensor | list[torch.Tensor]:
        if return_intermediate:
            result: list[torch.Tensor] = []

        t = t.float()

        for layer in [self.first] + self.embedding_layers:
            t = layer(t)

            if return_intermediate:
                result.append(t)

            t = self.dropout(t)
            t = F.relu(t)

        if return_intermediate:
            return result
        else:
            return t

    def mix(self, one: torch.Tensor, two: torch.Tensor) -> torch.Tensor:
        mixed = torch.cat((one, two), dim=1)

        for layer in self.mixer_layers:
            mixed = layer(mixed)
            mixed = self.dropout(mixed)
            mixed = F.relu(mixed)

        return mixed
