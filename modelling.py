import torch
import torch.nn.functional as F

class Embedder(torch.nn.Module):
    def __init__(self, n_concepts: int, embeddings_dimension: int, layers: int, output_dimension: int):
        super().__init__()

        self.first = torch.nn.Linear(n_concepts, embeddings_dimension)

        self.layers: list[torch.nn.Linear] = []
        for i in range(layers):
            layer = torch.nn.Linear(embeddings_dimension, embeddings_dimension)

            self.layers.append(layer)

            setattr(self, 'layer' + str(i), layer)

        self.last = torch.nn.Linear(embeddings_dimension * 2, output_dimension)

    def forward(self, one: torch.Tensor, two: torch.Tensor) -> torch.Tensor:
        one = self.embed(one)
        two = self.embed(two)

        return self.last(torch.cat((one, two), dim=1))

    def embed(self, t: torch.Tensor):
        t = t.float()

        for layer in [self.first] + self.layers:
            t = F.relu(layer(t))

        return t
