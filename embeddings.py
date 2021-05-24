from __future__ import annotations

import argparse
import os
import json

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

from utils import read_one_hot_encodings, to_dense
from modelling import Embedder


def load_model(config, device: str) -> Embedder:
    model = Embedder(
        n_concepts=config['n_concepts'],
        embedding_dimension=config['embedding_dimension'],
        embedding_layers=config['embedding_layers'],
        mixer_layers=config['mixer_layers'],
        output_dimension=config['n_similarities'],
        dropout=config['dropout'],
    )

    with open(config['weights_filename'], 'br') as f:
        weights = torch.load(f)

    model.load_state_dict(weights)

    model.eval()

    model.to(device)

    return model


def read_config(dirname: str):
    with open(os.path.join(dirname, 'config.json')) as f:
        return json.load(f)


@torch.no_grad()
def embed(indices: list[int], n_concepts: int, embedder: Embedder) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []

    tensor = to_dense(indices, n_concepts).float()

    for layer in [embedder.first] + embedder.layers:
        tensor = layer(tensor)

        # Append this embedding before the relu transformation
        result.append(tensor)

        tensor = F.relu(tensor)

    return result


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Embed a set of entities (already one-hot encoded) using a model trained to '
                    'predict semantic similarity.'
    )

    parser.add_argument(
        'dirname', metavar='DIRNAME',
        help='The directory where the output of training was stored.'
    )

    parser.add_argument(
        'encodings', metavar='ENCODINGS',
        help='The filename pointing at the one-hot encoding of the entities being compared.'
    )

    parser.add_argument(
        '-D', '--device', default='cpu',
        help='The device where the model will operate on. Defaults to cpu. Must be a string that '
             'torch recognizes as a valid device.'
    )

    return parser.parse_args()


def main():
    args = get_arguments()

    # Read the saved configuration parameters
    config = read_config(args.dirname)

    # Read a .ohe file
    entities = read_one_hot_encodings(args.encodings)

    # Load the model from the model.pt file (and other configuration parameters)
    embedder = load_model(config, args.device)

    # Read some configuration parameters
    n_concepts = config['n_concepts']
    embedding_layers = config['embedding_layers']

    # Embed the entities with the embedder's embedding layers
    with torch.no_grad():
        embeddings = {
            entity: embedder.embed(to_dense(indices, n_concepts), return_intermediate=True)
            for entity, indices in tqdm(entities.items())
        }

    # Save the several files (one for each hidden layer of the embedder)
    for layer in trange(embedding_layers):
        with open(os.path.join(args.dirname, f'embeddings.{layer}.txt'), 'w') as f:
            for entity, entity_embeddings in embeddings.items():
                f.write(entity)
                f.write(' - ')
                f.write(repr(entity_embeddings[layer].tolist()))
                f.write('\n')


if __name__ == '__main__':
    main()
