from __future__ import annotations

import owlready2
import torch


def replace_extension(filename: str, new_extension: str):
    parts = filename.split('.')

    if len(parts) > 1:
        del parts[-1]

    parts.append(new_extension)

    return '.'.join(parts)


def read_ontology_from_sqlite(filename: str) -> owlready2.Ontology:
    world = owlready2.World(filename=filename)

    return next(iter(world.ontologies.values()))


def read_one_hot_encodings(filename: str):
    encodings: dict[str, list[int]] = {}

    with open(filename) as f:
        for line in f:
            name, indices = line.rstrip('\n').split('\t')

            encodings[name] = [int(i) for i in indices.split(' ')]

    return encodings


def to_dense(indices: list[int], size: int) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        indices=[list(indices)],
        values=[1 for _ in indices],
        size=(size,),
        dtype=torch.int8
    )
