from __future__ import annotations

import argparse
from typing import Generator

from utils import read_ontology_from_sqlite, replace_extension


class Vocabulary:
    def __init__(self, iris: Generator[str]):
        self.iris = sorted(iris)

        self.to_idx = {
            iri: idx
            for idx, iri in enumerate(self.iris)
        }

    def get_idx(self, iri: str):
        return self.to_idx.get(iri)

    def save(self, filename: str) -> None:
        with open(filename, 'w') as f:
            print('\n'.join(self.iris), file=f)

    def __len__(self) -> int:
        return len(self.iris)

    @classmethod
    def from_file(cls, filename: str) -> Vocabulary:
        with open(filename) as f:
            return cls([line.rstrip('\n') for line in f])


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Builds a vocabulary of the classes in an ontology.'
    )

    parser.add_argument(
        'ontology', metavar='ONTOLOGY',
        help='The filename pointing at the SQLITE version of the ontology. See `to_sqlite.py`.'
    )

    parser.add_argument(
        '-o', '--output',
        help='The filename to store the vocabulary. If not provided, a default will be '
             'generated based on the input filename, where the extension is replaced with '
             '`.vocab`.'
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = replace_extension(args.ontology, 'vocab')

    return args


def main():
    args = get_arguments()

    ontology = read_ontology_from_sqlite(args.ontology)

    Vocabulary(c.iri for c in ontology.classes()).save(args.output)


if __name__ == '__main__':
    main()
