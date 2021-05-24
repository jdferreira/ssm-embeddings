from __future__ import annotations
import argparse

import owlready2
from vocabulary import Vocabulary

from utils import read_ontology_from_sqlite, replace_extension


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Compute and save the one-hot encoding of ontology classes or entities '
                    'annotated with those classes. A concept is converted into the list of '
                    'indices of its ancestors (including itself), and an annotated entity is '
                    'converted into the list of indices present in the union of its annotations.'
    )

    parser.set_defaults(command=None)

    subparsers = parser.add_subparsers()

    class_subparser = subparsers.add_parser(
        'classes',
        help='Encode the classes of the ontology.'
    )

    class_subparser.set_defaults(command='classes')

    class_subparser.add_argument(
        'ontology', metavar='ONTOLOGY',
        help='The filename pointing at the SQLITE version of the ontology. See `to_sqlite.py`.'
    )

    class_subparser.add_argument(
        'vocabulary', metavar='VOCABULARY',
        help='The filename pointing at the vocabulary of the ontology. See `vocabulary.py`.'
    )

    class_subparser.add_argument(
        '-o', '--output',
        help='The filename to store the one hot encodings. If not provided, a default will be '
             'generated based on the ontology filename, where the extension is replaced with '
             '`.classes.ohe`.'
    )

    entity_subparser = subparsers.add_parser(
        'entities',
        help='Encode the annotated entities.'
    )

    entity_subparser.set_defaults(command='entities')

    entity_subparser.add_argument(
        'encodings', metavar='ENCODINGS',
        help='The filename pointing at the class encodings, generated by this script with the '
             '`classes` command.'
    )

    entity_subparser.add_argument(
        'entities', metavar='ENTITIES',
        help='The filename pointing at a file containing the annotations of each entity. This '
             'must be a text file with each line containing the name of an entity, a tab '
             'character, and a space-separated list of IRIs representing the concepts with '
             'which the entity is annotated with.'
    )

    entity_subparser.add_argument(
        '-o', '--output',
        help='The filename to store the one hot encodings. If not provided, a default will be '
             'generated based on the entities filename, where the extension is replaced with '
             '`.ohe`.'
    )

    args = parser.parse_args()

    if args.command is None:
        parser.error(
            'You must provide either the `classes` or `entities` sub-command.'
        )

    if args.output is None:
        if args.command == 'classes':
            args.output = replace_extension(args.ontology, 'classes.ohe')
        elif args.command == 'entities':
            args.output = replace_extension(args.entities, 'ohe')

    return args


def encode_classes(args):
    # Open the ontology
    ontology = read_ontology_from_sqlite(args.ontology)

    # Open the vocabulary
    vocabulary = Vocabulary.from_file(args.vocabulary)

    # Find class ancestors, convert to indices, and save
    with open(args.output, 'w') as f:
        for iri in vocabulary.iris:
            c: owlready2.ThingClass | None = ontology.world[iri]

            if c is None:
                continue

            # Get the list of class indices for all ancestors
            indices = [vocabulary.get_idx(a.iri) for a in c.ancestors()]

            # Filter unknown IRIs, deduplicate, and sort the indices
            indices = sorted({idx for idx in indices if idx is not None})

            # Make a comma-separated list of the indices
            indices = ' '.join(str(i) for i in indices)

            # Store as another line in the output
            print(f'{iri}\t{indices}', file=f)


def encode_entities(args):
    class_encodings = read_class_encodings(args)

    with open(args.entities) as f_input, open(args.output, 'w') as f_output:
        for line in f_input:
            entity, iris = line.rstrip('\n').split('\t')

            indices = sorted({idx for iri in iris for idx in class_encodings[iri]})

            indices = ' '.join(str(i) for i in indices)

            print(f'{entity}\t{indices}', file=f_output)

def read_class_encodings(args):
    class_encodins: dict[str, set[int]] = {}

    with open(args.encodings) as f:
        for line in f:
            iri, indices = line.rstrip('\n').split('\t')
            indices = {int(i) for i in indices.split(' ')}
            class_encodins[iri] = indices

    return class_encodins

def main():
    args = get_arguments()

    if args.command == 'classes':
        encode_classes(args)
    elif args.command == 'entities':
        encode_entities(args)


if __name__ == '__main__':
    main()
