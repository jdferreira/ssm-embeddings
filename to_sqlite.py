from __future__ import annotations

import argparse

import owlready2

from utils import replace_extension


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Read, parse and store an OWL ontology into the owlready2 SQLITE format. '
                    'This enables opening the ontology faster, as no XML parsing needs to occur.'
    )

    parser.add_argument(
        'ontology', metavar='ONTOLOGY',
        help='The filename pointing at the OWL ontology.'
    )

    parser.add_argument(
        '-o', '--output',
        help='The filename to store the SQLITE database. If not provided, a default will be '
             'generated based on the input filename, where the extension is replaced with '
             '`.sqlite`.'
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = replace_extension(args.ontology, 'sqlite')

    return args


def main():
    args = get_arguments()

    ontology = owlready2.get_ontology(args.ontology).load()

    ontology.world.set_backend(filename=args.output)


if __name__ == '__main__':
    main()
