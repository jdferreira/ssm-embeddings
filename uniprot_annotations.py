from __future__ import annotations
import argparse

import gzip
from one_hot_encode import encode_entities
from utils import replace_extension


def read_uniprot(filename: str):
    with gzip.open(filename, 'rt') as f:
        current_id = None
        current_annotations: list[str] = []

        for line in f:
            line = line.rstrip('\n')

            op, rest = line[:4].strip(), line[5:].rstrip('\n')

            if op == '//':
                if current_id and current_annotations:
                    yield (current_id, current_annotations)

                current_id = None
                current_annotations = []

            elif op == 'ID':
                current_id = rest.split()[0]

            elif op == 'DR':
                fields = rest.split(';')

                if fields[0] == 'GO':
                    go_id = fields[1].split(':')[1]

                    current_annotations.append(
                        f'http://purl.obolibrary.org/obo/GO_{go_id}'
                    )


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Read a Uniprot text file and save a file containing the GO annotations of '
                    'each protein in it.'
    )

    parser.add_argument(
        'uniprot', metavar='UNIPROT',
        help='The filename pointing at the Uniprot file describing the proteins. The file must '
             'be in the `.txt.gz` format.'
    )

    parser.add_argument(
        '-o', '--output',
        help='The filename to store the annotations output. If not provided, a default will be '
             'generated based on the input filename, where the extension is replaced with '
             '`.annotations`.'
    )

    args = parser.parse_args()

    if args.output is None:
        if args.uniprot.endswith('.txt.gz'):
            args.output = args.uniprot[:-6] + 'annotations'
        else:
            args.output = replace_extension(args.uniprot, 'annotations')

    return args


def main():
    args = get_arguments()

    with open(args.output, 'w') as f:
        for entity, iris in read_uniprot(args.uniprot):
            iris = ' '.join(iris)

            print(f'{entity}\t{iris}', file=f)


if __name__ == '__main__':
    main()
