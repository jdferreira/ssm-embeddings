from __future__ import annotations

import argparse
import gzip

from tqdm.auto import tqdm

from utils import replace_extension


def read_string_db(filename: str) -> dict[str, str]:
    """
    Returns a dictionary that associates a ENSP code with one of the many
    Uniprot names correpsonding to the protein.
    """

    result = {}

    with gzip.open(filename, 'rt') as f:
        next(f)  # Skip the first line, it contains only a header

        for line in f:
            ensp, uniprot_name = line.rstrip('\n').split('\t')[:2]

            result[uniprot_name] = ensp

    return result


def read_uniprot(filename: str):
    """
    For each protein, associate each name with the set of GO term annotations of
    the protein. Notice that proteins have usually multiple valid unique names;
    this function returns a dictionary that points each name to the list of
    annotations
    """

    current_gene_names: list[str] = []
    current_annotations: list[str] = []

    with gzip.open(filename, 'rt') as f:
        for line in f:
            op, rest = line[:4].strip(), line[5:].rstrip('\n')

            if op == '//':
                if current_gene_names and current_annotations:
                    yield from ((name, current_annotations) for name in current_gene_names)

                current_gene_names = []
                current_annotations = []

            elif op == 'GN':
                if rest != 'and':
                    current_gene_names.extend(extract_names(rest))

            elif op == 'DR':
                fields = rest.split(';')

                if fields[0] == 'GO':
                    go_id = fields[1].split(':')[1].strip()

                    current_annotations.append(
                        f'http://purl.obolibrary.org/obo/GO_{go_id}'
                    )


def extract_names(line: str):
    for field in line.split(';'):
        field = field.strip()

        if not field:
            continue

        names = field.split('=', 1)[-1].split(',')

        for name in names:
            if '{' in name:
                name = name[:name.find('{')].strip()

            yield name


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Merge String DB and Uniprot data, generating a `.annotations` file that '
                    'described the GO annotations of String DB proteins'
    )

    parser.add_argument(
        'string_db', metavar='STRING_DB',
        help='The filename pointing at the String DB information on the proteins.'
    )

    parser.add_argument(
        'uniprot', metavar='UNIPROT',
        help='The filename pointing at the Uniprot information on the proteins.'
    )

    parser.add_argument(
        '-o', '--output',
        help='The filename to store the annotations. If not provided, a default will be '
             'generated based on the STRING_DB filename, where the extension is replaced with '
             '`.annotations`.'
    )

    args = parser.parse_args()

    if args.output is None:
        if args.string_db.endswith('.txt.gz'):
            args.output = args.string_db[:-6] + 'annotations'
        else:
            args.output = replace_extension(args.string_db, 'annotations')

    return args


def main():
    args = get_arguments()

    string_db = read_string_db(args.string_db)

    with open(args.output, 'w') as f:
        for uniprot_name, annotations in tqdm(read_uniprot(args.uniprot)):
            ensp = string_db.get(uniprot_name)

            if ensp is not None:
                annotations = ' '.join(annotations)

                print(f'{ensp}\t{annotations}', file=f)


if __name__ == '__main__':
    main()
