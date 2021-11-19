from __future__ import annotations

import argparse
import multiprocessing as mp
import random

import owlready2
from tqdm.auto import trange, tqdm

from compare import SetComparer, SimilarityComputer, get_all_comparers
from utils import read_ontology_from_sqlite, replace_extension


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Generate a random dataset where each instance is a pair of entities annotated '
                    'with ontology concepts and various semantic similarity values computed for '
                    'that pair of entities.'
    )

    parser.add_argument(
        'ontology', metavar='ONTOLOGY',
        help='The filename pointing at the SQLITE version of the ontology. See `to_sqlite.py`.'
    )

    parser.add_argument(
        'entities', metavar='ENTITIES',
        help='The filename pointing at a file containing the annotations of each entity. This '
             'must be a text file with each line containing the name of an entity, a tab '
             'character, and a space-separated list of IRIs representing the concepts with '
             'which the entity is annotated with.'
    )

    parser.add_argument(
        '-c', '--comparers',
        help='The filename pointing at a file whose lines are used to construct the comparers '
             'with which to compare the entities. Notice that the file must use python syntax, '
             'and must make use of the classes and functions provided in `compare.py`. For an '
             'example of a valid file to use here, see `comparers.example`. If not provided, all '
             'known comparers are used. **WARNING** You should only pass to this flag a trusted '
             'file. There are ways to convince python to execute malicious code from this vector! '
             'Do take care to only run lines of code from a file you trust. Know that malicous '
             'code used with this flag could in theory segfault the runtime or do any number of '
             'undesireable things. When in doubt, do not use this flag.'
    )

    parser.add_argument(
        '-o', '--output',
        help='The filename to store the dataset. If not provided, a default will be '
             'generated based on the input filename, where the extension is replaced with '
             '`.dataset`. This file contains a header that represents the name of the columns, '
             'and each subsequent line contains the name of an entity, followed by a tab '
             'character, followed by the name of a second entity, another tab, and a '
             'space-deparated list of floating point numbers containing the similarity values '
             'calculated with the comparers, in the order provided with the `--comparers` flag.'
    )

    parser.add_argument(
        '-n', '--size', type=int, default=10,
        help='The number of pairs of entities to generate.'
    )

    parser.add_argument(
        '-j', '--cores', type=int, default=1,
        help='The number of cores to use by the script. By default, only one core is used.'
    )

    parser.add_argument(
        '-s', '--seed', type=int, default=42,
        help='The seed used to initialize the random number generator. Defaults to 42.'
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = replace_extension(args.entities, 'dataset')

    if args.comparers is None:
        args.comparers = get_all_comparers()
    else:
        args.comparers = make_comparers(args.comparers)

    return args


def make_comparers(filename: str):
    # Setup a local python environment where nothing exists except the classes
    # in `compare.py`.

    # NOTICE: This is inherently unsafe! To make things a little less dangerous,
    # I'm disabling direct access to python builtins, but malicous code
    # introduced here could in theory segfault the runtime or do any number of
    # undesireable things.

    environ = {'__builtins__': {}}

    exec('from compare import *', environ)

    with open(filename) as f:
        return [eval(line.rstrip('\n'), environ) for line in f]


def read_entities(filename: str, ontology: owlready2.Ontology):
    entities: dict[str, list[owlready2.ThingClass]] = {}

    with open(filename) as f_input:
        for line in f_input:
            entity, iris = line.rstrip('\n').split('\t')
            concepts = [ontology.world[iri] for iri in iris.split(' ')]
            concepts = [c for c in concepts if c is not None]
            entities[entity] = concepts

    return entities


def print_header(comparers: list[SetComparer], f):
    header = [
        'Entity 1',
        'Entity 2',
        *[repr(i) for i in comparers],
    ]

    f.write('\t'.join(header) + '\n')


def format_instance(name1: str, name2: str, sims: list[float]):
    sims = '\t'.join(str(i) for i in sims)

    return f'{name1}\t{name2}\t{sims}'


class Generator:
    def __init__(self, args, idx=0):
        ontology = read_ontology_from_sqlite(args.ontology)

        self.similarity_computer = SimilarityComputer(
            ontology,
            args.comparers
        )

        self.entities = read_entities(args.entities, ontology)
        self.entity_names = list(self.entities)

        self.random = random.Random(args.seed + idx)

    def generate(self):
        name1, name2 = self.random.sample(self.entity_names, 2)

        sims = self.similarity_computer(
            self.entities[name1], self.entities[name2])

        return name1, name2, sims

    def run(self, queue: mp.Queue):
        # The queue is the way to return the result back to the main process

        while True:
            queue.put(self.generate())


def run_generator(args, queue, idx) -> mp.Process:
    generator = Generator(args, idx)

    generator.run(queue)


def main():
    args = get_arguments()

    with open(args.output, 'w') as f:
        print_header(args.comparers, f)

        if args.cores == 1:
            generator = Generator(args)

            for _ in trange(args.size, smoothing=0):
                name1, name2, sims = generator.generate()

                print(format_instance(name1, name2, sims), file=f)
        else:
            queue = mp.Queue()

            processes = [
                mp.Process(target=run_generator, args=(args, queue, i))
                for i in range(args.cores)
            ]

            for p in processes:
                p.start()

            bar = tqdm(total=args.size, smoothing=0)

            while True:
                name1, name2, sims = queue.get()

                print(format_instance(name1, name2, sims), file=f)

                bar.update()

                if bar.n == args.size:
                    for p in processes:
                        p.terminate()
                    break


if __name__ == '__main__':
    main()
