from __future__ import annotations

import owlready2


def replace_extension(filename: str, new_extension: str):
    parts = filename.split('.')

    if len(parts) > 1:
        del parts[-1]

    parts.append(new_extension)

    return '.'.join(parts)


def read_ontology_from_sqlite(filename: str) -> owlready2.Ontology:
    world = owlready2.World(filename=filename)

    return next(iter(world.ontologies.values()))


