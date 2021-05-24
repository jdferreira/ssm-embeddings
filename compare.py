from __future__ import annotations

import argparse
import math
import typing

import owlready2
import numpy as np


OwlCollection = typing.Collection[owlready2.ThingClass]


def ancestors(concept: owlready2.ThingClass) -> set[owlready2.ThingClass]:
    # `owlready2` includes `owl:Thing` as an ancestor to everything but reports
    # no children or subclasses for `owl:Thing`. As such, because this is such a
    # badly behaved class in `owlready2`, just remove it from the list of
    # ancestors. We must therefore, and sadly, code our own `ancestors`
    # accessor...

    return concept.ancestors() - {owlready2.Thing}


class World:
    def __init__(self, ontology):
        self.ontology: owlready2.Ontology = ontology

        self.classes: list[owlready2.ThingClass] = list(ontology.classes())


class WorldBound:
    def __init__(self):
        self.world = None

    def set_world(self, world: World):
        self.world = world

        for field in vars(self).values():
            if isinstance(field, WorldBound) and vars(field).get('world', None) is not world:
                field.set_world(world)


class PhantomObject(Exception):
    pass


class ICCalculator(WorldBound):
    def __init__(self, use_cache=True):
        self.cache: dict[owlready2.ThingClass, float] | None

        self.cache = {} if use_cache else None

    def __call__(self, concept: owlready2.ThingClass) -> float:
        if self.world is None:
            raise PhantomObject()

        if self.cache is not None:
            cached = self.cache.get(concept, None)

            if cached is not None:
                return cached

        result = self.calculate(concept)

        if self.cache is not None:
            self.cache[concept] = result

        return result

    def calculate(self, concept: owlready2.ThingClass) -> float:
        raise NotImplementedError()


class ProbabilityBasedICCalculator(ICCalculator):
    def __init__(self, use_cache=True):
        super().__init__(use_cache=use_cache)

        self.probability_cache: dict[owlready2.ThingClass, float] | None
        self.probability_cache = {} if use_cache else None

    def calculate(self, concept: owlready2.ThingClass) -> float:
        return -math.log(self.probability(concept)) / -math.log(self.min_probability)

    def probability(self, concept: owlready2.ThingClass) -> float:
        if self.probability_cache is not None:
            cached = self.probability_cache.get(concept)

            if cached is not None:
                return cached

        result = self.calculate_probability(concept)

        if self.probability_cache is not None:
            self.probability_cache[concept] = result

        return result

    def calculate_probability(self, concept: owlready2.ThingClass) -> float:
        raise NotImplementedError()


class ICSeco(ProbabilityBasedICCalculator):
    def set_world(self, world: World):
        super().set_world(world)

        self.n_classes = len(self.world.classes)

        self.min_probability = 1 / self.n_classes

    def calculate_probability(self, concept: owlready2.ThingClass) -> float:
        return len(concept.descendants()) / self.n_classes

    def __repr__(self):
        return 'ICSeco()'


class ICSanchez(ProbabilityBasedICCalculator):
    def set_world(self, world: World) -> None:
        super().set_world(world)

        self.n_leaves = 0

        highest_ancestor_count = 0

        for c in world.classes:
            if len(c.descendants()) == 1:
                self.n_leaves += 1

                highest_ancestor_count = max(
                    highest_ancestor_count,
                    len(c.ancestors())
                )

        self.min_probability = 1 / highest_ancestor_count

    def calculate_probability(self, concept: owlready2.ThingClass) -> float:
        leaf_descendants = sum(
            1
            for c in concept.descendants()
            if len(c.descendants()) == 1
        )

        ancestor_count = len(ancestors(concept))

        return leaf_descendants / ancestor_count / self.n_leaves

    def __repr__(self):
        return 'ICSanchez()'


class Comparer(WorldBound):
    def __call__(self, one: owlready2.ThingClass, two: owlready2.ThingClass) -> float:
        if self.world is None:
            raise PhantomObject()

        return self.compare(one, two)

    def compare(self, one: owlready2.ThingClass, two: owlready2.ThingClass) -> float:
        raise NotImplementedError()


class ICBasedComparer(Comparer):
    needs_probability = False

    def __init__(self, ic_calculator: ICCalculator):
        super().__init__()

        if self.needs_probability and not hasattr(ic_calculator, 'probability'):
            raise Exception('Needs probability')

        self.ic_calculator = ic_calculator


class Resnik(ICBasedComparer):
    def compare(self, one: owlready2.ThingClass, two: owlready2.ThingClass) -> float:
        common_ancestors = ancestors(one).intersection(ancestors(two))

        return max(self.ic_calculator(c) for c in common_ancestors) \
            if common_ancestors else 0.0

    def __repr__(self):
        return f'Resnik({self.ic_calculator!r})'


class Lin(ICBasedComparer):
    def compare(self, one: owlready2.ThingClass, two: owlready2.ThingClass) -> float:
        common_ancestors = ancestors(one).intersection(ancestors(two))

        num = 2 * max(self.ic_calculator(c) for c in common_ancestors) \
            if common_ancestors else 0.0

        den = self.ic_calculator(one) + self.ic_calculator(two)

        return num / den

    def __repr__(self):
        return f'Lin({self.ic_calculator!r})'


class JiangConrath(ICBasedComparer):
    def compare(self, one: owlready2.ThingClass, two: owlready2.ThingClass) -> float:
        common_ancestors = ancestors(one).intersection(ancestors(two))

        ic_mica = max(self.ic_calculator(c) for c in common_ancestors) \
            if common_ancestors else 0.0

        return self.ic_calculator(one) + self.ic_calculator(two) - 2 * ic_mica

    def __repr__(self):
        return f'JiangConrath({self.ic_calculator!r})'


class Rel(ICBasedComparer):
    needs_probability = True

    def compare(self, one: owlready2.ThingClass, two: owlready2.ThingClass) -> float:
        common_ancestors = ancestors(one).intersection(ancestors(two))

        if not common_ancestors:
            return 0.0

        ic_mica, mica = max(
            ((self.ic_calculator(c), c) for c in common_ancestors),
            key=lambda x: x[0]
        )

        p_mica = self.ic_calculator.probability(mica)

        num = 2 * ic_mica
        den = self.ic_calculator(one) + self.ic_calculator(two)

        return num / den * (1 - p_mica)

    def __repr__(self):
        return f'Rel({self.ic_calculator!r})'


class Faith(ICBasedComparer):
    def compare(self, one: owlready2.ThingClass, two: owlready2.ThingClass) -> float:
        common_ancestors = ancestors(one).intersection(ancestors(two))

        ic_mica = max(self.ic_calculator(c) for c in common_ancestors) \
            if common_ancestors else 0.0

        den = self.ic_calculator(one) + self.ic_calculator(two) - ic_mica

        return ic_mica / den

    def __repr__(self):
        return f'Faith({self.ic_calculator!r})'


class CMatch(Comparer):
    def compare(self, one: owlready2.ThingClass, two: owlready2.ThingClass) -> float:
        common_ancestors = ancestors(one).intersection(ancestors(two))
        union_ancestors = ancestors(one).union(ancestors(two))

        return len(common_ancestors) / len(union_ancestors)

    def __repr__(self):
        return 'CMatch()'


class SetComparer(WorldBound):
    def __call__(self, one: OwlCollection, two: OwlCollection):
        if self.world is None:
            raise PhantomObject()

        return self.compare(one, two)

    def compare(self, one: OwlCollection, two: OwlCollection) -> float:
        raise NotImplementedError()


def induced_ancestors(concepts: OwlCollection) -> set[owlready2.ThingClass]:
    return {ancestor for concept in concepts for ancestor in ancestors(concept)}


class SimUI(SetComparer):
    def compare(self, one: OwlCollection, two: OwlCollection) -> float:
        one_ancestors = induced_ancestors(one)
        two_ancestors = induced_ancestors(two)

        inter = one_ancestors.intersection(two_ancestors)
        union = one_ancestors.union(two_ancestors)

        if not union:
            return 0.0
        else:
            return len(inter) / len(union)

    def __repr__(self):
        return f'SimUI()'


class SimGIC(SetComparer):
    def __init__(self, ic_calculator: ICCalculator):
        self.ic_calculator = ic_calculator

    def compare(self, one: OwlCollection, two: OwlCollection) -> float:
        one_ancestors = induced_ancestors(one)
        two_ancestors = induced_ancestors(two)

        inter = one_ancestors.intersection(two_ancestors)
        union = one_ancestors.union(two_ancestors)

        if not union:
            return 0.0
        else:
            return sum(self.ic_calculator(c) for c in inter) / sum(self.ic_calculator(c) for c in union)

    def __repr__(self):
        return f'SimGIC({self.ic_calculator!r})'


class BMA(SetComparer):
    def __init__(self, base_comparer: Comparer):
        self.base_comparer = base_comparer

    def compare(self, one: OwlCollection, two: OwlCollection) -> float:
        if len(one) == 0 or len(two) == 0:
            return 0.0

        matrix: np.ndarray = np.zeros((len(one), len(two)))

        for i1, c1 in enumerate(one):
            for i2, c2 in enumerate(two):
                matrix[i1, i2] = self.base_comparer(c1, c2)

        return (matrix.max(axis=0).mean() + matrix.max(axis=1).mean()) / 2

    def __repr__(self):
        return f'BMA({self.base_comparer!r})'


class SimilarityComputer:
    def __init__(self, ontology: owlready2.Ontology, comparers: list[SetComparer] = None):
        if comparers is None:
            comparers = get_all_comparers()

        ontology = ontology
        world = World(ontology)

        self.comparers = comparers

        for c in self.comparers:
            c.set_world(world)

    def __call__(self, one: OwlCollection, two: OwlCollection):
        return [c(one, two) for c in self.comparers]


def get_all_comparers() -> list[SetComparer]:
    ic_seco = ICSeco()
    ic_sanchez = ICSanchez()

    return [
        BMA(Resnik(ic_seco)),
        BMA(Resnik(ic_sanchez)),
        BMA(Lin(ic_seco)),
        BMA(Lin(ic_sanchez)),
        BMA(JiangConrath(ic_seco)),
        BMA(JiangConrath(ic_sanchez)),
        BMA(Rel(ic_seco)),
        BMA(Rel(ic_sanchez)),
        BMA(Faith(ic_seco)),
        BMA(Faith(ic_sanchez)),
        BMA(CMatch()),
        SimUI(),
        SimGIC(ic_seco),
        SimGIC(ic_sanchez),
    ]
