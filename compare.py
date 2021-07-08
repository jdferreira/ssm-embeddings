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

    def get_children_of(self, concept: owlready2.ThingClass) -> list[owlready2.ThingClass]:
        return [
            c
            for c in self.ontology.get_children_of(concept)
            if isinstance(c, owlready2.ThingClass)
        ]

    def get_parents_of(self, concept: owlready2.ThingClass) -> list[owlready2.ThingClass]:
        return [
            c
            for c in self.ontology.get_parents_of(concept)
            if isinstance(c, owlready2.ThingClass)
        ]


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
                    len(ancestors(c))
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


class ICDepth(ICCalculator):
    def __init__(self, min_max):
        super().__init__(True)

        self.min_max = min_max

    def set_world(self, world: World):
        super().set_world(world)

        # We need to to go through all concepts. Let's find the roots and go our
        # way down
        todo = [
            c
            for c in self.world.classes
            if len(ancestors(c)) == 1 and len(self.world.get_children_of(c)) > 1
        ]

        self.depths = {
            c: 1
            for c in todo
        }

        while todo:
            c = todo.pop(0)

            children_depth = self.depths[c] + 1

            for child in self.world.get_children_of(c):
                if (
                    child not in self.depths or
                    (self.min_max == 'min' and self.depths[child] > children_depth) or
                    (self.min_max == 'max' and self.depths[child] < children_depth)
                ):
                    self.depths[child] = children_depth

                    todo.append(child)

        self.max_depth = max(self.depths.values())

    def calculate(self, concept: owlready2.ThingClass) -> float:
        return self.depths.get(concept, 0) / self.max_depth

    def __repr__(self):
        return f'ICDepth({self.min_max})'


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

        return num / den if den != 0 else 0

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

        return num / den * (1 - p_mica) if den != 0 else 0

    def __repr__(self):
        return f'Rel({self.ic_calculator!r})'


class Faith(ICBasedComparer):
    def compare(self, one: owlready2.ThingClass, two: owlready2.ThingClass) -> float:
        common_ancestors = ancestors(one).intersection(ancestors(two))

        ic_mica = max(self.ic_calculator(c) for c in common_ancestors) \
            if common_ancestors else 0.0

        den = self.ic_calculator(one) + self.ic_calculator(two) - ic_mica

        return ic_mica / den if den != 0 else 0

    def __repr__(self):
        return f'Faith({self.ic_calculator!r})'


class CMatch(Comparer):
    def compare(self, one: owlready2.ThingClass, two: owlready2.ThingClass) -> float:
        common_ancestors = ancestors(one).intersection(ancestors(two))
        union_ancestors = ancestors(one).union(ancestors(two))

        return len(common_ancestors) / len(union_ancestors) if union_ancestors else 0

    def __repr__(self):
        return 'CMatch()'


class SubsumerComparer(Comparer):
    def compare(self, one: owlready2.ThingClass, two: owlready2.ThingClass) -> float:
        return 1 if one in ancestors(two) or two in ancestors(one) else 0

    def __repr__(self):
        return 'SubsumerComparer()'


class ICSubsumerComparer(ICBasedComparer):
    def compare(self, one: owlready2.ThingClass, two: owlready2.ThingClass) -> float:
        if one in ancestors(two) or two in ancestors(one):
            return min(self.ic_calculator(one), self.ic_calculator(two))
        else:
            return 0

    def __repr__(self):
        return f'ICSubsumerComparer({self.ic_calculator!r})'


class AncestryPathDistanceCalculator:
    def __init__(self, world: World):
        self.world = world

        self.cache: dict[
            owlready2.ThingClass,
            dict[owlready2.ThingClass, int]
        ] = {}

    def calculate(self, concept: owlready2.ThingClass) -> dict[owlready2.ThingClass, int]:
        cached = self.cache.get(concept)

        if cached is not None:
            return cached

        parents = self.world.get_parents_of(concept)

        ancestry_of_parents = {
            parent: self.calculate(parent)
            for parent in parents
        }

        result: dict[owlready2.ThingClass, int] = {concept: 0}

        for ancestry in ancestry_of_parents.values():
            for ancestor, distance in ancestry.items():
                my_distance = distance + 1

                if ancestor in result:
                    result[ancestor] = min(result[ancestor], my_distance)
                else:
                    result[ancestor] = my_distance

        self.cache[concept] = result

        return result


class PathLengthComparer(Comparer):
    def set_world(self, world: World):
        super().set_world(world)

        self.ancestry_path_distance_calculator = AncestryPathDistanceCalculator(
            world
        )

    def compare(self, one: owlready2.ThingClass, two: owlready2.ThingClass) -> float:
        distances_one = self.ancestry_path_distance_calculator.calculate(one)
        distances_two = self.ancestry_path_distance_calculator.calculate(two)

        common = set(distances_one) & set(distances_two)

        if not common:
            return -1 # This should instead be a very large value
        else:
            return min(
                distances_one[c] + distances_two[c]
                for c in common
            )


    def __repr__(self):
        return 'PathLengthComparer()'


class SetComparer(WorldBound):
    def __call__(self, one: OwlCollection, two: OwlCollection):
        if self.world is None:
            raise PhantomObject()

        return self.compare(one, two)

    def compare(self, one: OwlCollection, two: OwlCollection) -> float:
        raise NotImplementedError()


def induced_ancestors(concepts: OwlCollection) -> set[owlready2.ThingClass]:
    return {ancestor for concept in concepts for ancestor in ancestors(concept)}


def induced_descendants(concepts: OwlCollection) -> set[owlready2.ThingClass]:
    return {descendant for concept in concepts for descendant in concept.descendants()}


class SimUI(SetComparer):
    def compare(self, one: OwlCollection, two: OwlCollection) -> float:
        one_ancestors = induced_ancestors(one)
        two_ancestors = induced_ancestors(two)

        inter = one_ancestors.intersection(two_ancestors)
        union = one_ancestors.union(two_ancestors)

        return len(inter) / len(union) if union else 0

    def __repr__(self):
        return f'SimUI()'


class ReverseSimUI(SetComparer):
    def compare(self, one: OwlCollection, two: OwlCollection) -> float:
        one_descendants = induced_descendants(one)
        two_descendants = induced_descendants(two)

        inter = one_descendants.intersection(two_descendants)
        union = one_descendants.union(two_descendants)

        return len(inter) / len(union) if union else 0

    def __repr__(self):
        return f'ReverseSimUI()'


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


class ReverseSimGIC(SetComparer):
    def __init__(self, ic_calculator: ICCalculator):
        self.ic_calculator = ic_calculator

    def compare(self, one: OwlCollection, two: OwlCollection) -> float:
        one_descendants = induced_descendants(one)
        two_descendants = induced_descendants(two)

        inter = one_descendants.intersection(two_descendants)
        union = one_descendants.union(two_descendants)

        if not union:
            return 0.0
        else:
            return sum(self.ic_calculator(c) for c in inter) / sum(self.ic_calculator(c) for c in union)

    def __repr__(self):
        return f'ReverseSimGIC({self.ic_calculator!r})'


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
    ic_depth_min = ICDepth('min')
    ic_depth_max = ICDepth('max')

    return [
        BMA(Resnik(ic_seco)),
        BMA(Resnik(ic_sanchez)),
        BMA(Resnik(ic_depth_min)),
        BMA(Resnik(ic_depth_max)),
        BMA(Lin(ic_seco)),
        BMA(Lin(ic_sanchez)),
        BMA(Lin(ic_depth_min)),
        BMA(Lin(ic_depth_max)),
        BMA(JiangConrath(ic_seco)),
        BMA(JiangConrath(ic_sanchez)),
        BMA(JiangConrath(ic_depth_min)),
        BMA(JiangConrath(ic_depth_max)),
        BMA(Rel(ic_seco)),
        BMA(Rel(ic_sanchez)),
        BMA(Faith(ic_seco)),
        BMA(Faith(ic_sanchez)),
        BMA(Faith(ic_depth_min)),
        BMA(Faith(ic_depth_max)),
        BMA(CMatch()),
        BMA(SubsumerComparer()),
        BMA(ICSubsumerComparer(ic_seco)),
        BMA(ICSubsumerComparer(ic_sanchez)),
        BMA(ICSubsumerComparer(ic_depth_min)),
        BMA(ICSubsumerComparer(ic_depth_max)),
        BMA(PathLengthComparer()),
        SimUI(),
        SimGIC(ic_seco),
        SimGIC(ic_sanchez),
        SimGIC(ic_depth_min),
        SimGIC(ic_depth_max),
        ReverseSimUI(),
        ReverseSimGIC(ic_seco),
        ReverseSimGIC(ic_sanchez),
        ReverseSimGIC(ic_depth_min),
        ReverseSimGIC(ic_depth_max),
    ]
