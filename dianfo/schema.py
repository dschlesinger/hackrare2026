from dataclasses import dataclass
from typing import List, Literal

@dataclass
class Gene:
    name: str
    diseases: List['Disease']

    def __hash__(self) -> str:
        return hash(self.name)

@dataclass
class Phenotype:
    name: str
    diseases: List['DiseasePhenotypeRelation']

    def __hash__(self) -> str:
        return hash(self.name)

@dataclass
class DiseasePhenotypeRelation:
    phenotype: Phenotype
    disease: 'Disease'
    relation: Literal['disease_phenotype_negative', 'disease_phenotype_positive']

    def __hash__(self) -> str:
        return hash(self.name)

@dataclass
class Disease:
    name: str
    genes: List[Gene]
    phenotypes: List[DiseasePhenotypeRelation]

    def __hash__(self) -> str:
        return hash(self.name)