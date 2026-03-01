from dataclasses import dataclass
from typing import List, Literal

@dataclass
class Gene:
    name: str
    diseases: List['Disease']

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __repr__(self) -> str:
        return self.name

@dataclass
class Phenotype:
    name: str
    diseases: List['DiseasePhenotypeRelation']

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __repr__(self) -> str:
        return self.name
    
@dataclass
class DiseasePhenotypeRelation:
    phenotype: Phenotype
    disease: 'Disease'
    relation: Literal['disease_phenotype_negative', 'disease_phenotype_positive']

@dataclass
class Disease:
    name: str
    genes: List[Gene]
    phenotypes: List[DiseasePhenotypeRelation]

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __repr__(self) -> str:
        return self.name