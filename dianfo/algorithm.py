import pandas as pd
import numpy as np
from dianfo.schema import Gene, Phenotype, Disease, DiseasePhenotypeRelation
from typing import Dict, Tuple, List
from collections import Counter, defaultdict

from rich import print

# df = pd.read_csv('mini.csv', low_memory=False, index_col=0)
df = pd.read_csv('gene_pheno_disease.csv', low_memory=False, index_col=0)

# Initalize lookup tables
def assemble_lookup_tables(df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:

    # Gene to diseases
    gene_lookup = {}

    # phenotypes to diseases
    phenotype_lookup = {}

    # disease name to disease
    disease_lookup = {}

    for _, row in df.iterrows():

        if row['y_type'] == 'gene/protein':

            gene = row['y_name']
            disease = row['x_name']

            if gene not in gene_lookup:
                g = Gene(gene, [])
                gene_lookup[gene] = g
            else:
                g = gene_lookup[gene]

            if disease not in disease_lookup:
                d = Disease(disease, [], [])
                disease_lookup[disease] = d
            else:
                d = disease_lookup[disease]
            
            g.diseases.append(d)
            d.genes.append(g)

        # disease -> phenotype
        elif row['y_type'] == 'effect/phenotype':

            disease = row['x_name']
            phenotype = row['y_name']

            if phenotype not in phenotype_lookup:
                p = Phenotype(phenotype, [])
                phenotype_lookup[phenotype] = p
            else:
                p = phenotype_lookup[phenotype]

            if disease not in disease_lookup:
                d = Disease(disease, [], [])
                disease_lookup[disease] = d
            else:
                d = disease_lookup[disease]

            ptd = DiseasePhenotypeRelation(p, d, row['relation'])

            p.diseases.append(ptd)
            d.phenotypes.append(ptd)

    return gene_lookup, phenotype_lookup, disease_lookup

gene_lookup, phenotype_lookup, disease_lookup = assemble_lookup_tables(df)
# DISEASE_ORDER = list(disease_lookup.keys())

def get_disease_probs(gene_tests, pheno_tests) -> Dict:

    patient_diseases: Dict = defaultdict(lambda: 0)

    for g in gene_tests:
        for d in gene_lookup[g].diseases:
            patient_diseases[d] += 1
    
    for p in pheno_tests:
        for d in phenotype_lookup[p].diseases:
            if d.relation == 'disease_phenotype_positive':
                patient_diseases[d.disease] += 1
            else:
                # It is a negaitve relationship
                patient_diseases[d.disease] -= 1
    
    counts = dict(patient_diseases)

    # Normalize by number of 
    norm_counts = {}
    for dn, c in counts.items():
        d = disease_lookup[dn.name]
        num_assoc = (len(d.genes) + len(d.phenotypes))
        norm_counts[dn.name] = c / num_assoc

    return norm_counts

def forward(gene_tests: List[str], pheno_tests: List[str]) -> List[Gene | Phenotype]:

    disease_probs: Dict = get_disease_probs(gene_tests, pheno_tests)

    # Get pheno and gene probs
    test_probs = defaultdict(lambda: 0)
    total_prob: float = 0.0

    for d, prob in disease_probs.items():
        for g in disease_lookup[d].genes:
            test_probs[g.name] += prob
        for p in disease_lookup[d].phenotypes:
            test_probs[p.phenotype.name + p.relation] += prob
        
        total_prob += prob

    print(test_probs, total_prob)

if __name__ == '__main__':

    # Test 1
    gt = ['A1BG']
    pt = ['Abnormal B cell count'] 

    forward(gt, pt)