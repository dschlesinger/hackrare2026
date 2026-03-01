import pandas as pd
import numpy as np
from dianfo.schema import Gene, Phenotype, Disease, DiseasePhenotypeRelation
from typing import Dict, Tuple, List
from collections import Counter, defaultdict
import sys
import math

import pydantic

from rich import print

from fastapi import FastAPI

app = FastAPI()

# flag = sys.argv[-1]

# if flag == '--test':
#     print('[bold cyan]Testing with mini.csv[/bold cyan]')
#     df = pd.read_csv('mini.csv', low_memory=False, index_col=0)
# else:
#     print('[bold cyan]Running with full dataset[/bold cyan]')
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

def get_disease_probs(gene_tests, pheno_tests, ruled_out_genes, ruled_out_phenotypes) -> Dict:

    gene_to_disease_counter = Counter(sum([gene_lookup[g].diseases for g in gene_tests], []))

    norm_gene_disease_probs = {d: c / (len(gene_tests) + len(disease_lookup[d.name].genes)) for d, c in gene_to_disease_counter.items()}
    
    pos_pheno_to_disease_counter = Counter(sum([[p.disease for p in phenotype_lookup[g].diseases if p.relation == 'disease_phenotype_positive'] for g in pheno_tests], []))
    neg_pheno_to_disease_counter = Counter(sum([[p.disease for p in phenotype_lookup[g].diseases if p.relation == 'disease_phenotype_negative'] for g in pheno_tests], []))

    norm_pos_pheno_disease_probs = {d: c / (len(pheno_tests) + len(disease_lookup[d.name].phenotypes)) for d, c in pos_pheno_to_disease_counter.items()}
    norm_neg_pheno_disease_probs = {d: c / (len(pheno_tests) + len(disease_lookup[d.name].phenotypes)) for d, c in neg_pheno_to_disease_counter.items()}

    # If the diseases have a gene or phenotype we have ruled out
    not_have_gene_coutner = Counter(sum([gene_lookup[g].diseases for g in ruled_out_genes], []))
    not_have_pos_pheno_coutner = Counter(sum([[pd.disease for pd in phenotype_lookup[p].diseases if pd.relation == 'disease_phenotype_positive'] for p in ruled_out_phenotypes], []))
    not_have_neg_pheno_coutner = Counter(sum([[pd.disease for pd in phenotype_lookup[p].diseases if pd.relation == 'disease_phenotype_negative'] for p in ruled_out_phenotypes], []))

    ruled_out_gene_disease_probs = {d: c / (len(gene_tests) + len(disease_lookup[d.name].genes)) for d, c in not_have_gene_coutner.items()}
    ruled_out_pos_pheno_disease_probs = {d: c / (len(pheno_tests) + len(disease_lookup[d.name].phenotypes)) for d, c in not_have_pos_pheno_coutner.items()}
    ruled_out_neg_pheno_disease_probs = {d: c / (len(pheno_tests) + len(disease_lookup[d.name].phenotypes)) for d, c in not_have_neg_pheno_coutner.items()}

    all_diseases = set(sum(map(lambda l: list(l.keys()), [norm_gene_disease_probs, norm_pos_pheno_disease_probs, norm_neg_pheno_disease_probs]), []))

    probs = {}

    for d in all_diseases:
        g = norm_gene_disease_probs.get(d, 0)
        pp = norm_pos_pheno_disease_probs.get(d, 0)
        pn = norm_neg_pheno_disease_probs.get(d, 0)
        rog = ruled_out_gene_disease_probs.get(d, 0)
        ropp = ruled_out_pos_pheno_disease_probs.get(d, 0)
        ropn = ruled_out_neg_pheno_disease_probs.get(d, 0)

        # Must have something going for it
        if g or pp:
            probs[d] = ( g + pp - pn - rog - ropp + ropn ) / 2

    # Shift so min is 1
    min_prob = min(probs.values())
    probs = {d: p - min_prob + 10e-9 for d, p in probs.items()}

    # Normalize by total and filter out zeros
    total = sum([p for p in probs.values()])
    probs = {d: p / total for d, p in probs.items()}

    return probs

def get_possible_tests(gene_tests, pheno_tests) -> List[Gene | Phenotype]:

    tests = set()

    gt = [gene_lookup[g] for g in gene_tests]
    pt = [phenotype_lookup[p] for p in pheno_tests]

    for t in gt:
        for d in t.diseases:
            for dg in d.genes:
                tests.add(dg)
            for dp in d.phenotypes:
                tests.add(dp.phenotype)
    
    for t in pt:
        for d in t.diseases:
            for dg in d.disease.genes:
                tests.add(dg)
            for dp in d.disease.phenotypes:
                tests.add(dp.phenotype)
    
    # Remove ones from patient
    tests -= set(gt) | set(pt)

    return list(tests)

def entropy(probs: List[float]) -> float:

    return -sum([p * math.log2(p) for p in probs])

def score_test(gene_tests, pheno_tests, not_present_genes, not_present_pheno, base_info: float, test: Gene | Phenotype) -> float:

    if isinstance(test, Gene):
        # Entropy if True
        disease_probs_t: Dict = get_disease_probs([*gene_tests, test.name], pheno_tests, not_present_genes, not_present_pheno)
        net = entropy(disease_probs_t.values())

        # Entorpy if False
        disease_probs_f: Dict = get_disease_probs(gene_tests, pheno_tests, [*not_present_genes, test.name], not_present_pheno)
    else:
        # Entropy if True
        disease_probs_t: Dict = get_disease_probs(gene_tests, [*pheno_tests, test.name], not_present_genes, not_present_pheno)
        net = entropy(disease_probs_t.values())

        # Entorpy if False
        disease_probs_f: Dict = get_disease_probs(gene_tests, pheno_tests, not_present_genes, [*not_present_pheno, test.name])

    nef = entropy(disease_probs_f.values())

    return base_info - ((net + nef) / 2)

def forward(gene_tests: List[str], pheno_tests: List[str], not_present_genes: List[str], not_present_pheno: List[str]) -> Tuple[Dict, Dict]:

    disease_probs: Dict = get_disease_probs(gene_tests, pheno_tests, not_present_genes, not_present_pheno)

    ptests = get_possible_tests(gene_tests, pheno_tests)

    print('[bold purple]Entropy: [/bold purple]', base_info := entropy(disease_probs.values()))

    test_info = {pt: score_test(gene_tests, pheno_tests, not_present_genes, not_present_pheno, base_info, pt) for pt in ptests}

    print(*[f'{d}: {test_info[d]}' for d in sorted(list(test_info.keys()), key=lambda d: test_info[d])], sep='\n')

    return {pt.name: s for pt, s in test_info.items()}, {d.name: s for d, s in disease_probs.items()}

class InferenceRequest(pydantic.BaseModel):
    gene_tests: List[str]
    pheno_tests: List[str]
    not_present_pheno: List[str]
    not_present_genes: List[str]

class InferenceReturn(pydantic.BaseModel):
    tests: Dict
    diseases: Dict

@app.post('/inference')
def inference(r: InferenceRequest) -> InferenceReturn:

    tests, disease = forward(r.gene_tests, r.pheno_tests, r.not_present_genes, r.not_present_pheno)

    return InferenceReturn(
        tests=tests,
        diseases=disease
    )

# if __name__ == '__main__':

#     # Test 1
#     gt = ['A1BG', 'CNR2']
#     pt = ['Abnormal B cell count'] 
#     ngt = []
#     npt = []

#     forward(gt, pt, ngt, npt)