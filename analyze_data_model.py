#!/usr/bin/env python3
"""Analyze data model connectivity and gaps."""
import pandas as pd
import json

silver_dir = 'diageno/data/silver'
artifacts_dir = 'diageno/data/model_artifacts'

# Load all data
diseases = pd.read_parquet(f'{silver_dir}/diseases.parquet')
disease_hpo = pd.read_parquet(f'{silver_dir}/disease_hpo.parquet')
disease_gene = pd.read_parquet(f'{silver_dir}/disease_gene.parquet')
id_mapping = pd.read_parquet(f'{silver_dir}/id_mapping.parquet')
cases = pd.read_parquet(f'{silver_dir}/cases.parquet')
pheno_events = pd.read_parquet(f'{silver_dir}/phenotype_events.parquet')
hpo_terms = pd.read_parquet(f'{silver_dir}/hpo_terms.parquet')
matrix = pd.read_parquet(f'{artifacts_dir}/disease_hpo_matrix.parquet')

print('=' * 60)
print('DATASET CONNECTIVITY ANALYSIS')
print('=' * 60)

print('\n## 1. Matrix vs Source Data ##')
print(f'Matrix shape: {matrix.shape}')
print(f'  Matrix diseases: {len(matrix.index)}')
print(f'  Matrix HPOs: {len(matrix.columns)}')
print(f'  diseases.parquet: {len(diseases)} diseases')
print(f'  disease_hpo.parquet: {len(disease_hpo)} associations')
print(f'  disease_gene.parquet: {len(disease_gene)} gene links')
print(f'  hpo_terms.parquet: {len(hpo_terms)} HPO terms')

print('\n## 2. HPO Coverage (Phenopackets vs Matrix) ##')
pheno_hpos = set(pheno_events['hpo_id'].unique())
matrix_hpos = set(matrix.columns)
overlap = pheno_hpos & matrix_hpos
missing = pheno_hpos - matrix_hpos
print(f'Phenopacket HPO terms: {len(pheno_hpos)}')
print(f'Matrix HPO terms: {len(matrix_hpos)}')
print(f'Overlap: {len(overlap)} ({100*len(overlap)/len(pheno_hpos):.1f}%)')
print(f'Missing from matrix: {len(missing)} phenopacket HPOs not in disease matrix')

print('\n## 3. ID Mapping (Cross-references) ##')
print(f'Total mappings: {len(id_mapping)}')
print(f'Unique ORPHA IDs with mappings: {id_mapping.orpha_id.nunique()}')
print(f'With OMIM: {id_mapping.omim_id.notna().sum()}')
print(f'With ICD-10: {id_mapping.icd10.notna().sum()}')

print('\n## 4. Ground Truth Disease Labels (from Phenopackets) ##')
# Extract disease IDs from phenopackets
gt_diseases = []
for raw in cases['raw_json']:
    try:
        data = json.loads(raw)
        if 'diseases' in data:
            for d in data['diseases']:
                gt_diseases.append(d.get('term', {}).get('id', ''))
        if 'interpretations' in data:
            for interp in data['interpretations']:
                gt_diseases.append(interp.get('diagnosis', {}).get('disease', {}).get('id', ''))
    except:
        pass

gt_diseases = [d for d in gt_diseases if d]
print(f'Phenopackets with disease labels: {len(gt_diseases)}')
gt_orpha = [d for d in gt_diseases if d.startswith('ORPHA:')]
gt_omim = [d for d in gt_diseases if d.startswith('OMIM:')]
gt_mondo = [d for d in gt_diseases if d.startswith('MONDO:')]
print(f'  ORPHA: {len(gt_orpha)}, OMIM: {len(gt_omim)}, MONDO: {len(gt_mondo)}')

# Check how many map to our matrix
orpha_in_matrix = set(matrix.index)
direct_match = len([d for d in gt_orpha if d in orpha_in_matrix])
print(f'\nDirect ORPHA matches (in matrix): {direct_match}/{len(gt_orpha)} ({100*direct_match/max(len(gt_orpha),1):.1f}%)')

# Check OMIM via id_mapping
omim_to_orpha = dict(zip(id_mapping['omim_id'].dropna(), id_mapping['orpha_id']))
omim_mapped = sum(1 for d in gt_omim if omim_to_orpha.get(d) in orpha_in_matrix)
print(f'OMIM→ORPHA via mapping: {omim_mapped}/{len(gt_omim)} ({100*omim_mapped/max(len(gt_omim),1):.1f}%)')

print('\n## 5. Gene Integration Status ##')
diseases_with_genes = disease_gene['disease_id'].nunique()
genes_total = disease_gene['gene_symbol'].nunique()
print(f'Diseases with gene associations: {diseases_with_genes}')
print(f'Unique genes: {genes_total}')
# Check if gene-associated diseases are in matrix
gene_diseases_in_matrix = len(set(disease_gene['disease_id']) & orpha_in_matrix)
print(f'Gene-linked diseases in matrix: {gene_diseases_in_matrix}/{diseases_with_genes}')

print('\n' + '=' * 60)
print('GAPS IDENTIFIED')
print('=' * 60)
print('''
1. HPO COVERAGE GAP: {pct:.1f}% of phenopacket HPOs not in disease matrix
   - Cause: Orphadata disease-HPO uses subset of HPO; phenopackets use full HPO
   - Impact: Some patient phenotypes contribute zero signal to scoring

2. ID NAMESPACE MISMATCH: Phenopackets use OMIM/MONDO, matrix uses ORPHA
   - Only {direct_pct:.1f}% of phenopacket diseases directly match matrix
   - id_mapping.parquet exists but NOT used during inference

3. GENE DATA UNUSED: {genes} genes linked to {gene_diseases} diseases
   - disease_gene.parquet exists but not incorporated into scoring
   - Gene findings in UI don't influence disease ranking

4. CALIBRATION USES SCORE DISTRIBUTION, NOT GROUND TRUTH
   - We have disease labels but can't use them due to namespace mismatch
   - Confidence is relative positioning, not true probability
'''.format(
    pct=100*len(missing)/len(pheno_hpos),
    direct_pct=100*direct_match/max(len(gt_orpha),1),
    genes=genes_total,
    gene_diseases=diseases_with_genes
))
