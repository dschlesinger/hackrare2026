import pandas as pd
import os

primekg = pd.read_csv('prime_kg.csv', low_memory=False)

# Filter for
# Gene -> Disease
# Disease -> Phenotype
filter_db = primekg[
    (((primekg['x_type'] == 'disease')
    |
    (primekg['y_type'] == 'disease'))
    &
    ((primekg['x_type'] == 'effect/phenotype')
    |
    (primekg['y_type'] == 'effect/phenotype')))
    |
    (((primekg['x_type'] == 'disease')
    |
    (primekg['y_type'] == 'disease'))
    &
    ((primekg['x_type'] == 'gene/protein')
    |
    (primekg['y_type'] == 'gene/protein')))
]

# Save new db
filter_db.to_csv('gene_pheno_disease.csv')

# Delete old db
os.remove('prime_kg.csv')