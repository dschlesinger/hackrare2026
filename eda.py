import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rich import print

df = pd.read_csv('gene_pheno_disease.csv', low_memory=False, index_col=0)

print(
    'Columns =', df.columns.to_list(), '\n',
    'n =', df.__len__(), '\n',
    '[bold red]Types of types[/bold red]', '\n,',
    node_types := np.unique(df[['x_type', 'y_type']].to_numpy()), '\n',
    '[bold red]Types of relations[/bold red]', '\n',
    *[f"{nt} = {[f'{name}: {count}' for name, count in zip((df[(df['x_type'] == nt) | (df['y_type'] == nt)]['relation']).unique().tolist(), df[(df['x_type'] == nt) | (df['y_type'] == nt)]['relation'].value_counts())]}\n" for nt in node_types],
    '[bold red]Sources[/bold red]', '\n',
    np.unique(df[['x_source', 'y_source']].to_numpy()), '\n',
    sep = ' ',
)