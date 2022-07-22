import pandas as pd

data = pd.read_pickle("FitSNAP.df")

print(data.columns)

print(data['Row_Type'])

print(data['Atom_Type'].values)
