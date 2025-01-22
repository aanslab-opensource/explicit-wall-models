import os
import pickle
import numpy as np
import pandas as pd

pd.set_option('display.float_format', '{:.4f}'.format)

laws = ["spalding", "reichardt-fixedB1B2", "eqode"]
groups = ["Parameter", "Spalding", "Reichardt", "EqODE"]

table_dict = {}

for idx_group, group  in enumerate(groups):
    
    if group=="Parameter":
        table_dict.update({(group,''):["mu", "sigma", "xi", "p", "s"]})
    else:
        with open(f'./results/global_{laws[idx_group-1]}.pkl', 'rb') as file:
            data = pickle.load(file)
        
        for idx_col, col in enumerate(["A[:,0]", "A[:,1]", "b"]):

            ll = []

            for param in ["mu1", "sigma1", "scale1", "p", "s"]:
                if col=="b":
                    ll.append(data['fields_regs'][param].intercept_)
                else:
                    ll.append(data['fields_regs'][param].coef_[idx_col])

            table_dict.update({(group, col) : ll})

df = pd.DataFrame(table_dict)

os.makedirs("./tables", exist_ok=True)

df.to_csv(f'./tables/global_regression_coefficients.csv', index=False)

print(df)
