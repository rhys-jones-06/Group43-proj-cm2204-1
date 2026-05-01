import pandas as pd


# Simple function stub for balancing the dataset according to a particular feature.
# Modify as you see fit, including the function name/template, etc!

def balance_dataset(dataset: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    print(f"No of rows: {len((dataset.get('sex')))}")
    print(f"No of males: {len([row for row in dataset.itertuples(index=True, name='Pandas') if row.sex == "male"])}")
    print(
        f"No of females: {len([row for row in dataset.itertuples(index=True, name='Pandas') if row.sex == "female"])}")

    if feature_name != "" and feature_name in dataset.columns:
        dataset.drop(columns=[feature_name], inplace=True)

    return dataset