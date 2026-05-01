import copy

import pandas as pd
from feature_engine.discretisation import DecisionTreeDiscretiser
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from c3_naive_bayes import NaiveBayes


# You are allowed to change anything you see fit here for the purpose of Portfolio 2!

# Simple file reading function. Reads the file at the offered path, and returns it as a DataFrame.
def read_data(path: str) -> pd.DataFrame:
    dataset = None
    try:
        dataset = pd.read_csv(path, na_values='?')
    except FileNotFoundError as error:
        print(error)
        print("The data you wanted to read was not at the location passed to the function. Please make sure to "
              "provide a correct path to file.")

    except TypeError as terror:
        print(terror)
        print("Please provide a proper path to file, the input is missing.")
    return dataset


# Discretises the chosen columns using the DecisionTreeDiscretiser. Yes, the resulting columns still show
# as numerical, but interpretation changes and there is a limited set of them.
# At input it takes:
# - vars_to_discretize - list of column names to be discretized
# - training_data - data to be discretised, but from which the discretiser is allowed to learn (not all discretisers
#                   require learning - the one used here does).
# - testing_data - data to be discretised, but from which the discretiser is not allowed to learn
# - class_name - name of the class column
#
# As output, it produces a tuple of discretised training and testing data, held as DataFrames.
def discretize(vars_to_discretize: list[str], training_data: pd.DataFrame, testing_data: pd.DataFrame,
               class_name: str) -> \
        tuple[pd.DataFrame, pd.DataFrame]:
    features_train = training_data.drop(class_name, axis=1, inplace=False)
    class_train = training_data[class_name]
    features_test = testing_data.drop(class_name, axis=1, inplace=False)
    class_test = testing_data[class_name]

    dt_discretiser = DecisionTreeDiscretiser(
        cv=3,
        scoring='accuracy',
        variables=vars_to_discretize,
        regression=False,
        param_grid={'max_depth': [2, 3]}
    )
    dt_discretiser.fit(features_train, class_train)

    discretised_features_train = dt_discretiser.transform(features_train)
    discretised_features_test = dt_discretiser.transform(features_test)

    discretised_training_data = copy.deepcopy(discretised_features_train)
    discretised_training_data[class_name] = class_train
    discretised_test_data = copy.deepcopy(discretised_features_test)
    discretised_test_data[class_name] = class_test

    return discretised_training_data, discretised_test_data


# Handles missing data in the cleveland dataset. In this case, it simply drops rows containing missing entries. Takes
# a DataFrame at input and returns one at output.
def handle_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = data.dropna()
    return df_cleaned


# Preprocesses the cleveland dataset. Function modifies column types, handles missing data, and discretises
# numerical variables. It also splits the dataset into training and testing parts.
def preprocess(data: pd.DataFrame, class_name: str, n_splits: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = copy.deepcopy(data)

    # training_dataset, testing_dataset = train_test_split(
    #     dataset,
    #     test_size=0.3,
    #     stratify=dataset[['target', 'sex']],
    #     random_state=42
    # )

    skf = StratifiedKFold(n_splits=10)
    folds = []

    #vars_to_discretize = [col for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'] if col in dataset.columns]

    # We discretize the data. Don't be alarmed just because the returned values are numbers.
    for train_index, test_index in skf.split(dataset, dataset['target']):
        training_dataset = dataset.iloc[train_index]
        testing_dataset = dataset.iloc[test_index]

        #training_dataset, testing_dataset = discretize( train_data, test_data, class_name)

        # We change feature types
        training_dataset = training_dataset.astype(object)
        testing_dataset = testing_dataset.astype(object)

        folds.append((training_dataset, testing_dataset))

    return folds

def analyse_feature_importance(nb_classifier: NaiveBayes, features_values, target_values):

    perm_importance = permutation_importance(
        nb_classifier.inner_nb,
        features_values,
        target_values,
        n_repeats=10,
    )

    feature_names = nb_classifier.encoder.get_feature_names_out()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std,
    }).sort_values(by='importance', ascending=False)

    return importance_df
