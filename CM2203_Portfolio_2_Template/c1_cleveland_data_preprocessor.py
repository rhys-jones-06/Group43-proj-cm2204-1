import copy
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from feature_engine.discretisation import DecisionTreeDiscretiser



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

    # We handle missing entries
    dataset = handle_missing_data(dataset)

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
    ssd = sss.split(dataset, dataset[class_name])
    train_index, test_index = next(ssd)
    training_dataset = dataset.iloc[train_index]
    testing_dataset = dataset.iloc[test_index]

    vars_to_discretize = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    # We discretize the data. Don't be alarmed just because the returned values are numbers.
    training_dataset, testing_dataset = discretize(vars_to_discretize, training_dataset, testing_dataset, class_name)

    # We change feature types
    training_dataset = training_dataset.astype(object)
    testing_dataset = testing_dataset.astype(object)
    return training_dataset, testing_dataset
