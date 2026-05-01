# This is just a simple function to execute the code in general
# This file does not need to be submitted. Fiddle with it any way you see fit.
from functools import partial

from sklearn.metrics import accuracy_score, precision_score, recall_score

from c1_cleveland_data_preprocessor import *
from c2_data_balancing import balance_dataset
from c3_naive_bayes import NaiveBayes
from c4_ethical_evaluation import Evaluator

from hyperparameter_tuner import tune_var_smoothing
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# We load the dataset. Change the path as you see fit.
dataset_path = "../processed_cleveland_dataset.csv"
dataset = read_data(dataset_path)

# That's the name of the class variable in this dataset
class_name = 'target'

# We handle missing entries
dataset = handle_missing_data(dataset)

balanced_data = balance_dataset(dataset, "chol")

# We preprocess the dataset. This includes splitting it into training and testing data, cleaning, discretising, etc.
# The process used here is ugly. Don't worry about it. You will be improving on it in a different assessment :)
folds = preprocess(balanced_data, class_name)

# make_graphs(training_data, testing_data)

stats_avg = []
bias_avg = []
importance_avg = []

for (training_data, testing_data) in folds:

    # We extract some data for booting up our classifier
    full_data = pd.concat([training_data, testing_data])

    feature_info = {col: sorted(full_data[col].unique().tolist()) for col in full_data.columns}
    class_values = feature_info.pop(class_name, None)

    # Tune var_smoothing on training data only, then fit
    best_var_smoothing = tune_var_smoothing(training_data, class_name)
    nb_classifier = NaiveBayes((class_name, class_values), feature_info, var_smoothing=best_var_smoothing)
    nb_classifier.fit(training_data)
    classified_data = nb_classifier.predict(testing_data)

    # Analyze feature importance
    features_test = nb_classifier.encoder.transform(testing_data.drop(class_name, axis=1, errors='ignore'))
    importance_df = analyse_feature_importance(nb_classifier, features_test, classified_data[class_name])
    importance_avg.append(importance_df)

    # Evaluations are made based on the predictions

    true_classes = classified_data[class_name]
    predicted_classes = classified_data["PredictedClass"]
    evaluator = Evaluator(class_values)
    stats = evaluator.evaluate_classification(true_classes, predicted_classes)
    stats_avg.append(stats)
    # print(stats)

    # Ethical evaluation is made based on the 'sex' column
    bias = evaluator.compute_group_fairness_ethical_evaluation(true_classes, predicted_classes, testing_data['sex'])
    bias_avg.append(bias)
    # print(bias)

average_stats = pd.DataFrame(stats_avg).mean()
print("\naverage:")
print(average_stats)


average_bias = pd.concat(bias_avg).groupby(level=0).mean()
print("\nbias:")
print(average_bias)

# Average feature importance across all folds
print("\n=== AVERAGE Feature Importance Across All Folds ===")
combined_importance = pd.concat(importance_avg)
avg_importance = combined_importance.groupby('feature')[['importance', 'std']].mean().sort_values('importance', ascending=False)

# Calculate z-scores
mean_importance = avg_importance['importance'].mean()
std_importance = avg_importance['importance'].std()
avg_importance['z_score'] = (avg_importance['importance'] - mean_importance) / std_importance

for feature in avg_importance.index:
    importance = avg_importance.loc[feature, 'importance']
    std = avg_importance.loc[feature, 'std']
    z_score = avg_importance.loc[feature, 'z_score']
    print(f"{feature} | Importance: {importance:.6f} | Std: {std:.6f} | Z-Score: {z_score:.3f}")