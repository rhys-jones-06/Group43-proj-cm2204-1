# This is just a simple function to execute the code in general
# This file does not need to be submitted. Fiddle with it any way you see fit.
from functools import partial

from sklearn.metrics import accuracy_score, precision_score, recall_score

from c1_cleveland_data_preprocessor import *
from c3_naive_bayes import NaiveBayes
from c4_ethical_evaluation import Evaluator

# We load the dataset. Change the path as you see fit.
dataset_path = "../processed_cleveland_dataset.csv"
dataset = read_data(dataset_path)

# That's the name of the class variable in this dataset
class_name = 'target'

# We preprocess the dataset. This includes splitting it into training and testing data, cleaning, discretising, etc.
# The process used here is ugly. Don't worry about it. You will be improving on it in a different assessment :)
training_data, testing_data = preprocess(dataset, class_name)

# We extract some data for booting up our classifier
full_data = pd.concat([training_data, testing_data])
feature_info = {col: sorted(full_data[col].unique().tolist()) for col in full_data.columns}
class_values = feature_info.pop(class_name, None)

evaluator = Evaluator(class_values)
flipped_data = evaluator.flip_dataset(testing_data)

# Classifier is created, trained, and predictions are made
nb_classifier = NaiveBayes((class_name, class_values), feature_info)
nb_classifier.fit(training_data)
classified_data = nb_classifier.predict(testing_data)
# To check counterfactual fairness
flipped_results = nb_classifier.predict(flipped_data)

# Evaluations are made based on the predictions

true_classes = classified_data[class_name]
predicted_classes = classified_data["PredictedClass"]
flipped_classes = (flipped_results["PredictedClass"])
print(evaluator.evaluate_classification(true_classes, predicted_classes))

# Ethical evaluation is made based on the 'sex' column
print(evaluator.compute_group_fairness_ethical_evaluation(true_classes, predicted_classes, testing_data['sex']))

flip_rate = evaluator.compute_counterfactual_fairness(predicted_classes, flipped_classes)
print(f"flip rate: {round(flip_rate, 3)}")

transition_matrix = pd.crosstab(predicted_classes, flipped_classes, rownames=['Original'], colnames=['Flipped'])
print(transition_matrix)

#  Split the testing data by gender to observe directional bias
males_test = testing_data[testing_data['sex'] == 'male']
females_test = testing_data[testing_data['sex'] == 'female']

# 
def generate_sankey_data(original_df, title):
    # Flip the specific group
    flipped_df = evaluator.flip_dataset(original_df)
    
    # Get predictions for both
    orig_results = nb_classifier.predict(original_df)
    flip_results = nb_classifier.predict(flipped_df)
    
    # Extract predicted columns
    orig_preds = orig_results["PredictedClass"]
    flip_preds = flip_results["PredictedClass"]
    
    matrix = pd.crosstab(orig_preds, flip_preds)
    
    print(f"\n--- {title} ---")
    print(f"Group Flip Rate: {round(evaluator.compute_counterfactual_fairness(orig_preds, flip_preds), 3)}")
    for start_node in matrix.index:
        for end_node in matrix.columns:
            count = matrix.loc[start_node, end_node]
            if count > 0:
                print(f"Original: {start_node} [{count}] Flipped: {end_node}")


generate_sankey_data(males_test, "Male to Female Sankey Data")
generate_sankey_data(females_test, "Female to Male Sankey Data")