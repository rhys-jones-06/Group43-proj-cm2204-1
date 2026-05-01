import copy

import numpy
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class NaiveBayes:

    def __init__(self, class_info: tuple[str, list[str]], feature_info: dict[str, list[str]], var_smoothing: float = 1e-9):
        self.class_info = class_info
        self.feature_info = feature_info
        self.inner_nb = GaussianNB(var_smoothing=var_smoothing)
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.scaler = StandardScaler()

    def fit(self, training_data: pd.DataFrame):
        class_name = self.class_info[0]
        features = self.encoder.fit_transform(training_data.drop(class_name, axis=1)).astype(float)
        features = self.scaler.fit_transform(features)
        self.inner_nb.fit(features, training_data[class_name])

    def predict(self, testing_data: pd.DataFrame) -> pd.DataFrame:
        class_name = self.class_info[0]
        features = self.encoder.transform(testing_data.drop(class_name, axis=1, errors='ignore')).astype(float)
        features = self.scaler.transform(features)
        predicted_class = self.inner_nb.predict(features)
        classified_data = copy.deepcopy(testing_data)
        classified_data['PredictedClass'] = predicted_class
        return classified_data

    def retrieve_class_probability(self, class_value: str) -> float:
        class_index = numpy.where(self.inner_nb.classes_ == class_value)[0][0]
        return numpy.exp(self.inner_nb.class_log_prior_[class_index])

    def retrieve_conditional_probability(self, class_value: str, feature_name: str, feature_value: str) -> float:
        class_index = numpy.where(self.inner_nb.classes_ == class_value)[0][0]
        feature_index = list(self.encoder.feature_names_in_).index(feature_name)
        # Scale the value the same way the scaler was fitted, then compute Gaussian PDF
        x_scaled = (float(feature_value) - self.scaler.mean_[feature_index]) / self.scaler.scale_[feature_index]
        mean = self.inner_nb.theta_[class_index, feature_index]
        var = self.inner_nb.var_[class_index, feature_index]
        return float((1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x_scaled - mean) ** 2) / var))