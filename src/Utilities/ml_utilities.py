import math
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

def train_test_split(df):
    """
    :param df: panda dataFrame which has means of ROIs
    :return: Training data set and testing dataset
    """

    df = shuffle(df)

    split_index = math.floor(0.8 * len(df))
    train = df[:split_index]
    test = df[split_index:]

    return train, test


def balanced_accuracy(predictions, true_values):
    label1 = list(
        map(lambda predictions, true_values: predictions == true_values & true_values == 1, predictions, true_values))
    label2 = list(
        map(lambda predictions, true_values: predictions == true_values & true_values == 2, predictions, true_values))
    label3 = list(
        map(lambda predictions, true_values: predictions == true_values & true_values == 3, predictions, true_values))
    true_label1 = float(sum(true_values == 1))
    true_label2 = float(sum(true_values == 2))
    true_label3 = float(sum(true_values == 3))
    denominator = 3
    if not true_label1 :
        true_label1 = 1.0
        denominator = 2
    if not true_label2:
        true_label2 = 1.0
        denominator = 2
    if not true_label3:
        true_label3 = 1.0
        denominator = 2


    baccuracy = ((sum(label1) / true_label1) +
                 (sum(label2) / true_label2) +
                 (sum(label3) / true_label3)) / denominator

    return baccuracy


def model_fitting(model_name, X, y, kFold=10, normalize=False):
    classification = True
    min_max_scaler = preprocessing.MinMaxScaler()
    if normalize:
        X_minmax = min_max_scaler.fit_transform(X,y)
        X = X_minmax
    if model_name == "svm_kernel_default":
        model = svm.SVC(kernel='rbf', C=4, gamma=2 ** -5)
    elif model_name == "svm_kernel_tuned":
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                        'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 2 ** -5, 2 ** -10, 2 ** 5], 'kernel': ['rbf']}
        grid = GridSearchCV(svm.SVC(), param_grid, refit=True, cv=kFold, iid=False)
        grid.fit(X, y)
        best_param = grid.best_params_
        model = svm.SVC(kernel=best_param['kernel'], C=best_param['C'], gamma=best_param['gamma'])
    elif model_name == "naive_bayes":
        model = GaussianNB()
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier()
    elif model_name =="svm_linear":
        model = svm.SVC(kernel="linear")
    elif model_name == "rfc":
        model = RandomForestClassifier(n_estimators=200)
    elif model_name == "logistic_regression":
        model = LogisticRegression(solver="liblinear", multi_class='auto')
    elif model_name == "linear_reg":
        model = LinearRegression()
    elif model_name == "polynomial_reg":
        model = Pipeline([('poly', PolynomialFeatures(degree=4)),
                          ('linear', LinearRegression())])
    elif model_name == 'lasso':
        model = Lasso(alpha=0.1)
    elif model_name == 'svr_kernel_default':
        model = svm.SVR(kernel='linear',  C=4, gamma=2 ** -5)
        model.fit(X, y)
        scores = model.score(X, y)
        pred = model.predict(X)
        pred = np.round(pred)
        balanced_accuracy =  mean_squared_error(y, pred, multioutput='raw_values')
        classification = False
    elif model_name == 'svr_kernel_tuned':
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 2 ** -5, 2 ** -10, 2 ** 5], 'kernel': ['linear']}
        grid = GridSearchCV(svm.SVR(), param_grid, refit=True, cv=kFold)
        grid.fit(X, y)
        best_param = grid.best_params_
        model = svm.SVR(kernel=best_param['kernel'], C=best_param['C'], gamma=best_param['gamma'])
        model.fit(X, y)
        scores = model.score(X, y)
        pred = model.predict(X)
        pred = np.round(pred)
        balanced_accuracy =  mean_squared_error(y, pred, multioutput='raw_values')
        classification = False
    elif model_name == 'gpr_default':
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        model = GaussianProcessRegressor( n_restarts_optimizer=9)
        model.fit(X, y)
        scores = model.score(X, y)
        pred, sigma = model.predict(X, return_std=True)
        pred = np.round(pred)
        balanced_accuracy = mean_squared_error(y, pred, multioutput='raw_values')
        classification = False
    else:
        model = svm.SVC(kernel='rbf', C=4, gamma=2 ** -5)
        model.fit(X, y)
        scores = model.score(X, y)
        pred = model.predict(X)
        #balanced_accuracy = balanced_accuracy(pred, y)
        balanced_accuracy = balanced_accuracy_score(y,pred)
    if classification:
        model.fit(X, y)
        scores = model.score(X, y)
        pred = model.predict(X)
        #balanced_accuracy = balanced_accuracy(pred,y)
        balanced_accuracy = balanced_accuracy_score(y, pred)

    #print("Predicted:%s, Actual:%s"%(pred[0], y[0]))


    return scores, balanced_accuracy, model, min_max_scaler


def model_test(test, model):
    # Testing
    test_data = test.loc[:, test.columns != "label"].values
    test_actual_results = np.asarray(test.label).astype(float)
    test_prediction = model.predict(test_data)
    total_test_samples = test_data.shape[0]
    total_correct_predictions = np.count_nonzero(test_actual_results == test_prediction)
    test_accuracy = np.asarray(total_correct_predictions / total_test_samples)
    #print("Test Accuracy is {}.".format(test_accuracy))
    #print(test_accuracy)
    return test_accuracy


def get_features_labels(data):
    """
    :param data: pandas dataframe which has label, subject_cont and ROI mean values
    :return: X: Features and y: corresponding labels

    """
    p = data.drop('subject_cont', axis = 1)
    X = p.loc[:, p.columns != "label"].values
    y = np.asarray(data.label).astype(int)

    return X,y


def missing_values(df, method=1):
    """
    This function replaces NaN in the df with 0. This is temporary fix.
    @TODO: Should do research for better method to handle missing solutions
    :param df: panda dataFrame which has means of ROIs
    :param method: 0 for filling it with 0s, 1 for mean replacement
    :return: panda dataFrame which has means of ROIs with no missing values
    """
    # TODO: Should do research for better method to handle missing solutions

    # np.all(np.isfinite(df))  # to check if there is any infinite number
    # np.any(np.isnan(df))  # to check if there is any nan
    # np.where(np.asanyarray(np.isnan(df)))  # to find the index of nan

    if method == 1:
        # Mean replacement for nan numbers in particular label with the mean of non missing values in the same label.
        df.fillna(df.mean(), inplace=True)

    elif method == 2:
        # Median replacement for nan numbers in particular label with the Median of non missing values in the same label
        df.fillna(df.median(), inplace=True)

    elif method == 3:
        # Remove the columns even if one value is missing. This is in attempt to handle Perfect Separation of GLMfit

        #try with last value seen for a column
        #df.fillna(method='ffill', inplace=True)

        #try with mean of other non-null column volumes of the same subject
        #df.fillna(df.mean(), axis=0, inplace=True)

        df.dropna(axis=0, how="any", inplace=True)
    else:

        df.fillna(0, inplace=True)
    return df


def permutation_test(X, y, estimator, n_permutations, kFold):

    score, permutation_scores, p_value = permutation_test_score(estimator=estimator, X=X, y=y,
                                                                scoring='balanced_accuracy', cv=StratifiedKFold(kFold),
                                                                n_permutations=n_permutations, n_jobs=1)
    return score, permutation_scores, p_value


def preprocess_remove_gender(df):
    """
    :param df: data frame which contains mean activation values in 116 brainn areas
    :return: male and female combined training and testing data where testing data is standardized using mean and variance of
             the respective gender training data.
    """

    # Split the data into 80% training and 20% testing.
    train, test = train_test_split(df)

    # Obtaining male and female dataframes
    train_male = train.loc[train["gender"] == 1]
    train_female = train.loc[train["gender"] == 2]
    test_male = test.loc[test["gender"] == 1]
    test_female = test.loc[test["gender"] == 2]

    # Removing age and gender info from dataframe, so that only mean activation values in 116 brain regions are considered.
    train_male = train_male.drop(['gender', 'age'], axis=1, errors='ignore')
    train_female = train_female.drop(['gender', 'age'], axis=1, errors='ignore')
    test_male = test_male.drop(['gender', 'age'], axis=1, errors='ignore')
    test_female = test_female.drop(['gender', 'age'], axis=1, errors='ignore')

    # Converting dataframes into X and Y arrays wrt male and female
    x_train_male, y_train_male = get_features_labels(train_male)
    x_train_female, y_train_female = get_features_labels(train_female)

    x_test_male, y_test_male = get_features_labels(test_male)
    x_test_female, y_test_female = get_features_labels(test_female)

    # Standardisation of male training data and female training data
    scaler_male = StandardScaler()
    scaler_female = StandardScaler()
    x_train_male = scaler_male.fit_transform(x_train_male, y_train_male)
    x_train_female = scaler_female.fit_transform(x_train_female, y_train_female)

    # Standardisation of male testing data using mean and variance from male training data scale.
    x_test_male = scaler_male.transform(x_test_male)

    # Standardisation of female testing data using mean and variance from female training data scale.
    x_test_female = scaler_female.transform(x_test_female)

    # Combining male training data and female training data.
    x_train = np.concatenate((x_train_male, x_train_female))
    y_train = np.concatenate((y_train_male, y_train_female))

    # Combining male testing data and female testing data.
    x_test = np.concatenate((x_test_male, x_test_female))
    y_test = np.concatenate((y_test_male, y_test_female))

    return x_train, y_train, x_test, y_test


def merge_additional_data(df, additional_data_folder="../data_info/"):
    """
    :param df: dataFrame which consists of 116 brain regions activations, label, and subject_cont (id of subject)
    :param additional_data_folder: Folder which consists additional information of the subjects like age, gender.
            The folder should contain subject_name.txt which has list of subject id and n300.csv which contains
            additional info like age, gender etc of these subjects in the same order as of subject_name.txt
    :return: input df along with age and gender information
    """

    # Age and gender information along with subject id is extracted
    file = open(additional_data_folder + "/subject_name.txt", "r")
    ids = file.read().split()
    ids = [int(float(id)) for id in ids]
    edf = pd.read_csv(additional_data_folder + '/n300.csv')
    edf['subject_cont'] = ids
    edf = edf[['KJØNN', 'subject_cont', 'ALDER']]
    edf = edf.rename(columns={'KJØNN': 'gender', 'ALDER': 'age'})
    df = pd.merge(df, edf, on=['subject_cont'], how='inner')

    return df


def prepare_y_for_glmfit(y, user_group):
    y_final = np.array([], dtype=np.int64)

    if user_group == 12:
        first_label = 1
    elif user_group == 23:
        first_label = 2
    elif user_group == 31:
        first_label = 3

    for x in y:
        if x == first_label:
            y_final = np.append(y_final, np.array([[1, 0]], dtype=np.int64))
        else:
            y_final = np.append(y_final, np.array([[0, 1]], dtype=np.int64))

    y_final = y_final.reshape(int(len(y_final)/2), 2)

    return y_final