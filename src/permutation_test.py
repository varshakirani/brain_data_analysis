'''
This script writes permutation test for all features
nBack : Contrast 2 and Contrast 3: SVM, Random Forest and Logistic Regression
Face : Contrast 5 and 4 : SVM, Random Forest
'''
import sys
import os
import re
sys.path.insert(0, '../src/Utilities')
import pandas as pd
import tools
import ml_utilities as mlu
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import time
from sklearn.model_selection import GridSearchCV
import numpy as np


def run_perm_test(df, contrast_name, classifier_no, out, n_iterations):
    models = ["svm_kernel_default", "svm_kernel_tuned", "rfc", 'logistic_regression']
    X, y = mlu.get_features_labels(df)

    for i in range(n_iterations):
        train, test = mlu.train_test_split(df)
        #x_train, y_train = mlu.get_features_labels(train)
        #x_test, y_test = mlu.get_features_labels(test)
        x_train, y_train, x_test, y_test = mlu.preprocess_remove_gender(df)
        for model_name in models:
            if model_name == "svm_kernel_default":
                model = svm.SVC(kernel='rbf', C=4, gamma=2 ** -5)
            elif model_name == "svm_kernel_tuned":
                param_grid = {'C': [0.1, 1, 10, 100, 1000],
                              'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 2 ** -5, 2 ** -10, 2 ** 5], 'kernel': ['rbf']}
                grid = GridSearchCV(svm.SVC(), param_grid, refit=True, cv=10, iid=False)
                grid.fit(X, y)
                best_param = grid.best_params_
                model = svm.SVC(kernel=best_param['kernel'], C=best_param['C'], gamma=best_param['gamma'])
            elif model_name == "rfc":
                model = RandomForestClassifier(n_estimators=200)
            elif model_name == "logistic_regression":
                model = LogisticRegression(solver="liblinear", multi_class='auto')

            trained_model = model.fit(x_train,y_train)
            scores = mlu.balanced_accuracy(trained_model.predict(x_test), y_test)
            if os.path.isfile(options.input):
                df_res = pd.read_csv(options.input)
            else:
                df_res = pd.DataFrame(
                    columns=['contrast', 'class', 'Model', 'original_accuracy'])

            df_res = df_res.append(
                {'contrast': contrast_name, 'class': classifier_no, 'Model': model_name,
                 'original_accuracy': scores}, ignore_index=True)

            df_res.to_csv(out + "permutation_result_%s_%s.csv" % (contrast_name, classifier_no), index=False)

            ## Only at the last iteration, permutation test is run 10000 times and using the performance scores and
            # mean of non-permutated accuracy of n_iterations, p-value can be calculated

            if i == n_iterations-1:
                x_train, y_train, x_test, y_test = mlu.preprocess_remove_gender(df)
                x_data = np.concatenate((x_train, x_test))
                y_data = np.concatenate((y_train, y_test))
                scores, permutation_scores, p_value = mlu.permutation_test(x_data, y_data, model, 10000, 10)
                performance_file = contrast_name[0]+contrast_name[-1]+"_"+classifier_no+"_"+model_name
                np.savetxt(out+"%s.csv" % performance_file, permutation_scores, fmt="%10.18f")



if __name__ == '__main__':
    print("Permutation_test")
    start = time.time()
    options = tools.parse_options()

    #mat_files = os.listdir(options.data)
    #contrast_list = list(filter(None, filter(lambda x: re.search('.*_.....mat', x), mat_files)))
    #n_back_list = list(filter(lambda x: 'nBack' in x and ('2' in x or '3' in x ), contrast_list))
    #faces_list = list(filter(lambda x: 'Faces' in x and ('5' in x or '4' in x), contrast_list))
    #relevant_mat_files = n_back_list  + faces_list

    #for mat_file in relevant_mat_files:
    mat_file = options.mat_file
    print(mat_file)
    df1, df2, df3, contrast_name = tools.data_extraction(options.data, 2, mat_file, 'face_aal')

    df1 = shuffle(df1) #.iloc[0:int(n_rows/2)]
    df2 = shuffle(df2) #.iloc[0:int(n_rows/2)]
    df3 = shuffle(df3) #.iloc[0:int(n_rows/2)]
        # Combining two pairs off all combination
    df12 = df1.append(df2)
    df23 = df2.append(df3)
    df31 = df3.append(df1)

        # Handle missing values
    df12 = mlu.missing_values(df12)
    df23 = mlu.missing_values(df23)
    df31 = mlu.missing_values(df31)

    #run_perm_test(df12, contrast_name, 12, options.output, options.number_iterations)
    #run_perm_test(df23, contrast_name, 23, options.output, options.number_iterations)
    #run_perm_test(df31, contrast_name, 31, options.output, options.number_iterations)

    df12 = mlu.merge_additional_data(df12, options.additional_data)
    df23 = mlu.merge_additional_data(df23, options.additional_data)
    df31 = mlu.merge_additional_data(df31, options.additional_data)

    if options.class_label == '12':
        df = df12
    elif options.class_label == '23':
        df = df23
    else:
        df = df31

    run_perm_test(df, contrast_name, options.class_label, options.output, options.number_iterations)


    print("It took %s seconds to perform permutation test on %s iterations "
          % (time.time() - start, options.number_iterations))












