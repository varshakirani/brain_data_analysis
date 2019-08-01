import sys
import time
sys.path.insert(0, '../src/Utilities')
import pandas as pd
import tools
import ml_utilities as mlu
import os
import re
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def run_gender_cor(df, options, n, scoresdf, contrast_name, label):
    classification = True
    if label == 'gender':
        df.drop(['label', 'age'], axis=1, inplace=True)
        models = ["svm_kernel_default", "svm_kernel_tuned", "naive_bayes", "decision_tree", "rfc",
                  'logistic_regression']

    elif label == 'age':
        df.drop(['label', 'gender'], axis=1, inplace=True)
        models = ['linear_reg', 'lasso', 'polynomial_reg']
        models = ['svr_kernel_default', 'svr_kernel_tuned', 'gpr_default']
        classification = False

    df = df.rename(columns={label: 'label'})

    for i in range(options.number_iterations):
        train, test = mlu.train_test_split(df)
        x_train, y_train = mlu.get_features_labels(train)
        x_test, y_test = mlu.get_features_labels(test)
        if classification:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train, y_train)
            x_test = scaler.transform(x_test)

        if options.model == 'all':
            for model_name in models:
                train_score, train_balanced_score, trained_model, min_max_scaler = mlu.model_fitting(model_name,
                                                                                                     x_train, y_train,
                                                                                                     options.kFold,
                                                                                                     options.normalize)
                if options.normalize:
                    x_test_minmax = min_max_scaler.transform(x_test)
                    x_test = x_test_minmax

                test_score = trained_model.score(x_test, y_test)

                test_balanced_score = mlu.balanced_accuracy(trained_model.predict(x_test), y_test)

                if not classification:

                    if model_name == "gpr_default":
                        pred, sigma = trained_model.predict(x_test, return_std=True)
                    else:
                        pred = trained_model.predict(x_test)
                    test_balanced_score = mean_squared_error(y_test, pred, multioutput='raw_values')

                # print(model_name + " Train:"+ str(train_score) + "  Test:" +str(test_score) +" Contrast:" +contrast_name)
                scoresdf = scoresdf.append(
                    {'Score': train_score, 'Type': 'train', 'Model': model_name, 'Classifier': n,
                     'Contrast_name': contrast_name, 'Balanced_accuracy': train_balanced_score}, ignore_index=True)
                scoresdf = scoresdf.append(
                    {'Score': test_score, 'Type': 'test', 'Model': model_name, 'Classifier': n,
                     'Contrast_name': contrast_name, 'Balanced_accuracy': test_balanced_score}, ignore_index=True)
        else:
            train_score, train_balanced_score, trained_model, min_max_scaler = mlu.model_fitting(options.model, x_train, y_train)
            test_score = trained_model.score(x_test, y_test)
            test_balanced_score = mlu.balanced_accuracy(trained_model.predict(x_test), y_test)
            scoresdf = scoresdf.append(
                {'Score': train_score, 'Type': 'train', 'Model': options.model, 'Classifier': n,
                 'Contrast_name': contrast_name, 'Balanced_accuracy': train_balanced_score}, ignore_index=True)
            scoresdf = scoresdf.append(
                {'Score': test_score, 'Type': 'test', 'Model': options.model, 'Classifier': n,
                 'Contrast_name': contrast_name, 'Balanced_accuracy': test_balanced_score}, ignore_index=True)

    return scoresdf


if __name__ == "__main__":
    options = tools.parse_options()

    if os.path.isfile(options.input):
        scoresdf = pd.read_csv(options.input)
    else:
        scoresdf = pd.DataFrame(columns=['Score', 'Type', 'Model', 'Classifier', 'Contrast_name', 'Balanced_accuracy'])


    ## Gender information and adding it as label to the data by linking the subject_cont
    file = open(options.additional_data + "/subject_name.txt", "r")
    ids = file.read().split()
    ids = [int(float(id)) for id in ids]
    gdf = pd.read_csv(options.additional_data + '/n300.csv')
    gdf['subject_cont'] = ids
    gdf = gdf[['KJØNN', 'subject_cont','ALDER']].copy()
    gdf = gdf.rename(columns={'KJØNN':'gender', 'ALDER':'age'})

    label = options.age_gender

    mat_files = os.listdir(options.data)
    contrast_list = list(filter(None, filter(lambda x: re.search('.*_.....mat', x), mat_files)))
    n_back_list = list(filter(lambda x: 'nBack' in x and ('2' in x or '3' in x), contrast_list))
    faces_list = list(filter(lambda x: 'Faces' in x and ('5' in x or '4' in x or '3' in x), contrast_list))
    relevant_mat_files = n_back_list + faces_list

    for mat_file in relevant_mat_files:
        print(mat_file)
        for nClass in range(2, 4, 1):
            if nClass == 3:
                df, contrast_name = tools.data_extraction(options.data, nClass, mat_file, options.data_type)
                # Adding Age and gender to the dataframe
                df = pd.merge(df, gdf, on=['subject_cont'], how='inner')
                df = mlu.missing_values(df)
                scoresdf = run_gender_cor(df, options, 123, scoresdf, contrast_name, label)

            elif nClass == 2:
                df1, df2, df3, contrast_name = tools.data_extraction(options.data, nClass, mat_file, options.data_type)

                #Adding Age and gender to the dataframe
                df1 = pd.merge(df1, gdf, on=['subject_cont'], how='inner')
                df2 = pd.merge(df2, gdf, on=['subject_cont'], how='inner')
                df3 = pd.merge(df3, gdf, on=['subject_cont'], how='inner')

                # Combining two pairs off all combination
                df12 = df1.append(df2)
                df23 = df2.append(df3)
                df31 = df3.append(df1)

                # Handle missing values
                df12 = mlu.missing_values(df12)
                df23 = mlu.missing_values(df23)
                df31 = mlu.missing_values(df31)

                df1 = mlu.missing_values(df1)
                df2 = mlu.missing_values(df2)
                df3 = mlu.missing_values(df3)

                scoresdf = run_gender_cor(df1, options, 1, scoresdf, contrast_name, label)
                scoresdf = run_gender_cor(df2, options, 2, scoresdf, contrast_name, label)
                scoresdf = run_gender_cor(df3, options, 3, scoresdf, contrast_name, label)

        scoresdf.to_csv(options.output + "{}_predict.csv".format(label), index=False)