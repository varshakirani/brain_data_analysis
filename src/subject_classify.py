import sys
sys.path.insert(0, '../src/Utilities')
import pandas as pd
import tools
import ml_utilities as mlu
import os
import time
import re
import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler


def run_basic_ml(df, options, n, scoresdf, contrast_name):
    print(contrast_name)
    models = ["svm_kernel_default", "svm_kernel_tuned", "naive_bayes", "decision_tree", "rfc", 'logistic_regression']

    for i in range(options.number_iterations):
        train, test = mlu.train_test_split(df)
        x_train, y_train = mlu.get_features_labels(train)
        x_test, y_test = mlu.get_features_labels(test)

        if options.model == "all":
            for model_name in models:

                train_score, train_balanced_score, trained_model, min_max_scaler = mlu.model_fitting(model_name, x_train, y_train, options.kFold, options.normalize)
                if options.normalize:
                    x_test_minmax = min_max_scaler.transform(x_test)
                    x_test = x_test_minmax

                test_score = trained_model.score(x_test, y_test)
                test_balanced_score = mlu.balanced_accuracy(trained_model.predict(x_test),y_test)

                scoresdf = scoresdf.append(
                    {'Score': train_score, 'Type': 'train', 'Model': model_name, 'Classifier': n,
                     'Contrast_name':contrast_name, 'Balanced_accuracy':train_balanced_score}, ignore_index=True)
                scoresdf = scoresdf.append(
                    {'Score': test_score, 'Type': 'test', 'Model': model_name, 'Classifier': n,
                     'Contrast_name':contrast_name,'Balanced_accuracy':test_balanced_score}, ignore_index=True)

        else:
            train_score, train_balanced_score, trained_model, min_max_scaler = mlu.model_fitting(options.model, x_train, y_train, options.kFold, True)
            test_score = trained_model.score(x_test, y_test)
            scoresdf = scoresdf.append(
                {'Score': train_score, 'Type': 'train', 'Model': options.model, 'Classifier': n,
                 'Contrast_name':contrast_name, 'Balanced_accuracy':train_balanced_score}, ignore_index=True)
            scoresdf = scoresdf.append(
                {'Score': test_score, 'Type': 'test', 'Model': options.model, 'Classifier': n,
                 'Contrast_name':contrast_name,'Balanced_accuracy':test_balanced_score}, ignore_index=True)

    return scoresdf


def contrast_permutation(contrast_list):
    combi_temp = []

    for r in itertools.product(contrast_list, contrast_list):
        if r[0] != r[1]:
            combi_temp.append(r)
    combi_contrast = list(map(tuple, set(map(frozenset, combi_temp))))
    return combi_contrast


def main():
    options = tools.parse_options()
    start = time.time()
    if options.combine:
        o_subtitle = 'combined'
    else:
        o_subtitle = 'individual'

    if os.path.isfile(options.input):  # if results are already stored then use that as input
        scoresdf = pd.read_csv(options.input)
    else:  # in previous experiments, if results are not stored then create new dataframe to store the results
        scoresdf = pd.DataFrame(columns=['Score', 'Type', 'Model', 'Classifier', 'Contrast_name', 'Balanced_accuracy'])

    mat_files = os.listdir(options.data)
    contrast_list = list(filter(None, filter(lambda x: re.search('.*_.....mat', x), mat_files)))
    combi_contrast = contrast_permutation(contrast_list)

    if options.combine:
        clist = combi_contrast
    else :
        clist = contrast_list

    for i in range(len(clist)):

    #Getting Contrast name
        if options.combine:
            c1_name = clist[i][0].split(".")[0]
            c2_name = clist[i][1].split(",")[0]
            contrast_name = c1_name + '&' + c2_name
        else:
            contrast_name = clist[i].split(".")[0]

        # Checking if the training is already made for the particular contrast
        # TODO Uncomment this for checking if contrast is present in the file
        if len(scoresdf[scoresdf['Contrast_name'] == contrast_name]):
            continue

        for nClass in range(2,4,1):

            if nClass == 3:

                # Read Data and put it into panda data frame. Initially considering only means
                if options.combine:
                    df, contrast_name = tools.combine_contrast(options.data, nClass, clist[i][0],clist[i][1], options.data_type)
                else:
                    df, contrast_name = tools.data_extraction(options.data, nClass, clist[i], options.data_type)
                df = mlu.missing_values(df)
                scoresdf = run_basic_ml(df, options, 123, scoresdf,contrast_name)


            elif nClass == 2:

                if options.combine:
                    df1, df2, df3, contrast_name = tools.combine_contrast(options.data, nClass, clist[i][0], clist[i][1], options.data_type)

                else:
                    df1, df2, df3, contrast_name = tools.data_extraction(options.data, nClass, clist[i], options.data_type)
                # Combining two pairs off all combination
                df12 = df1.append(df2)
                df23 = df2.append(df3)
                df31 = df3.append(df1)

                # Handle missing values
                df12 = mlu.missing_values(df12)
                df23 = mlu.missing_values(df23)
                df31 = mlu.missing_values(df31)

                scoresdf = run_basic_ml(df12, options, 12, scoresdf ,contrast_name)
                scoresdf = run_basic_ml(df23, options, 23, scoresdf ,contrast_name)
                scoresdf = run_basic_ml(df31, options, 31, scoresdf, contrast_name)


        scoresdf.to_csv(options.output + "basic_%s.csv" % (o_subtitle), index=False)


    print("It took %s seconds to run %s iterations for %s model" % (time.time() - start, options.number_iterations,
                                                                    options.model))

    print("It took %s seconds to run %s iterations for %s model after removing gender effect" % (time.time() - start, options.number_iterations,
                                                                    options.model))



if __name__ == '__main__':
    main()

