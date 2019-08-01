from argparse import ArgumentParser

import json
import scipy.io as sio

import sys
import os
import pandas as pd
import numpy as np


def parse_options():
    parser = ArgumentParser()
    #parser.add_argument("-a", "--all", required=False, default=False,
    #                   action="store_true",
    #                   help="Run all the ML algorithms.")

    parser.add_argument("-n", "--number_iterations", required=False,
                        default=100, type=int,
                        help="Number of iterations to run the cross validation")

    parser.add_argument("-k", "--kFold", required=False,
                        default=10, type=int,
                        help="k fold number in Stratified Cross Validation")

    parser.add_argument("-d", "--data", required=False,
                        default="../../Data", type=str,
                        help="Path to data folder")

    parser.add_argument("-m", "--model", required=False,
                        default="all", type=str,
                        help="Model name to run. pass 'all' to run all the models")

    parser.add_argument("-o", "--output", required=False,
                        default="outputs/", type=str,
                        help="Output Folder where results are to stored")

    parser.add_argument("--missing_data", required=False,
                        default=0, type=int,
                        help="0-> fill it with 0; 1-> Mean Replacement; 2-> Median Replacement")


    parser.add_argument("-c", "--combine", required=False,
                        default=False, action="store_true",
                        help="An indicator to combine 2 contrasts side by side keeping subject number in mind")

    parser.add_argument("--normalize", required=False,
                        default=True, action="store_true",
                        help="An indicator to specify the normalization of both training and testing set before every "
                             "fold in cross validation of SVM RBF Kernel hyperparameter tuning ")

    parser.add_argument('-i', '--input', required=True,
                        default='out/output_scores_testing/Faces_con_0001&Faces_con_0001_389.csv', type=str,
                        help='Path to input csv file which contains information about the scores ')

    parser.add_argument('-dt', '--data_type', required=False,
                        default='face__aal' , type=str,
                        help='brodmann for Brodmann data type and face_aal for AAL data_type')

    parser.add_argument('-ad', '--additional_data', required=False,
                        default='../data_info', type=str,
                        help='Path to folder which contains additional information of the data')

    parser.add_argument('-ag','--age_gender', required=False,
                        default='age', type=str,
                        help='Pass age to run relevance of age info to the data and gender to check for gender')

    parser.add_argument('-mf','--mat_file', required=False,
                        default='nBack_con_0003.mat', type=str,
                        help='Matfile name to run experiments on a particular contrast. '
                        )
    parser.add_argument('-cl','--class_label', required=False,
                        default='12', type=str,
                        help='class labels: 1 for BD, 2: for Schizo and 3 for control. 12, 23 and 31 are for ' \
                             'combinations of the same ')
    options = parser.parse_args()
    return options


def data_extraction(data_folder, nClass, mat_file = "Faces_con_0001.mat", type='face_aal' ):
    """
    This function currently reads single contrast
    :param data_folder: Path to the folder that contains Data
    :param nClass: 2: for divinding the labels for biclass, 3: all 3 class in same dataframe
    :return: df: When nClass=3 Single panda dataframe containing means of various Region of interest (ROI) of Brain of all the three classes combined
            df1, df2, df3: Separated dataframes for each class when nClass is 2
    """

    # ----------------- Brodmann---------------------
    if type=='brodmann':
        contrast_name = mat_file.split(".")[0]
        data = sio.loadmat(data_folder + "/" + mat_file)

        # Extract Roi names for the DataFrame column names
        RoiNames = (data["roiName"][0])
        colRoi = []
        for i in range(len(RoiNames)):
            colRoi.append(data["roiName"][0][i][0])

        # prepare ROI data to add it to the dataFrame
        data_list = []
        [data_list.append(data["datas"][i]) for i in range(len(data["datas"]))]

        # Adding all values to the DataFrame: ROI, label and subject id
        df = pd.DataFrame(data_list, columns=colRoi, dtype=np.float64)
        df['label'] = pd.DataFrame(np.transpose(data['label']))
        df['subject_cont'] = pd.DataFrame(np.transpose(data['subjects']))
        if nClass == 3:  # No need for separated data
            return df, contrast_name

        elif nClass == 2:
            df1 = df[df.label == 1]
            df2 = df[df.label == 2]
            df3 = df[df.label == 3]
            return df1, df2, df3, contrast_name


      # ----------------- AAL all ROI---------------------
    else:
        contrast_name = mat_file.split(".")[0]
        data = sio.loadmat(data_folder+"/" + mat_file)
        data_list = []
        for i in range(len(data["means"])):
            d = data["means"][i], data["label"][0][i]
            data_list.append(d)
        columns = ["means", "label"]

        df = pd.DataFrame(data_list, columns=columns)
        RoiNames = (data["RoiName"][:, 0])
        colRoi = []
        for roi in RoiNames:
            colRoi.append(roi[0])
        df[colRoi] = pd.DataFrame(df.means.values.tolist(), index=df.index)
        df.drop(['means'], axis=1, inplace=True)
        df["subject_cont"] = pd.DataFrame(np.transpose(data["subject_cont"]))

        #print(df.shape)
        if nClass == 3: # No need for separated data
            return df,contrast_name

        elif nClass == 2:
            df1 = df[df.label == 1]
            df2 = df[df.label == 2]
            df3 = df[df.label == 3]
            return df1, df2, df3, contrast_name

def combine_contrast(data_folder,nClass, contrast1='Faces_con_0001.mat', contrast2='Faces_con_0001_389.mat',data_type='face_aal'):
    """
    Combines the two contrast with respect to subject_cont. It performs inner join. Returns panda dataframes according to nClass
    :param data_folder: Input folder which contains the contrast mat file
    :param nClass: 2: 3 dataframes of 2 labels combination, 3: all 3 class in same dataframe
    :param contrast1: First contrast for the merge
    :param contrast2: Second contrast for the merge
    :return: df: When nClass=3 Single panda dataframe containing means of various Region of interest (ROI) of Brain of all the three classes combined
            df1, df2, df3: Separated dataframes for each class when nClass is 2
    """
    df1,contrast1 = data_extraction(data_folder,3,contrast1, data_type)
    df2,contrast2 = data_extraction(data_folder,3,contrast2, data_type)

    df = pd.merge(df1,df2,how='inner',on='subject_cont')
    df['label'] = df['label_x'] & df['label_y']
    df.drop(['label_y','label_x'], axis=1, inplace=True)
    if nClass == 3: # No need for separated data
        return df, contrast1 + "&" + contrast2

    elif nClass == 2:
        df1 = df[df.label == 1]
        df2 = df[df.label == 2]
        df3 = df[df.label == 3]
        return df1, df2, df3, contrast1 + "&" + contrast2



def dump_results_to_json(model_name, results, output_folder, n, typeS="train"):
    """
    :param model_name: Machine learning model name
    :param results: Scores of kFold stratified cross Validation
    :param output_folder: Folder where the json has to be written
    :param n: option of classes. 12 or 23 or 31 or 123. Used for naming the files"
    :param typeS: train results or test results
    :return:
    """

    res_file = open(output_folder+"results_%s_%s_%s.json" % (model_name, typeS, n), "w", encoding='utf-8')
    # jsonList = [o.__dict__ for o in results]
    json_list = [o.tolist() for o in results]
    json.dumps(json_list)

    json.dump(json_list, res_file, sort_keys=True, indent=4)






