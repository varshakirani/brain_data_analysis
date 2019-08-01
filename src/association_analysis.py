import sys
import time #to check total time took for running the script or function
sys.path.insert(0, '../src/Utilities')
import pandas as pd
import tools
import ml_utilities as mlu
import os
import numpy as np
import re
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


def fdr(p, q):
    # Sort p-values
    p = np.sort(p)
    V = len(p)
    I = np.arange(1, V + 1)

    cVID = 1
    cVN = np.sum(1 / I)

    pID = p[p <= I / V * q / cVID].max() if len(p[p <= I / V * q / cVID]) else -100;
    pN = p[p <= I / V * q / cVN].max() if len(p[p <= I / V * q / cVN]) else -100;

    return (pID, pN)


def fdr_analysis(options):

    df = pd.read_csv(options.input)

    all_c = df['Contrast_name'].unique()
    face_c = []
    nback_c = []
    for c in all_c:
        if (len(c.split('_')) == 3) & ('Faces' in c):
            face_c.append(c)
        elif (len(c.split('_')) == 3) & ('nBack' in c):
            nback_c.append(c)

    groups = [12, 23, 31]
    tasks = ['nBack', 'Face']
    t = {'nBack': nback_c,
         'Face': face_c}
    results = {}
    sig_features = {}
    j = 0
    i = 0

    for group in groups:
        for task in tasks:

            pvalues = df[(df['Contrast_name'].isin(t[task])) & (df['Labels'] == group)]["pvalue_f"]
            p = np.asarray(pvalues)
            q = 0.05
            (pID, pN) = fdr(p, q)

            if pID != -100:
                for contrast in t[task]:

                    features = df[(df['Contrast_name'] == contrast)
                                  & (df['Labels'] == group)
                                  & (df['pvalue_f'] <= pID)]['feature'].values
                    f = {}
                    k = 0
                    for feature in features:
                        f[k] = {
                            'feature': feature,
                            'beta_f': df[(df['Contrast_name'] == contrast)
                                         & (df['Labels'] == group)
                                         & (df['pvalue_f'] <= pID) & (df['feature'] == feature)]['beta_f'].values[0],
                            'pvalue_f': df[(df['Contrast_name'] == contrast)
                                           & (df['Labels'] == group)
                                           & (df['pvalue_f'] <= pID) & (df['feature'] == feature)]['pvalue_f'].values[0]
                        }
                        k += 1

                    sig_features[j] = {
                        'contrast': contrast,
                        'group': group,
                        'pID': pID,
                        'significant_features': f

                    }
                    j += 1

            results[i] = {
                'task': task,
                'group': group,
                'pID': pID,
                'pN': pN
            }
            i += 1

            # results

    df = pd.DataFrame(columns=['Significant Features', 'Beta Value', 'p-Value'])
    for i in range(len(sig_features)):
        print(sig_features[i]['contrast'])
        print(sig_features[i]['group'])
        print(np.format_float_scientific(sig_features[i]['pID'], exp_digits=4, precision=4))
        for k, v in sig_features[i]['significant_features'].items():
            df = df.append({'Significant Features': v['feature'],
                            'Beta Value': np.around(v['beta_f'], decimals=3),
                            'p-Value': np.format_float_scientific(v['pvalue_f'], exp_digits=4, precision=4)},
                           ignore_index=True)
        print(df)

def prepare_y(y, labels):
    y_final = np.array([], dtype=np.int64)

    if labels == 12:
        first_label = 1
    elif labels == 23:
        first_label = 2
    elif labels == 31:
        first_label = 3

    for x in y:
        if x == first_label:
            y_final = np.append(y_final, np.array([[1, 0]], dtype=np.int64))
        else:
            y_final = np.append(y_final, np.array([[0, 1]], dtype=np.int64))

    y_final = y_final.reshape(int(len(y_final)/2), 2)

    return y_final


def run_glm_fit(df, labels, contrast_name, scoresdf):

    features = list(
        filter(lambda x: x != 'label' and x != 'subject_cont' and x != 'gender' and x != 'age', list(df.columns)))

    df_male = df.loc[df["gender"] == 1]
    df_female = df.loc[df["gender"] == 2]

    xtemp_male = df_male.loc[:, ~df_male.columns.isin(['subject_cont', 'gender', 'age', 'label'])].values
    ytemp_male = np.asarray(df_male.label).astype(int)
    x_age_male = np.asarray(df_male.age)
    x_gender_male = np.asarray(df_male.gender)
    x_male = StandardScaler().fit_transform(xtemp_male, ytemp_male)


    xtemp_female = df_female.loc[:, ~df_female.columns.isin(['subject_cont', 'gender', 'age', 'label'])].values
    ytemp_female = np.asarray(df_female.label).astype(int)
    x_age_female = np.asarray(df_female.age)
    x_gender_female = np.asarray(df_female.gender)

    x_female = StandardScaler().fit_transform(xtemp_female, ytemp_female)

    x = np.concatenate((x_male,x_female))
    y = np.concatenate((ytemp_male,ytemp_female))
    age = np.concatenate((x_age_male,x_age_female))
    gender = np.concatenate((x_gender_male,x_gender_female))


    for i in range(x.shape[1]):

        X = np.c_[ x[:,i], age, gender]
        Y = prepare_y(y,labels)
    #for feature in features:
    #    X = np.array(df[[feature, 'age', 'gender']])
    #    Y = prepare_y(np.array(df['label']), labels)
        model = sm.GLM(Y, X, family=sm.families.Binomial())
        results = model.fit()
        params = results.params
        pvalues = results.pvalues
        scoresdf = scoresdf.append(
            {'feature': features[i],
             'beta_f': params[0],
             'beta_a': params[1],
             'beta_g': params[2],
             'pvalue_f': pvalues[0],
             'pvalue_a': pvalues[1],
             'pvalue_g': pvalues[2],
             'Contrast_name': contrast_name,
             'Labels': labels}, ignore_index=True)

    return scoresdf


def main():

    options = tools.parse_options()
    start = time.time()

    ## Get Age, Gender and Subject_cont information ###

    file = open(options.additional_data + "subject_name.txt", "r")
    ids = file.read().split()
    ids = [int(float(id)) for id in ids]
    gdf = pd.read_csv(options.additional_data + 'n300.csv')
    gdf.loc[:, 'subject_cont'] = ids
    gdf = gdf[['KJØNN', 'subject_cont', 'ALDER']]
    gdf = gdf.rename(columns={'KJØNN': 'gender', 'ALDER': 'age'})

    mat_files = os.listdir(options.data)
    contrast_list = list(filter(None, filter(lambda x: re.search('.*_.....mat', x), mat_files)))
    n_back_list = list(filter(lambda x: 'nBack' in x and ('2' in x or '3' in x ), contrast_list))
    faces_list = list(filter(lambda x: 'Faces' in x and ('5' in x or '4' in x or '3' in x), contrast_list))
    relevant_contrast_list = n_back_list + faces_list  # extracted nBack 2,3 and Faces 3,4,5 contrasts

    if os.path.isfile(options.input):
        scoresdf = pd.read_csv(options.input)
    else:
        scoresdf = pd.DataFrame(columns=['feature', 'beta_f', 'beta_a', 'beta_g', 'pvalue_f', 'pvalue_a', 'pvalue_g',
                                         'Contrast_name', 'Labels'])

    for contrast in relevant_contrast_list:
        contrast_name = contrast.split(".")[0]
        if len(scoresdf[scoresdf["Contrast_name"] == contrast_name]):
            continue

        df1, df2, df3, contrast_name = tools.data_extraction(options.data, 2, contrast, options.data_type)

        # Combining two pairs off all combination
        df12 = df1.append(df2)
        df23 = df2.append(df3)
        df31 = df3.append(df1)

        # Handle missing values
        df12 = mlu.missing_values(df12)
        df23 = mlu.missing_values(df23)
        df31 = mlu.missing_values(df31)

        # Adding age and gender data for Standardization purpose. This additional data will be removed in
        # data preprocessing
        df12 = pd.merge(df12, gdf, on=['subject_cont'], how='inner')
        df23 = pd.merge(df23, gdf, on=['subject_cont'], how='inner')
        df31 = pd.merge(df31, gdf, on=['subject_cont'], how='inner')

        scoresdf = run_glm_fit(df12, 12, contrast_name, scoresdf)
        scoresdf = run_glm_fit(df23, 23, contrast_name, scoresdf)
        scoresdf = run_glm_fit(df31, 31, contrast_name, scoresdf)

        scoresdf.to_csv(options.output + "individual.csv", index=False)

    fdr_analysis(options)

if __name__ == "__main__":
    main()