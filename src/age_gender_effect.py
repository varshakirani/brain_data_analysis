from scipy.stats import ttest_ind
import sys
sys.path.insert(0, '../src/Utilities')
import numpy as np
import statsmodels.api as sm
import pandas as pd
import os
import tools
import matplotlib.pyplot as plt
import seaborn as sns


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


def t_test(df, name, t_test_scores, task_name):
    t_scores = ttest_ind(np.asarray(df['label']), np.asarray(df['age']), equal_var=False)
    t_test_scores = t_test_scores.append({
        'statistic': t_scores[0],
        'pvalue': t_scores[1] ,
        'user group':name,
        'task_name' :task_name
    }, ignore_index=True)

    return t_test_scores


def plot_age_box_plot(df1, df2, df3, df, task_name , options):

    df1["Groups"] = "Bipolar Disorder"
    df2["Groups"] = "Schizophrenia"
    df3["Groups"] = "Control"
    df["Groups"] = "All"

    df_com = pd.concat([df1[["Groups", "age"]],df2[["Groups", "age"]],df3[["Groups", "age"]],df[["Groups", "age"]]])

    bplot = sns.boxplot(y='age', x='Groups',
                        data=df_com,
                        width=0.5,
                        palette="colorblind")

    bplot.axes.set_title(task_name,
                         fontsize=16)

    bplot.set_xlabel("Groups",
                     fontsize=14)

    bplot.set_ylabel("Age",
                     fontsize=14)

    bplot.tick_params(labelsize=10)

    bplot.figure.savefig('{}age_box_{}.png'.format(options.output, task_name),
                         dpi=100)

    plt.show()


def plot_age_dist(df1, df2, df3, task_name , options):
    sns.distplot(df1['age'], kde=False, rug=True, label="Bipolar Disorder subjects")
    sns.distplot(df2['age'], kde=False, rug=True, label="Schizophrenia subjects")
    sns.distplot(df3['age'], kde=False, rug=True, label="Control subjects")
    plt.title(task_name)
    plt.legend()
    plt.savefig('{}age_dis_{}.png'.format(options.output, task_name))
    plt.show()


def plot_gender_dist(options):
    barwidth = 0.25

    bars1 = [40, 66, 50]
    bars2 = [60, 33, 50]
    r1 = np.arange(len(bars1))
    r2 = [x + barwidth for x in r1]

    plt.bar(r1, bars1, color='#7f6d5f', width=barwidth, edgecolor='white', label='Male Count')
    plt.bar(r2, bars2, color='#557f2d', width=barwidth, edgecolor='white', label='Female Count')

    plt.xlabel('User Group', fontweight='bold')
    plt.xticks([r + barwidth for r in range(len(bars1))], ['Bipolar Disorder', 'Schizophrenia', 'Control'])

    plt.legend()
    plt.savefig('{}gender_dis.png'.format(options.output))
    plt.show()


def run_glm_fit(df, labels,scoresdf, param, task_name):

    # The scores for nback task and face task are different. Hence only one contrast from each of the task is
    # considered
    X = np.array(df[[param]])
    y = prepare_y(np.array(df['label']), labels)
    model = sm.GLM(y, X, family=sm.families.Binomial())
    results = model.fit()
    params = results.params
    pvalues = results.pvalues
    scoresdf = scoresdf.append({
             'beta': params[0],
             'pvalue': pvalues[0],
             'Labels': labels,
             'variable': param,
             'task_name': task_name}, ignore_index=True)

    return scoresdf


def main():
    options = tools.parse_options()

    data = options.data
    additional_data = options.additional_data

    file = open(additional_data + "/subject_name.txt", "r")
    ids = file.read().split()
    ids = [int(float(id)) for id in ids]
    gdf = pd.read_csv(additional_data + '/n300.csv')
    gdf['subject_cont'] = ids
    gdf = gdf[['KJØNN', 'subject_cont', 'ALDER']].copy()
    gdf = gdf.rename(columns={'KJØNN': 'gender', 'ALDER': 'age'})

    mat_files = os.listdir(data)
    n_back_file = list(filter(lambda x: 'nBack' in x, mat_files))[0]
    face_file = list(filter(lambda x: 'Faces' in x, mat_files))[0]
    contrasts = [n_back_file, face_file]
    t_test_scores = pd.DataFrame(columns=['statistic', 'pvalue', 'user group', 'task_name'])
    scoresdf = pd.DataFrame(columns=['beta', 'pvalue', 'Labels', 'variable', 'task_name'])
    params = ['age', 'gender']
    print(contrasts)
    for mat_file in contrasts:
        for param in params:
            df1, df2, df3, contrast_name = tools.data_extraction(data, 2, mat_file)
            df1.fillna(df1.mean(), inplace=True)
            df2.fillna(df2.mean(), inplace=True)
            df3.fillna(df3.mean(), inplace=True)

            df1 = pd.merge(df1, gdf, on=['subject_cont'], how='inner')
            df2 = pd.merge(df2, gdf, on=['subject_cont'], how='inner')
            df3 = pd.merge(df3, gdf, on=['subject_cont'], how='inner')

            df = df1.append(df2).append(df3)
            df = df.loc[:, df.columns.intersection([param, 'label'])]

            df12 = df1.append(df2)
            df23 = df2.append(df3)
            df31 = df3.append(df1)

            task_name = mat_file.split("_")[0]
            if param == "age":
                plot_age_box_plot(df1, df2, df3, df, task_name, options)

            scoresdf = run_glm_fit(df12, 12, scoresdf, param, task_name)
            scoresdf = run_glm_fit(df23, 23, scoresdf, param, task_name)
            scoresdf = run_glm_fit(df31, 31, scoresdf, param, task_name)

            t_test_scores = t_test(df12, "BD-Sc", t_test_scores, task_name)
            t_test_scores = t_test(df23, "Sc-Co", t_test_scores, task_name)
            t_test_scores = t_test(df31, "Co-BD", t_test_scores, task_name)

        plot_age_dist(df1, df2, df3, task_name, options)
    plot_gender_dist(options)
    print("\nGLM fit with age and gender variable used individually\n")
    print(scoresdf)

    print("\n\nT-test scores to analyse age distribution\n")
    print(t_test_scores)


if __name__ == "__main__":
    main()