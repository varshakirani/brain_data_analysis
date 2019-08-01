from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def parse_options():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", required=False,
                        default="outputs", type=str,
                        help="Path to output folder")

    parser.add_argument('-i','--input', required=True,
                        default='out/output_scores_testing/Faces_con_0001&Faces_con_0001_389.csv', type=str,
                        help='Path to input csv file which contains information about the scores ')
    parser.add_argument("-t", "--type", required=False,
                        default=1, type=int,
                        help="0 for old files and 1 for new files ")
    parser.add_argument("-p", '--performance_plot', required=False,
                       default=False, action="store_true",
                       help="Performance of different Contrasts")
    parser.add_argument("-gp", '--gender_performance_plot', required=False,
                        default=False, action="store_true",
                        help="Classification of gender using brain data")
    parser.add_argument("-pp", '--permutation_plot', required=False,
                        default=False, action="store_true",
                        help="Label permutation plots")
    parser.add_argument("-c", '--combined_contrast', required=False,
                        default=False, action="store_true",
                        help="If the input file is for combined contrasts")
    parser.add_argument('-if', '--input_folder', required=False,
                        default='../outputs/no_gender/permutation', type=str,
                        help='Path to folder containing permutations results ')

    options = parser.parse_args()
    return options


def calculate_pvalue(raw_folder, df_orig):
    per_files = list(filter(lambda x: x.startswith('n') or x.startswith('F'), os.listdir(raw_folder)))
    contrast_names = {
        'n2' : 'nBack_con_0002',
        'n3' : 'nBack_con_0003',
        'F3' : 'Faces_con_0003',
        'F4' : 'Faces_con_0004',
        'F5' : 'Faces_con_0005'
    }
    df_pvalues = pd.DataFrame(columns=['contrast', 'Model', 'class', "Avg original accuracy", "p_value"])

    for file in per_files:
        contrast = contrast_names[file[0:2]]
        classifier = file[3:5]
        model = file[6:].split(".")[0]

        original_accuracy = df_orig[(df_orig['contrast'] == contrast) & (df_orig['Model'] == model)
                                    & (df_orig['class'] == int(classifier) )]["Avg original accuracy"].values[0]

        df_perm = pd.read_csv(raw_folder+file, header=None, names=['performance_scores'])

        #performance_scores = np.asarray(df_perm['performance_scores'])
        #p_value = sum(performance_scores >= original_accuracy)/10000
        p_value = (df_perm[df_perm['performance_scores'] >= original_accuracy]['performance_scores'].count() + 1)/(10000 +1)
        df_pvalues = df_pvalues.append({'contrast': contrast,
                                        "Model":model,
                                        "class":int(classifier),
                                        "Avg original accuracy":original_accuracy,
                                       "p_value":p_value},
                                       ignore_index=True)

    return df_pvalues

def permutation_plots(df, output):
    df = df.sort_values("Model")
    all_c = df['contrast'].unique()
    face_c = []
    nback_c = []
    for c in all_c:
        if (len(c.split('_')) == 3) & ('Faces' in c):
            face_c.append(c)
        elif (len(c.split('_')) == 3) & ('nBack' in c):
            nback_c.append(c)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))

    axs = axes.ravel()

    y_val = 'p_value'
    sns.pointplot(x='Model', y=y_val, hue='contrast',
                  data=df[(df['class'] == 12) &
                          (df['contrast'].isin(face_c))], ax=axs[0]).set_title('Faces\nBipolar Disorder and Schizophrenia', fontsize=16)

    sns.pointplot(x='Model', y=y_val, hue='contrast',
                  data=df[(df['class'] == 23) &
                          (df['contrast'].isin(face_c))], ax=axs[1]).set_title('Faces\nSchizophrenia and Control', fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='contrast',
                  data=df[(df['class'] == 31) &
                          (df['contrast'].isin(face_c))], ax=axs[2]).set_title('Faces\nControl and Bipolar Disorder', fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='contrast',
                  data=df[(df['class'] == 12) &
                          (df['contrast'].isin(nback_c))], ax=axs[3]).set_title('nBack\nBipolar Disorder and Schizophrenia', fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='contrast',
                  data=df[(df['class'] == 23) &
                          (df['contrast'].isin(nback_c))], ax=axs[4]).set_title('nBack\nSchizophrenia and Control', fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='contrast',
                  data=df[(df['class'] == 31) &
                          (df['contrast'].isin(nback_c))], ax=axs[5]).set_title('nBack\nControl and Bipolar Disorder', fontsize=16)

    for i in range(6):
        axs[i].set_xticklabels(['lr', 'rfc', 'rbf_b', 'rbf'])
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')
        axs[i].set_ylim(0.0001,1)
        axs[i].set(yscale="log")
        axs[i].tick_params(labelsize=20)
        axs[i].axhline(y=0.05, xmax=1, xmin=0, linestyle="--", color="grey")
        if i == 2 or i == 5:
            continue
        axs[i].legend_.remove()
    for i in range(3, 6, 1):
        axs[i].set_xlabel('Models', fontsize=22)
    for i in range(0,4,3):
        axs[i].set_ylabel('p-value' , fontsize=22)

    fig.suptitle('Significance of model classification accuracy using Permutation test after removal of gender information', fontsize=22)
    plt.savefig("%sOverallPerformance.png" % (output))
    #plt.show()
    plt.cla()
    plt.clf()
    plt.close()


def get_avg_original_accuracy(raw_folder):
    original_files = list( filter(lambda x: x.startswith('permutation_result_'), os.listdir(raw_folder)))
    result = pd.DataFrame(columns=['contrast', 'class', 'Model', 'Avg original accuracy'])
    for file in original_files:
        df = pd.read_csv(raw_folder+file)
        res = pd.DataFrame({"Avg original accuracy":
                        df.groupby(['contrast','class','Model'])['original_accuracy'].mean()}).reset_index()
        result = pd.concat([res,result], ignore_index=True)
    return result


def gender_performance_plots(df, output):

    all_c = df['Contrast_name'].unique()
    face_c = []
    nback_c = []
    for c in all_c:
        if (len(c.split('_')) == 3) & ('Faces' in c):
            face_c.append(c)
        elif (len(c.split('_')) == 3) & ('nBack' in c):
            nback_c.append(c)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 20))
    axs = axes.ravel()

    y_val = 'Balanced_accuracy'
    print(df[df["Classifier"] == 1])
    print(df[(df['Classifier'] == 1) & (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')])
    sns.pointplot(x='Model',y=y_val,hue='Contrast_name',
                  data=df[(df['Classifier'] == 1) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[0]).set_title('Faces\nBipolar Disorder', fontsize=16)

    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 2) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[1]).set_title('Faces\nSchizophrenia', fontsize=16)
    sns.pointplot(x='Model',y=y_val,hue='Contrast_name',
                  data=df[(df['Classifier'] == 3) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[2]).set_title('Faces\nControl', fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 123) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[3]).set_title('Faces\nBipolar, Schizo, Control', fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 1) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[4]).set_title('nBack\nBipolar Disorder', fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 2) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[5]).set_title('nBack\nSchizophrenia', fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 3) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[6]).set_title('nBack\nControl', fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 123) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[7]).set_title('nBack\nBipolar, Schizo, Control', fontsize=16)
    for i in range(8):
        axs[i].set_xticklabels(['svm', 'svm\ntuned', 'nb', 'dt', 'rf', 'lr'])
        axs[i].set_xlabel('')
        axs[i].set_ylim(0.25,0.80)
        axs[i].set_ylabel('')
        axs[i].tick_params(labelsize=16)
        if i == 3 or i == 7:
            axs[i].axhline(y=0.33, xmax=1, xmin=0, linestyle="--", color="grey")
            continue
        else:
            axs[i].axhline(y=0.5, xmax=1, xmin=0, linestyle="--", color="grey")


        axs[i].legend_.remove()

    for i in range(0,5,4):
        axs[i].set_ylabel("Balanced Accuracy", fontsize=22)
    for i in range(4,8,1):
        axs[i].set_xlabel('Models', fontsize=22)

    # For basic classification
    #fig.suptitle('Classification Results for All Contrasts before removing gender effect', fontsize=24)
    #plt.savefig("%sbasic_individual.png" % (output))

    # For classification after gender effect removal
    #fig.suptitle('Classification Results for All Contrasts after removing gender effect', fontsize=24)
    #plt.savefig("%sno_gender_individual.png" % (output))


    #For gender classification
    fig.suptitle('Gender Classification Results for All Contrasts', fontsize=24)
    plt.savefig("%sgender_classification.png" % (output))
    plt.cla()
    plt.clf()
    plt.close()


def performance_plots(df, output):

    all_c = df['Contrast_name'].unique()
    face_c = []
    nback_c = []
    for c in all_c:
        if (len(c.split('_')) == 3) & ('Faces' in c):
            face_c.append(c)
        elif (len(c.split('_')) == 3) & ('nBack' in c):
            nback_c.append(c)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 20))
    axs = axes.ravel()

    y_val = 'Score'
    y_val = 'Balanced_accuracy'

    sns.pointplot(x='Model',y=y_val,hue='Contrast_name',
                  data=df[(df['Classifier'] == 12) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[0]).set_title('Faces\nBipolar Disorder and Schizophrenia' , fontsize=16)

    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 23) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[1]).set_title('Faces\nSchizophrenia and Control' , fontsize=16)
    sns.pointplot(x='Model',y=y_val,hue='Contrast_name',
                  data=df[(df['Classifier'] == 31) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[2]).set_title('Faces\nControl and Bipolar Disorder' , fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 123) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[3]).set_title('Faces\nBipolar, Schizo, Control' , fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 12) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[4]).set_title('nBack\nBipolar Disorder and Schizophrenia' , fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 23) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[5]).set_title('nBack\nSchizophrenia and Control' , fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 31) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[6]).set_title('nBack\nControl and Bipolar Disorder' , fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 123) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[7]).set_title('nBack\nBipolar, Schizo, Control' , fontsize=16)
    for i in range(8):
        axs[i].set_xticklabels(['svm', 'svm\ntuned', 'nb', 'dt', 'rf', 'lr'])
        axs[i].set_xlabel('')
        axs[i].set_ylim(0.25,0.80)
        axs[i].set_ylabel('')
        axs[i].tick_params(labelsize=16)
        if i == 3 or i == 7 :
            axs[i].axhline(y=0.333, xmax=1, xmin=0, linestyle="--", color="grey")
            continue
        else:
            axs[i].axhline(y=0.5, xmax=1, xmin=0, linestyle="--", color="grey")
        axs[i].legend_.remove()

    for i in range(0,5,4):
        axs[i].set_ylabel("Balanced Accuracy", fontsize=22)
    for i in range(4,8,1):
        axs[i].set_xlabel('Models', fontsize=22)

    # For basic classification
    #fig.suptitle('Classification Results for All Contrasts before removing gender effect', fontsize=24)
    #plt.savefig("%sbasic_individual.png" % (output))

    # For classification after gender effect removal
    fig.suptitle('Classification Results for All Contrasts after removing gender effect', fontsize=24)
    plt.savefig("%sno_gender_individual.png" % (output))

    plt.cla()
    plt.clf()
    plt.close()

def combined_performance_plots(df, output):

    all_c = df['Contrast_name'].unique()
    face_c = all_c[0:14]
    nback_c = all_c[14:28]

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 20))
    axs = axes.ravel()

    y_val = 'Score'
    y_val = 'Balanced_accuracy'

    sns.pointplot(x='Model',y=y_val,hue='Contrast_name',
                  data=df[(df['Classifier'] == 12) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[0]).set_title('Bipolar Disorder and Schizophrenia' , fontsize=16)

    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 23) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[1]).set_title('Schizophrenia and Control' , fontsize=16)
    sns.pointplot(x='Model',y=y_val,hue='Contrast_name',
                  data=df[(df['Classifier'] == 31) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[2]).set_title('Control and Bipolar Disorder' , fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 123) &
                          (df['Contrast_name'].isin(face_c)) & (df['Type'] == 'test')], ax=axs[3]).set_title('Bipolar, Schizo, Control' , fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 12) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[4]).set_title('Bipolar Disorder and Schizophrenia' , fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 23) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[5]).set_title('Schizophrenia and Control' , fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 31) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[6]).set_title('Control and Bipolar Disorder' , fontsize=16)
    sns.pointplot(x='Model', y=y_val, hue='Contrast_name',
                  data=df[(df['Classifier'] == 123) &
                          (df['Contrast_name'].isin(nback_c)) & (df['Type'] == 'test')], ax=axs[7]).set_title('Bipolar, Schizo, Control' , fontsize=16)
    for i in range(8):
        axs[i].set_xticklabels(['svm', 'svm\ntuned', 'nb', 'dt', 'rf', 'lr'])
        axs[i].set_xlabel('')
        axs[i].set_ylim(0.25,0.80)
        axs[i].set_ylabel('')
        axs[i].tick_params(labelsize=16)
        if i == 3 or i == 7 :
            axs[i].axhline(y=0.33, xmax=1, xmin=0, linestyle="--", color="grey")
            continue
        else:
            print("else")
            axs[i].axhline(y=0.5, xmax=1, xmin=0, linestyle="--", color="grey")
        axs[i].legend_.remove()

    for i in range(0,5,4):
        axs[i].set_ylabel("Balanced Accuracy", fontsize=22)
    for i in range(4,8,1):
        axs[i].set_xlabel('Models', fontsize=22)
    # For basic combined classification
    fig.suptitle('Classification Results using combination of 2 contrasts', fontsize=24)
    plt.savefig("%sbasic_combined.png" % (output))

    plt.cla()
    plt.clf()
    plt.close()

def main():
    options = parse_options()

    if os.path.isfile(options.input):
        print("File is present")
        scoresdf = pd.read_csv(options.input)
    else:
        scoresdf = pd.DataFrame()
        print("If you are tring to plot anything other than permutation, provide input file")


    if options.performance_plot:
        #for plotting classification accuracy
        performance_plots(scoresdf, options.output)

    elif options.gender_performance_plot:
        gender_performance_plots(scoresdf, options.output)

    elif options.permutation_plot:
        df = get_avg_original_accuracy(options.input_folder)
        df_pvalues = calculate_pvalue(options.input_folder, df)
        print(df_pvalues[df_pvalues['p_value'] <= 0.05].sort_values('contrast'))
        permutation_plots(df_pvalues, options.output)

    elif options.combined_contrast :
        scoresdf = scoresdf.replace('Faces_con_0002&Faces_con_0001', 'Faces_con_0001&Faces_con_0002')
        scoresdf = scoresdf.replace('Faces_con_0002&nBack_con_0001', 'nBack_con_0001&Faces_con_0002')
        scoresdf = scoresdf.replace('Faces_con_0005&Faces_con_0003', 'Faces_con_0003&Faces_con_0005')
        scoresdf = scoresdf.replace('Faces_con_0003&nBack_con_0001', 'nBack_con_0001&Faces_con_0003')
        scoresdf = scoresdf.replace('Faces_con_0002&Faces_con_0005', 'Faces_con_0005&Faces_con_0002')
        scoresdf = scoresdf.replace('Faces_con_0004&nBack_con_0003', 'nBack_con_0003&Faces_con_0004')
        scoresdf = scoresdf.replace('Faces_con_0001&Faces_con_0005', 'Faces_con_0005&Faces_con_0001')
        scoresdf = scoresdf.replace('nBack_con_0003&nBack_con_0001', 'nBack_con_0001&nBack_con_0003')
        scoresdf = scoresdf.replace('Faces_con_0004&Faces_con_0003', 'Faces_con_0003&Faces_con_0004')
        scoresdf = scoresdf.replace('Faces_con_0002&nBack_con_0002', 'nBack_con_0002&Faces_con_0002')
        scoresdf = scoresdf.replace('nBack_con_0003&Faces_con_0003', 'Faces_con_0003&nBack_con_0003')
        scoresdf = scoresdf.replace('Faces_con_0004&nBack_con_0001','nBack_con_0001&Faces_con_0004' )
        scoresdf = scoresdf.replace('Faces_con_0001&nBack_con_0002',  'nBack_con_0002&Faces_con_0001')
        scoresdf = scoresdf.replace('Faces_con_0003&nBack_con_0002',   'nBack_con_0002&Faces_con_0003')
        scoresdf = scoresdf.replace('Faces_con_0001&nBack_con_0001', 'nBack_con_0001&Faces_con_0001')
        scoresdf = scoresdf.replace('Faces_con_0004&Faces_con_0001', 'Faces_con_0001&Faces_con_0004')
        scoresdf = scoresdf.replace('Faces_con_0001&Faces_con_0003', 'Faces_con_0003&Faces_con_0001')
        scoresdf = scoresdf.replace('Faces_con_0002&Faces_con_0003', 'Faces_con_0003&Faces_con_0002')
        scoresdf = scoresdf.replace('Faces_con_0004&Faces_con_0005', 'Faces_con_0005&Faces_con_0004')
        scoresdf = scoresdf.replace('Faces_con_0004&Faces_con_0002', 'Faces_con_0002&Faces_con_0004')
        scoresdf = scoresdf.replace('Faces_con_0005&nBack_con_0002', 'nBack_con_0002&Faces_con_0005')
        scoresdf = scoresdf.replace('Faces_con_0005&nBack_con_0003', 'nBack_con_0003&Faces_con_0005')
        scoresdf = scoresdf.replace('nBack_con_0002&nBack_con_0001', 'nBack_con_0001&nBack_con_0002')
        scoresdf = scoresdf.replace('Faces_con_0002&nBack_con_0003', 'nBack_con_0003&Faces_con_0002')
        scoresdf = scoresdf.replace('Faces_con_0001&nBack_con_0003', 'nBack_con_0003&Faces_con_0001')
        scoresdf = scoresdf.replace('Faces_con_0004&nBack_con_0002', 'nBack_con_0002&Faces_con_0004')
        scoresdf = scoresdf.replace('Faces_con_0005&nBack_con_0001', 'nBack_con_0001&Faces_con_0005')
        scoresdf = scoresdf.replace('nBack_con_0003&nBack_con_0002', 'nBack_con_0002&nBack_con_0003')

        combined_performance_plots(scoresdf, options.output)
if __name__ == '__main__':
    main()