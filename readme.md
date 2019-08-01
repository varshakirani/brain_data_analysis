
    Data Folder : Mat files i.e contrasts
    data_info : Additional information 


##### age_gender_effect.py 

This script is used to analyse the correlation of age 
and gender with user groups and visualization of its 
distributions. 

     python age_gender_effect.py -d="../../Data/" -ad="../../data_info/" -o="../outputs/age_gender_effect/" -i=""

    
##### age_gender_predict.py

This script is used in prediction of age and gender of the subjects 
using mean activation score

To predict age:

    python age_gender_predict.py -d="../../Data/" -ad="../../data_info" -i="../outputs/age_gender_effect/age_predict.csv" -o="../outputs/age_gender_effect/" -ag="age" -n=100

To predict gender:

    python age_gender_predict.py -d="../../Data/" -ad="../../data_info" -i="../outputs/age_gender_effect/age_predict.csv" -o="../outputs/age_gender_effect/" -ag="gender" -n=100

To visualize classified gender results:

    python plotting.py -i="../outputs/age_gender_effect/gender_predict.csv" -o="../outputs/age_gender_effect/" -gp


##### subject_classify.py

This script runs 5 ml models on brain data to classify the 
subjects into bipolar, schizo or control category. 

This experiment is run on all the contrast individually and
combined

For individual:
    
    python subject_classify.py -d="../../Data/" -i="../outputs/subject_classify/basic_individual.csv" -o="../outputs/subject_classify/" -n=100
    
To visualize :

    python plotting.py -i="../outputs/subject_classify/basic_individual.csv" -o="../outputs/subject_classify/" -p

For combined contrasts:

    python subject_classify.py -d="../../Data/" -i="../outputs/subject_classify/basic_combined.csv" -o="../outputs/subject_classify/" -n=100 -c
    

##### no_gender.py

This script runs 5 ml models on selective contrasts after
removal of gender effect

    python no_gender.py -d="../../Data/" -ad="../../data_info" -i="../outputs/subject_classify/no_gender_individual.csv" -o="../outputs/subject_classify/" -n=100
    
To visualize :

    python plotting.py -i="../outputs/subject_classify/no_gender_individual.csv" -o="../outputs/subject_classify/" -p

##### permutation_test.py

This script is used to run label permutation test on 
Face_contrasts 3,4,5 and n_back_contrasts 2, 3 after removal of 
gender effect

Command to run for single contrast

    python permutation_test.py -d="../../Data" -n=30 -o="../outputs/no_gender/permutation/" -i="../outputs/no_gender/permutation/permutation_result_nBack_con_0003_12.csv" --missing_data=1 -mf='nBack_con_0003.mat' -cl='12' -ad="../../data_info/"
    
To visualize the results:
    
    python plotting.py -i=""  -o="../outputs/no_gender/permutation/" -if="../outputs/no_gender/permutation/" -pp
