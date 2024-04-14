# Here importing all the necssary libraries and functions
import numpy as np
import collections
import random
import pandas as pd
import matplotlib.pyplot as plt
from ML_functions import ns_ML_model_test
import scipy as sp
from tqdm import tqdm
plt.rcParams["font.family"] = "Arial" 

def binary_column_ratio(col_name, name_bigger_no, csv_frame):
    uni_val = np.unique(csv_frame.loc[:,col_name])
    uni_val = sorted(uni_val)
    
    if(len(uni_val)==3):
        Nan_val = len(csv_frame[csv_frame.loc[:,col_name] == uni_val[2]])
        print("There are {} Nan values in {} column".format(Nan_val,name_bigger_no))
    
    big_ind = len(csv_frame[csv_frame.loc[:,col_name] == uni_val[1]])
    total = len(csv_frame)
    print("Total subjects with {}: {}/{} => {}%".format(name_bigger_no,big_ind,total,round(big_ind*100/total,2)))
    

#Impute the missing data in discrete, ordinal AND categorical variables by replacing with random labels with same distribution as original data \n",
# Mean for categorical variables
# con_feat_list, dis_feat_list,dataframe = continuous_var,discrete_var,csv_frame2
def fill_conti_discrete_nan(con_feat_list,dis_feat_list,dataframe):
    
    # For continuous variable => Mean value
    dataframe.loc[:,con_feat_list]=dataframe.loc[:,con_feat_list].fillna(dataframe.loc[:,con_feat_list].mean(skipna=True))
    
    #  For discrete variable => Probability distribution
    for feature in dis_feat_list:
        random.seed(0) # fixing seed so that results do not vary each time
        num_na=np.isnan(dataframe.loc[:,feature]).sum()
        na_index = dataframe[np.isnan(dataframe.loc[:,feature])].index
        elements_count = collections.Counter(dataframe.loc[:,feature].dropna())
    
        weight_list = np.array(list(elements_count.values()))/sum(elements_count.values())
        fill_list=random.choices(list(elements_count.keys()), weights=weight_list, k=num_na)
        
        if (len(fill_list)!=0):
            series_tofill = pd.Series(data = fill_list)
            series_tofill.index = na_index
            
            # Substituing Na values to the dataframe
            dataframe.loc[:,feature] = dataframe.loc[:,feature].fillna(value = series_tofill)
        
    return dataframe

# =============================================================================
# From feature list - it replaces the features with the hot encoded ones if needed
# =============================================================================

def change_feat_to_hotenc(feat_names, encod_col, final_frame):
    # TO deal with hot endcoded features as their name changes when we hot encode them
    to_code = list(set(feat_names).intersection(set(encod_col)))
    feat_names1 = [i for i in feat_names if i not in to_code] # Removing the previous names
    if(len(to_code)!=0):
        to_add = []
        for col_name in to_code:
            to_add.append([col for col in final_frame.columns if col_name in col])
        ravel_list = np.concatenate(to_add).ravel()
        feat_names1  = feat_names1 + list(ravel_list) # Adding the new names it has after dummies
        
    return feat_names1

# =============================================================================
# Reverse One hot encode
# =============================================================================

def change_hotenc_to_feat(feat_names, encod_col, X):
    ####### Dealing with the Hot Encoder features
    to_code = list(set(feat_names).intersection(set(encod_col)))
    #feat_names1 = [i for i in time_col if i not in to_code] # Removing the previous names
    if(len(to_code)!=0):
        col_names=[]
        for col_name1 in to_code:
            cols_for_col = [col for col in X.columns if col_name1 in col]
            col_names.append(col_name1)  
            # Now we need to remove intersected feature and add the encoded ones
            feat_names.remove(col_name1)
            feat_names = feat_names+ cols_for_col
            
    return feat_names

###################################################################
### Change names of the say T_surg_type to one in the file like 'Surgery types'
###########################################################

def change_names(arr,col_name, df):
    # Create an empty list to store the replaced names
    replaced_names = []
    
    # Loop through each element in the array
    for name in arr:
      # Check if the name exists in the "name format" column of the dataframe
      if name in df["Columns"].values:
        # Find the row index where the name matches
        index = df[df["Columns"] == name].index[0]
        # Get the corresponding value from the "name replace" column
        new_name = df.loc[index, col_name]
        # Append the new name to the list
        replaced_names.append(new_name)
      else:
        # If the name does not exist in the dataframe, keep it as it is
        replaced_names.append(name)

    return np.array(replaced_names)




def permutation_test(X, Y, model_list, model_list1, apply_SMOTE= False, n_perms= 100):
    """Run permutation test for a model"""
    
    # Get regular CV scores
    feature_imp, feature_all, df, _, _ = ns_ML_model_test(model_list,model_list1,X,Y,5,apply_SMOTE=apply_SMOTE,feat_len=None,return_proba=True, return_coef=False, n_repeats=5)
    true_score  = np.mean(df.roc_auc[0])
    
    scores = []

    for i in tqdm(range(n_perms)):

        # Permute labels
        Y_perm = pd.DataFrame(np.random.permutation(Y))

        # Get CV scores on perm data
        _, _, df_perm, _, _ = ns_ML_model_test(model_list,model_list1,X,Y_perm[0],5,apply_SMOTE=False,feat_len=None,return_proba=True, return_coef=False, n_repeats=5)

        # Extract score
        score = np.mean(df_perm.roc_auc[0])
        scores.append(score)

    return true_score, scores

def calc_pvalue(true_score, scores):
    t_value, p_value = sp.stats.ttest_1samp(np.array(scores), true_score)
    return t_value, p_value


def permutation_test_pval(true_score, scores):

    count = 0
    for score in scores:
        if abs(score) >= abs(true_score):
            count += 1
            
    C = count
    n_permutations = len(scores)

    p = (C + 1) / (n_permutations + 1)
    
    return p
