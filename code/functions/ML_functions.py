from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score,confusion_matrix, average_precision_score,precision_score
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score, roc_curve
from itertools import chain
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score, roc_curve
from sklearn.model_selection import StratifiedKFold, LeaveOneOut


def evaluation_per_class(y, pred_y):
    correct_label_list = [0, 0]
    total_label_list = [0, 0]
    for i in range(len(y)):
        if (y[i] == pred_y[i]):
            correct_label_list[y[i]] += 1
        total_label_list[y[i]] += 1

    TP = correct_label_list[1]
    TN = correct_label_list[0]
    FP = total_label_list[0] - correct_label_list[0]
    FN = total_label_list[1] - correct_label_list[1]
    # print "TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN
    sens = float(TP) / float(TP + FN) if float(TP + FN) != 0.0 else 0.0
    spec = float(TN) / float(TN + FP) if float(TN + FP) != 0.0 else 0.0
    PPV = float(TP) / float(TP + FP) if float(TP + FP) != 0.0 else 0.0
    f1 = 2.0 * PPV * sens / (PPV + sens) if float(PPV + sens) != 0.0 else 0.0
    bl_acc = 1/2 * (float(TP)/float(total_label_list[1]) +float(TN)/float(total_label_list[0]))# balanced accurancy 
    return sens, spec, PPV, f1, bl_acc


def accuracy_matrices(y_true, y_pred,y_prob=None):
    
    acc = round(accuracy_score(y_pred, y_true)*100,2)
    acc_bal= round(balanced_accuracy_score(y_true,y_pred)*100,2)
    
    roc_auc = np.nan
    prc = np.nan
    if (y_prob!=None):
        roc_auc = round(roc_auc_score(y_true, np.array(y_prob)[:,1]),3)
        prc = round(average_precision_score(y_true, np.array(y_prob)[:,1]),3)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = round(tn*100 / (tn+fp),2)
    sensitivity = round(tp*100 / (tp+fn),2)
    precision = round(tp*100 / (tp+fp),2)
    recall = round(tp*100 / (tp+fn),2)
    
    return acc, acc_bal, roc_auc, specificity, sensitivity, precision, recall, prc





def ns_stratified_fold_LinearModel(X,Y,model, cv_fold,random_seed,apply_SMOTE=False,feat_len=None,return_proba=None,return_coef=False, Verbose=False):
    scores_train=[]
    scores_test=[]
    Y_test_list=[]
    Y_pred_test_list=[]
    Y_proba_test_list =[]
    feature_all_frame = pd.DataFrame([])
    model_coefficient=[]
    time_diff =[]
    
    # Stratified K-cv folds
    skf = StratifiedKFold(n_splits=cv_fold,shuffle=True, random_state=random_seed)
    l=-1

    for train_index, test_index in skf.split(X,Y):
        l=l+1
        # Creating train and test dataset = > It's train and cross validation data to be precise
        X_train, X_test = X.iloc[train_index,:].values, X.iloc[test_index,:].values
        Y_train, Y_test = Y.iloc[train_index].values, Y.iloc[test_index].values
        
        
        # # Debugging the indexes
        # train_ind = np.linspace(0, len(train_index)-1,num=5, dtype=int)
        # test_ind = np.linspace(0, len(test_index)-1,num=5, dtype=int)
        # print("Train indexes:", train_index[train_ind])
        # print("Test indexes:", test_index[test_ind])
        # print('\n')
        
        if (apply_SMOTE != False):
             #sm = SMOTE(sampling_strategy=apply_SMOTE, random_state=0,n_jobs=-1)
             sm= SMOTE(sampling_strategy=apply_SMOTE, random_state=0)
             X_train, Y_train = sm.fit_resample(X_train, Y_train)
             if ((Verbose) and l==0):
                 print('\n---- Applying SMOTE----')

        
        # Standarizing all features (done across all columns)
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test= sc_X.transform(X_test)
        
        
        start_time = time.time()
        # Fitting the model
        model.fit(X_train,Y_train)
        
        # If feature importance is needed then "return_coef" should be set to True
        # But it is only available for Linear Classifiers
        if (return_coef==True):
            model_str= str(model)
            # Random forest don't use .coef_ to get feature importance, instead it used .feature_importances_
            if (model_str.startswith('RandomForest')):
                model_coefficient.append(model.feature_importances_)
            else:
            # Other than RandomForest all other classifier uses .coef_ to get the feature importance
                model_coefficient.append(model.coef_)
                
        # Getting traning accuracy
        scores_train.append(accuracy_score(Y_train, model.predict(X_train)))  
        
        time_diff.append(time.time() - start_time)
        
        # Appending the Y_true and Y_pred => After completion of all cross validation folds, we can calculate accuracy instead of averaging over
        Y_test_list.append(Y_test)
        Y_pred_test_list.append(model.predict(X_test))
        
        # If the classifier allows the probability output, only then we can calculate the ROC AUC value
        if (return_proba == True):
            Y_proba_test_list.append(model.predict_proba(X_test))

    #####################################################################################
    # Here the loop across corss-validation ends and we have our appended (Y_true, Y_pred) and feature coefficient
    feature_imp = pd.DataFrame([])
    if (return_coef == True):
            ## Coeff Importance
            avg_coefficient = np.mean(model_coefficient,axis=0) #Averaging coefficients across cross-validation folds
            feature_imp= pd.DataFrame({'Feat_name':list(X.columns),'Model_coef':np.ravel(avg_coefficient)})
            feature_imp = feature_imp.sort_values(by='Model_coef', ascending=False,key=abs) # Arranging feature importance by absolute value of coefficient
            feature_imp= feature_imp.reset_index(drop=True)
    
    # For test data => we used function "accuracy_matrices" (defined above) which do it for test data by default
    # As the (Y_true, Y_pred) are list inside list => We can use "chain.from_iterable" to unravel them into a single list
    # If the classifier allows the probability output, only then we can calculate the ROC AUC value
    if (return_proba == True): 
        Y_test_list,Y_pred_test_list,Y_proba_test_list = list(chain.from_iterable(Y_test_list)),list(chain.from_iterable(Y_pred_test_list)),list(chain.from_iterable(Y_proba_test_list))
        test_data_stats= accuracy_matrices(Y_test_list, Y_pred_test_list,Y_proba_test_list)
    else:
        Y_test_list,Y_pred_test_list, = list(chain.from_iterable(Y_test_list)),list(chain.from_iterable(Y_pred_test_list))
        test_data_stats= accuracy_matrices(Y_test_list, Y_pred_test_list)    
        
    # Rounding off training accuracy for easy interpretation
    acs_train=round((np.mean(scores_train,axis=0))*100,2)
    std_train=round((np.std(scores_train,axis=0))*100,2)
    
    # Only relevant to feature selection 
    feature_all_frame=feature_all_frame.reset_index(drop=True)
    
    # If our classifier outputs probability (to calculate ROC AUC), then we need to output probability for test data not the label (0,1)
    if (return_proba == True):
        Y_pred_test_list = np.array(Y_proba_test_list)[:,1]

    return len(X_test[0]),Y_test_list,Y_pred_test_list,test_data_stats, acs_train, std_train, feature_all_frame, feature_imp, time_diff

#############################################################################




############################################################################
## This file allow us to test for multiple classifier/regression models and report there results in a dataframe
def ns_ML_model_test(model_list,model_list1,X,Y,cv_fold=10,apply_SMOTE=False,feat_len=None,return_proba=None,return_coef=False,n_repeats=1, Verbose=False):
    all_sites_target_frame= pd.DataFrame([])
    for model_no in tqdm(range(len(model_list))):  #'tqdm' helps keep track of processing time
        model= model_list[model_no] # Get the model to train on
        
        random_seed = np.linspace(0,100,n_repeats, dtype=int)
        repeat_frame  = pd.DataFrame([])
        for i, repeat in enumerate(range(n_repeats)):
            
            
            # Get output from out Cross validation folds
            ml_output = ns_stratified_fold_LinearModel(X,Y,model, cv_fold, random_seed= random_seed[i],apply_SMOTE=apply_SMOTE, feat_len=feat_len,return_proba=return_proba,return_coef=return_coef, Verbose=Verbose)
            # Unreavel the outputs from the above
            feat_len_out,Y_test_list,Y_pred_test_list,test_data_stats, acs_train, std_train, feature_all_frame, feature_imp, time_diff= ml_output
            # Further unraveling the "test_data_stats" to its corresponding outputs
            acc, acc_bal, roc_auc, specificity, sensitivity, precision, recall, prc = test_data_stats
            
            fpr, tpr = 0, 0
            if (return_proba == True):
                fpr, tpr, _ = roc_curve(Y_test_list, Y_pred_test_list) 

            temp_repeat=pd.DataFrame({'model':[model_list1[model_no]],
                               'acc':[acc],
                               'acc_bal': [acc_bal],
                               'roc_auc':[roc_auc],
                               'specificity': [specificity],
                               'sensitivity':[sensitivity],
                               'precision': [precision],
                               'recall': [recall],
                               'prc':prc,
                               'simple_train':[acs_train],
                               'feat_sel_len':[feat_len_out],
                               'time':np.mean(time_diff),
                               'fpr': [fpr],
                               'tpr': [tpr]})  
            
            repeat_frame = pd.concat([repeat_frame,temp_repeat],axis=0, join='outer') 
            
        repeat_frame = repeat_frame.reset_index(drop=True)
        
        
        temp=pd.DataFrame({'model': [model_list1[model_no]],
                           'acc': [repeat_frame.loc[:,'acc'].values],
                           'acc_bal': [repeat_frame.loc[:,'acc_bal'].values],
                           'roc_auc': [repeat_frame.loc[:,'roc_auc'].values],
                           'specificity': [repeat_frame.loc[:,'specificity'].values],
                           'sensitivity': [repeat_frame.loc[:,'sensitivity'].values],
                           'precision': [repeat_frame.loc[:,'precision'].values],
                           'recall': [repeat_frame.loc[:,'recall'].values],
                           'prc': [repeat_frame.loc[:,'prc'].values],
                           'simple_train': [repeat_frame.loc[:,'simple_train'].values],
                           'feat_sel_len': [feat_len_out],
                           'time':[repeat_frame.loc[:,'time'].values],
                           'fpr': [repeat_frame.loc[:,'fpr'].values],
                           'tpr': [repeat_frame.loc[:,'tpr'].values]})

        all_sites_target_frame= pd.concat([all_sites_target_frame,temp],axis=0, join='outer') 
        ##########################################################
    
        #print('\n','Acc_test_bal: ',simple_test_acc[1] ,'Acc_train: ',simple_train_acc[0] )
    # Creating final output dataframe from all input models
    all_sites_target_frame= all_sites_target_frame.reset_index(drop=True)
    
    return feature_imp, feature_all_frame,all_sites_target_frame,Y_test_list,Y_pred_test_list
