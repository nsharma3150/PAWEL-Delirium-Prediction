# Here importing all the necssary libraries and functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy as sp
from utils import change_names
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from adjustText import adjust_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

plt.rcParams["font.family"] = "Arial" 



def confidence_plot(dataframe, model_list1,image_str,var_list, image_folder):
        
    plot_frame = pd.DataFrame()
    for i, model_name in enumerate(model_list1):
        model_frame = dataframe[dataframe.model == model_name]
        for var_name in var_list:
            var_values = model_frame.loc[:,var_name].values[0]
            temp_frame = pd.DataFrame({'Model': np.full(shape=len(var_values),fill_value=model_name), 'Metric_val': var_values,
                                       'Metric': np.full(shape=len(var_values),fill_value=var_name)})
            
            plot_frame = pd.concat([plot_frame,temp_frame],axis=0, join='outer') 
        
    plot_frame = plot_frame.reset_index(drop=True)
    
    fig, axs = plt.subplots(figsize=(18,12))
    sns.violinplot(data=plot_frame, x="Model", y="Metric_val", hue="Metric", ax=axs)
    
    fig.set_size_inches((18,12), forward=False)
    plt.savefig(os.path.join(image_folder,'Confidence_interval'+image_str+'.png'))


############################################
# Plotting the permutation test 

def permutation_histogramplot(scores, true_score, img_str, image_folder):
        
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    
    scores = np.array(scores, dtype=float)
    mu = np.mean(scores)  
    sigma = np.std(scores)
    
    plt.figure(figsize=(12,6))
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.hist(scores, bins=20, density=True, color='lightblue', alpha=0.5)
    plt.plot(x, sp.stats.norm.pdf(x, mu, sigma), color='black', label='Gaussian approximation')
    
    plt.axvline(x=true_score, color='red', linestyle='--', label='True score')
    
    lower_bound = sp.stats.norm.ppf(0.025, mu, sigma)
    upper_bound = sp.stats.norm.ppf(0.975, mu, sigma)
    
    plt.axvline(x=lower_bound, color='green', linestyle='--', label='Lower bound')
    plt.axvline(x=upper_bound, color='green', linestyle='--', label='Upper bound')
    
    plt.xlabel('Scores')
    plt.ylabel('Density')
    plt.title(img_str)  # 'Histogram of Scores with Gaussian Approximation'
    plt.legend(bbox_to_anchor=[1,0.5])
    plt.tight_layout()
    
    plt.savefig(os.path.join(image_folder, 'Histogram ' + img_str + '.png'), dpi=300)



    # =============================================================================
# ROC AUC in curve form for all models
# =============================================================================
def plot_roc_curves(ML_result_frame1, model_list1, img_str, image_folder):

  # Color map for models
  color_map = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD']  #['r','b','g','orange']
  
  fig, ax = plt.subplots(figsize=(12,8))

  for i, model_name in enumerate(model_list1):

    fpr = ML_result_frame1[ML_result_frame1['model'] == model_name]['fpr'].values[0]
    tpr = ML_result_frame1[ML_result_frame1['model'] == model_name]['tpr'].values[0]
    
    ### 
    interp_fpr = np.linspace(0, 1, 500) 
    fpr1=np.empty((fpr.shape[0],interp_fpr.shape[0]))
    tpr1=np.empty((tpr.shape[0],interp_fpr.shape[0]))
    for j in range(fpr.shape[0]):
        interp_tpr = np.interp(interp_fpr, fpr[i], tpr[i])

        fpr1[j,:] = interp_fpr
        tpr1[j,:] = interp_tpr
        

    mean_fpr = np.mean(fpr1, axis=0)
    mean_tpr = np.mean(tpr1, axis=0)
    
    std_fpr = np.std(fpr1, axis=0)
    std_tpr = np.std(tpr1, axis=0)

    # Plot shaded CI region
    ax.fill_between(mean_fpr, mean_tpr - 1.96*std_tpr/np.sqrt(len(tpr)),  
                    mean_tpr + 1.96*std_tpr/np.sqrt(len(tpr)), alpha=0.1, 
                    color=color_map[i])

    # Plot mean ROC curve   
    ax.plot(mean_fpr, mean_tpr, linewidth=1.5, color=color_map[i],
            label=model_name, alpha = 1)

  ax.plot([0,1], [0,1], 'k--')
  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.05])

  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  ax.set_title(img_str + ' ROC-AUC curve plot for all models')

  ax.legend(loc='lower right', borderaxespad=0,title='Models') 
  
  fig.savefig(os.path.join(image_folder, img_str + ' ROC-AUC curve plot for all models.png'), dpi=300)
  
  return fig, ax



# =============================================================================
# ROC AUC in curve form for all different cases but not for all models but just for two cases of one model
# =============================================================================
def plot_roc_curves_one_model(ML_result_frame1, img_str,image_folder ,model_name='Random Forest',color_map = ['#1F77B4', '#FF7F0E'], legend_list=['Features','Without Cognitive', 'With Cognitive']):

  # Color map for models
    #['r','b','g','orange']
  
  fig, ax = plt.subplots(figsize=(12,8))

  for i, oversample in enumerate([legend_list[1], legend_list[2]]):
      
    prefix = ''
    if (oversample == legend_list[2]):
        prefix = 'S_'

    fpr = ML_result_frame1[ML_result_frame1['model'] == model_name][prefix+'fpr'].values[0]
    tpr = ML_result_frame1[ML_result_frame1['model'] == model_name][prefix+'tpr'].values[0]
    
    mean_fpr = np.array(fpr)[0]
    mean_tpr = np.array(tpr)[0]
    if (fpr.shape[0] > 1):
        ### 
        interp_fpr = np.linspace(0, 1, 500) 
        fpr1=np.empty((fpr.shape[0],interp_fpr.shape[0]))
        tpr1=np.empty((tpr.shape[0],interp_fpr.shape[0]))
        for j in range(fpr.shape[0]):
            interp_tpr = np.interp(interp_fpr, fpr[i], tpr[i])
    
            fpr1[j,:] = interp_fpr
            tpr1[j,:] = interp_tpr
        
        mean_fpr = np.mean(fpr1, axis=0)
        mean_tpr = np.mean(tpr1, axis=0)
        
        std_fpr = np.std(fpr1, axis=0)
        std_tpr = np.std(tpr1, axis=0)
        

        # Plot shaded CI region
        ax.fill_between(mean_fpr, mean_tpr - 1.96*std_tpr/np.sqrt(len(tpr)),  
                        mean_tpr + 1.96*std_tpr/np.sqrt(len(tpr)), alpha=0.1, 
                        color=color_map[i])

    # Plot mean ROC curve   
    ax.plot(mean_fpr, mean_tpr, linewidth=1.5, color=color_map[i],
            label=oversample, alpha = 1)

  ax.plot([0,1], [0,1], 'k--')
  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.05])

  ax.set_xlabel('False Positive Rate')
  ax.set_ylabel('True Positive Rate')
  ax.set_title(img_str + ' ROC-AUC Curve') 

  ax.legend(loc='lower right', borderaxespad=0,title=legend_list[0]) 
  
  fig.savefig(os.path.join(image_folder, img_str + ' ROC-AUC Curve' + '.png'), dpi=300)
  
  return fig, ax




# =============================================================================
# Plotting bar plot for each model
# =============================================================================
def bar_plot(dataframe, model_list1,y_lim, y_label,image_folder, title='Title', 
             image_str='img_str', legend_list = ['Oversampling', 'None', 'SMOTE'],  
             var_list=['acc_bal','S_acc_bal'], axs=None, set_y_line=None, 
             x_label=None, text_wrap=30, use_hue=True, p_vals = None):
    
    plot_frame = pd.DataFrame()
    for i, model_name in enumerate(model_list1):
        model_frame = dataframe[dataframe.model == model_name]
        for var_name in var_list:
            var_values = model_frame.loc[:,var_name].values[0]
            temp_frame = pd.DataFrame({'Model': np.full(shape=len(var_values),fill_value=model_name), 'Metric_val': var_values,
                                       'Metric': np.full(shape=len(var_values),fill_value=var_name)})
            
            plot_frame = pd.concat([plot_frame,temp_frame],axis=0, join='outer') 
        
    plot_frame = plot_frame.reset_index(drop=True)
    
    
    # Changing legend name
    plot_frame.columns = ['Model', y_label, legend_list[0]] 

    if use_hue:
        plot_frame[legend_list[0]] = plot_frame[legend_list[0]].map({var_list[0]: legend_list[1], 
                                                                    var_list[1]: legend_list[2]})
    
    # check if ax is None or not
    if axs is None: # create a new figure and axes if ax is not provided
        fig, ax = plt.subplots(figsize=(10,7), gridspec_kw={'width_ratios': [1/4]})
    else:
        ax = axs
    # else use the provided axes object to plot
    
    if (set_y_line !=None):
        ax.axhline(y=set_y_line, linestyle='--', linewidth=2, color='black',zorder=1)

    
    # Rest of plot code
    if use_hue:
        bar = sns.barplot(data=plot_frame, x='Model', y=y_label, hue=legend_list[0], ax=ax, 
                palette=['steelblue', 'orange'], width=0.5, capsize=0.05)
    else:
        bar = sns.barplot(data=plot_frame, x='Model', y=y_label, ax=ax, 
                palette=['steelblue'], width=0.5, capsize=0.05)

    
    middle_point=[]
    for b in bar.patches:
        middle_point.append(b.get_x() + b.get_width() / 2)

    # set y-axis limits and labels
    ax.set_ylim(0, y_lim)
    
    if axs is None:
        # add title to plot
        ax.set_title(title, fontsize=16)
    
    # # TO put p values on the bar plot and the significance star symbol along with horizontal brackets
    # if p_vals is not None:
    #     # Map p-values to significance levels
    #     p_map = {p: '' if p >= 0.05 else '*' if 0.01 <= p < 0.05 else '**' for p in p_vals}
        

    #     # Brackets between each pair of hues
    #     for i,p in enumerate(p_vals):
          
    #       y = max(plot_frame.loc[2*i, y_label], plot_frame.loc[2*i+1, y_label]) + 0.03
          
    #       # Plotting text
    #       ax.text(i,  y+ 0.008, 'p={:.4f}'.format(p), ha='center')
    #       ax.text(i,  y+ 0.02, p_map[p], ha='center')
          
    #       # Plotting bracket
    #       ax.hlines(y=y, xmin=middle_point[i], xmax=middle_point[len(middle_point)//2+i], color='k', linestyle='-', lw=1) # horizontal line
    #       ax.vlines(x=[middle_point[i], middle_point[len(middle_point)//2+i]], ymin=y, ymax= y-0.02, color='k', linestyle='-', lw=1) # vertical line
          
  
    if axs is None: # only call fig.tight_layout() if fig is created
        fig.tight_layout()
        
    ax.tick_params(axis='x', labelsize=9)

    
    # add x-label and y-label superlabels with labelpad and fontdict arguments
    ax.set_xlabel(plot_frame.columns[0], labelpad=5, fontdict={'size': 12}) #'weight': 'bold'
    ax.set_ylabel(plot_frame.columns[1], labelpad=5, fontdict={'size': 12}) # 'Metric Value (95% CI)'

    if (x_label !=None):
        ax.set_xlabel(x_label, labelpad=5, fontdict={'size': 12}) #'weight': 'bold'
    
     # Modifications to legend and other plot aesthetics
    if use_hue:
        ax.legend(loc='upper right', borderaxespad=0, title=plot_frame.columns[2])

    # Rotate x labels
    ## For long labels
    import textwrap
    labels = [textwrap.fill(label.get_text(), text_wrap) for label in ax.get_xticklabels()]


    ax.set_xticklabels(labels, rotation=0)
    
    # add major grid lines with light gray color and dashed linestyle
    ax.grid(axis='y', color='lightgray', linestyle='--')
    # make the grid lines appear behind the bars
    ax.set_axisbelow(True)
    
    fig.tight_layout()

    # save the figure using fig.savefig()
    if axs is None: # only save the figure if fig is created
        fig.savefig(os.path.join(image_folder,image_str+'.png'), dpi=300)
    


# =============================================================================
# SHAP Plot
# =============================================================================


def shap_plot(X,Y,Cognitive,image_folder, feature_frame, clf_ind=2, title='Title', image_str='img_str',axs=None, scatter_plot=False, top_feat=15, use_hue=True, return_val=False):
    
    feature_names= X.columns
    
    shap_avg =[]
    for apply_SMOTE, oversampling in zip([False, True],['None','SMOTE']):
        # Stratified K-cv folds
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
        for train_index, test_index in skf.split(X,Y):
            # Creating train and test dataset = > It's train and cross validation data to be precise
            X_train, X_test = X.iloc[train_index,:].values, X.iloc[test_index,:].values
            Y_train, Y_test = Y.iloc[train_index].values, Y.iloc[test_index].values
            
            if (apply_SMOTE != False):
                #sm = SMOTE(sampling_strategy=apply_SMOTE, random_state=0,n_jobs=-1)
                sm= SMOTE(sampling_strategy=apply_SMOTE, random_state=0)
                X_train, Y_train = sm.fit_resample(X_train, Y_train)
    
            # Standarizing all features (done across all columns)
            sc_X = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            X_test= sc_X.transform(X_test)
        
        if clf_ind == 0:
            clf = XGBClassifier(objective='binary:logistic', booster='gbtree')
            col_name = 'Gradient boosting'
        if clf_ind == 1:
            clf = LogisticRegression(C=1,max_iter=1000000,n_jobs=-1)
            col_name = 'Logistic RegressionR'
        if clf_ind == 2:
            clf = RandomForestClassifier(n_jobs=-1, random_state=0)
            col_name = 'Random Forest'
        if clf_ind == 3:
            clf = SVC(C=1,kernel='linear',probability=True)
            col_name = 'Linear SVC'
            
        # train ML models
        clf.fit(X_train, Y_train)
        
        #Calculate SHAP values
        if (clf_ind==0): # For gradient boosting tree
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test)
            shap_values_pl = explainer(X_test)
            
        #Calculate SHAP values
        elif (clf_ind==2): # For random forest 
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test)
            shap_values_pl = explainer(X_test)
            
            shap_values = shap_values[1][:][:]
            shap_values_pl = shap_values_pl[:,:,1]
            
        elif (clf_ind==1 or clf_ind==3):# For logistic regression and linear SVM
            background = shap.maskers.Independent(X_train, max_samples=1000)
            explainer = shap.Explainer(clf.predict, background, seed=0)
            shap_values_pl = explainer(X_test)
            shap_values = shap_values_pl.values    
         
        rf_resultX = pd.DataFrame (shap_values, columns = feature_names)
        vals = np.abs (rf_resultX.values).mean (0)
        
        shap_avg.append(vals)
        
    temp_frame = pd.DataFrame({'Features':feature_names,'None': shap_avg[0], 'SMOTE':shap_avg[1] })
        
    temp_frame = temp_frame.sort_values (by= ['None'], ascending=False)
    temp_frame= temp_frame.reset_index(drop=True)
    

    if ((scatter_plot==True) and (clf_ind==2)):
        
        shap_importance1 = pd.DataFrame({'Features':list(temp_frame.Features) + list(temp_frame.Features), 
                                   'Mean(|SHAP value|)':list(temp_frame.loc[:,'None'].values)+list(temp_frame.loc[:,'SMOTE'].values),
                                   'Oversampling':list(np.full(shape=len(temp_frame),fill_value='None'))+list(np.full(shape=len(temp_frame),fill_value='SMOTE'))})
        
        # =============================================================================
        # Test time plot using SHAP values
        # =============================================================================
        test_time = r'..\data'
        test_time_frame = pd.read_csv(os.path.join(test_time,'test_time.csv'))
    
        
        shap_importance2 = shap_importance1[shap_importance1.Oversampling=='None']
        shap_importance2 = shap_importance2.reset_index(drop=True)
        
        
        ## Selecting only cognitive feature
        
        # If want in general
        #col_intersection = list(set(test_time_frame.Features).intersection(set(shap_importance2.Features)))
        
        #### If want only cognitive features
        #col_intersection = list(set(test_time_frame.Features).intersection(set(list(cognitive))))
        col_intersection = list(set(shap_importance2.Features).intersection(set(list(Cognitive))))
        
        shap_val  =  [shap_importance2[shap_importance2.Features == i].loc[:,'Mean(|SHAP value|)'].values[0] for i in col_intersection]
        
        
    
        # test_time_frame.insert(loc=3,column='SHAP_val', value="")
        # test_time_frame = 
        
        #### If in general
        #time_col = test_time_frame.Features
        
        # If Cognitive Only
        time_col = list(Cognitive)
        
        # Hot encoded features:
        encod_col = ['location','T1_SMI', 'T2_narc_kind', 'T2_OP_kind_general']
        
        # TO deal with hot endcoded features as their name changes when we hot encode them
        to_code = list(set(time_col).intersection(set(encod_col)))
        #feat_names1 = [i for i in time_col if i not in to_code] # Removing the previous names
        if(len(to_code)!=0):
            col_names=[]
            shap_vals = []
            for col_name1 in to_code:
                cols_for_col = [col for col in X.columns if col_name1 in col]
                col_names.append(col_name1)
                a = np.sum([shap_importance2[shap_importance2.Features == i].loc[:,'Mean(|SHAP value|)'].values[0] for i in cols_for_col])
                shap_vals.append(a)
            
            col_intersection = col_intersection + col_names
            shap_val = shap_val + shap_vals
    
        ##########
        
        time_new  =  [test_time_frame[test_time_frame.Features == i].loc[:,'Time'].values[0] for i in col_intersection]
        
        shap_val_frame = pd.DataFrame({'Time':time_new,'Features':col_intersection, 'Mean(|SHAP value|)':shap_val})
        
        shap_val_frame = shap_val_frame.sort_values(by='Mean(|SHAP value|)',ascending=False,ignore_index=True)
        
        # shap_val_frame.to_csv(os.path.join(save_files,'cognitive_feat_shap_value.csv'), index=False)
        
        # Plotting it
        n=len(shap_val_frame.loc[:,'Mean(|SHAP value|)'])
        # Use a style sheet
        # Create a subplot
        fig1, axis1 = plt.subplots(figsize=(14,12))
        
        # Scatter plot with different colors
        axis1.scatter(shap_val_frame.Time, shap_val_frame.loc[:,'Mean(|SHAP value|)'], alpha=0.5)
        
        colors = plt.cm.tab20(np.linspace(0, 1, n)) # use a colormap
        axis1.scatter(shap_val_frame.loc[:(n-1),'Time'], shap_val_frame.loc[:(n-1),'Mean(|SHAP value|)'], c=colors)
        
        # Set x-axis ticks and limits
        axis1.set_xticks([0.1,1,2,3,4,5,6,13])
        axis1.set_xlim([-0.5,14])
        
        # Set x-axis and y-axis labels
        axis1.set_xlabel("Time (sec)")
        axis1.set_ylabel("Mean(|SHAP value|)")
        
        # # Add text labels and adjust them
        # val = 10
        # texts = []
        # for i in range(val):
        #     if (i<20):
        #         texts.append(axis1.text(shap_val_frame.loc[i,'Time']+0.1, shap_val_frame.loc[i,'Mean(|SHAP value|)'], 
        #                         str(shap_val_frame.loc[i,'Features']), dict( va='center',ha='left',fontsize=10)))
        #     else:
        #         texts.append(axis1.text(shap_val_frame.loc[i,'Time'], shap_val_frame.loc[i,'Mean(|SHAP value|)']+0.05*shap_val_frame.loc[i,'Mean(|SHAP value|)'],
        #                                 str(shap_val_frame.loc[i,'Features']), dict( va='top',ha='left',fontsize=10)))
                
        shap_val_frame.loc[:,'Features'] = change_names(arr=shap_val_frame.loc[:,'Features'].values, col_name='Name_scatter1', df=feature_frame)

        texts = []
        for i in range(n):
             texts.append(axis1.text(shap_val_frame.loc[i,'Time']+0.1, shap_val_frame.loc[i,'Mean(|SHAP value|)'], 
                            str(shap_val_frame.loc[i,'Features']), dict( va='center',ha='left',fontsize=10)))

        adjust_text(texts, avoid_self=False, force_text=(0.05, 0.05), force_static = (0.05, 0.05),
                    time_lim=200) # this will adjust the labels automatically
        
        # Add a title and a legend
        axis1.set_title("Scatter plot of mean SHAP values by features")
        #axis1.legend(['Feature ' + str(i) for i in range(1, 11)])
        
        axis1.grid(True)
        
        # Save the figure
        fig1.savefig(os.path.join(image_folder, col_name+ '-None' +'- Scatter plot time'+'.png'), dpi=300)
        
        # Show the plot
        #plt.show()



    ############ Plotting a Vertical Bar plot
    
    temp_frame1 = temp_frame.copy()
    
    # =============================================================================
    #Summing up the features with same name but One hot encoded
    # Hot encoded features:
    to_code = ['location','T1_SMI', 'T2_narc_kind', 'T2_OP_kind_general']

    #feat_names1 = [i for i in time_col if i not in to_code] # Removing the previous names
    col_intersection = []
    shap_val_NONE = []
    shap_val_SMOTE = []
    to_delete = []
    if(len(to_code)!=0):
        col_names=[]
        for col_name1 in to_code:
            cols_for_col = [col for col in X.columns if col_name1 in col]
            
            if (len(cols_for_col)==0):
                continue;
            col_names.append(col_name1)
            SOMTE_val = np.sum([temp_frame1[temp_frame1.Features == i].loc[:,'SMOTE'].values[0] for i in cols_for_col])
            None_val = np.sum([temp_frame1[temp_frame1.Features == i].loc[:,'None'].values[0] for i in cols_for_col])
            
            shap_val_SMOTE.append(SOMTE_val)
            shap_val_NONE.append(None_val)
            
            to_delete.append(cols_for_col)
        
        col_intersection = col_intersection + col_names
    
    #To delete all the columns not relevant
    to_delete = [item for sublist in to_delete for item in sublist]

    mask = temp_frame1['Features'].isin(to_delete)
    # Invert the mask to keep only the rows that are not in the to_delete array
    mask = ~mask
    # Filter the dataframe using the mask
    temp_frame1 = temp_frame1[mask] 
    temp_frame1 = temp_frame1.reset_index(drop=True)
    
    new_df = pd.DataFrame({'Features': col_intersection, 'None': shap_val_NONE, 'SMOTE':shap_val_SMOTE})
    # Concatenate the two dataframes along the columns axis
    temp_frame1 = pd.concat([temp_frame1, new_df], axis=0)    
    
    temp_frame1 = temp_frame1.sort_values(by='None',ascending=False)
    temp_frame1 = temp_frame1.reset_index(drop=True)
    
    shap_val_frame11 =temp_frame1.copy()
    # =============================================================================
    
    temp_frame1 = temp_frame1[0:top_feat]
    
    
    if use_hue:
        shap_importance = pd.DataFrame({'Features':list(temp_frame1.Features) + list(temp_frame1.Features), 
                                   'Mean(|SHAP value|)':list(temp_frame1.loc[:,'None'].values)+list(temp_frame1.loc[:,'SMOTE'].values),
                                   'Oversampling':list(np.full(shape=len(temp_frame1),fill_value='None'))+list(np.full(shape=len(temp_frame1),fill_value='SMOTE'))})
        
    else:
        shap_importance = pd.DataFrame({'Features':list(temp_frame1.Features), 
                                   'Mean(|SHAP value|)':list(temp_frame1.loc[:,'None'].values),
                                   'Oversampling':list(np.full(shape=len(temp_frame1),fill_value='None'))})
 
    
    shap_importance.loc[:,'Features'] = change_names(arr=shap_importance.loc[:,'Features'].values, col_name='Name_bar1',df=feature_frame)
    
    # check if ax is None or not
    if axs is None: # create a new figure and axes if ax is not provided
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        ax =axs
    
    # use palette argument to specify two contrasting colors for each hue level
    # Other color options: ['steelblue', 'orange'], ['blue', 'orange'], ['lightgreen', 'lavender'], ['lightblue', 'lightcoral']
    # ['lightpink', 'lightgray'], ['steelblue', 'lightcyan']
    
    if use_hue:
        sns.barplot(data=shap_importance, y=shap_importance.columns[0], x=shap_importance.columns[1], hue=shap_importance.columns[2], ax=ax, palette=['steelblue', 'orange'], width=0.5,
                    capsize=0.05, orient='h')
    else:
        sns.barplot(data=shap_importance, y=shap_importance.columns[0], x=shap_importance.columns[1], ax=ax, palette=['steelblue'], width=0.5,
                    capsize=0.05, orient='h')
    
    # add title to plot
    if axs is None:
        ax.set_title(title, fontsize=16)
    
    if axs is None: # only call fig.tight_layout() if fig is created
        fig.tight_layout()
        
    # Y ticks size
    ax.tick_params(axis='y', labelsize=9)
    
    # add x-label and y-label superlabels with labelpad and fontdict arguments
    ax.set_ylabel(shap_importance.columns[0], labelpad=5, fontdict={'size': 12}) #'weight': 'bold'
    ax.set_xlabel(shap_importance.columns[1], labelpad=5, fontdict={'size': 12}) # 'Metric Value (95% CI)'
    
    if use_hue:
        # Legend
        ax.legend(loc='lower right', borderaxespad=0,title=shap_importance.columns[2]) #, bbox_to_anchor=(0.95, 0.05)

    # rotate the y-labels by 30 degrees
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    # add major grid lines with light gray color and dashed linestyle
    ax.grid(axis='x', color='lightgray', linestyle='--')
    # make the grid lines appear behind the bars
    ax.set_axisbelow(True)

    # save the figure using fig.savefig()
    if axs is None: # only save the figure if fig is created
        fig.savefig(os.path.join(image_folder, col_name+' SHAP abs value' +'.png'), dpi=300)
    
    if return_val:
        return shap_val_frame11



# Plot for two classifiers

def two_classifier_plot(dataframe,variable_str,var_dict,classifier_list,image_str,image_folder):
        
    column = 'acc_bal'
    column1 = 'S_acc_bal'
    
    new_dataframe=pd.DataFrame([])
    for model_name in classifier_list:
        new_dataframe= pd.concat([new_dataframe,dataframe[dataframe.loc[:,'model']==model_name]],axis=0, join='outer') 
        
        
    dataframe = new_dataframe
    grouped = dataframe.groupby(variable_str)
    
    names, vals,vals1,xs,xs1 = [], [] ,[],[],[]
    
    for i, (name, subdf) in enumerate(grouped):
        names.append(name)
        vals.append(subdf[column].tolist())
        vals1.append(subdf[column1].tolist())
    
    
    # This makes the plot order on x-axis consistent with the other plots
    vals11,vals22 = [],[]
    for i,order_name in enumerate(list(var_dict.values())):
        names_series = pd.Series(names)
        name_ind = names_series[names_series==order_name].index[0]
    
        vals11.append(vals[name_ind])
        vals22.append(vals1[name_ind])
        
        
    vals, vals1, names = vals11, vals22, list(var_dict.values())
    
    
    ####
    left_plot_points=np.linspace(0, 20, len(names))
    right_plot_points=np.linspace(0, 20, len(names))+0.5
    
    #####
    for i in left_plot_points:
        xs.append(np.random.normal(i, 0.00001, subdf.shape[0]))
        
    for i in right_plot_points:
        xs1.append(np.random.normal(i, 0.00001, subdf.shape[0]))
    ###
            
    fig, ax = plt.subplots()
    
    plt.violinplot(vals,positions=left_plot_points,showmeans=False, showmedians=False,showextrema=False)
    plt.violinplot(vals1,positions=right_plot_points, showmeans=False, showmedians=False,showextrema=False)

    
    plt.xticks((2*np.linspace(0, 20, len(names))+0.5)/2, names)
    ######################################
    
    ######################################
    vals_dash= np.transpose(vals)
    vals1_dash= np.transpose(vals1)
    
    xs_dash =  np.transpose(xs)
    xs1_dash =  np.transpose(xs1)
    
    # Plotting two classifier's balanced accuracy for SMORE and no SMOTE case
    colur_list=['limegreen','mediumblue']
    i=-1
    for x, val, colour in zip(xs_dash, vals_dash, colur_list):
        i=i+1;
        plt.scatter(x, val, c=colour, alpha=0.4, label =classifier_list[i],marker='o',s=60)
        
    for x, val, colour in zip(xs1_dash, vals1_dash, colur_list):
        plt.scatter(x, val, c=colour, alpha=0.4,marker='D',s=60)
    
    plt.grid(visible=True, axis='y',linestyle='--')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='upper right')
    plt.title('Left: Balanced Accuracy   ||     Right:  SMOTE Balanced Accuracy',fontsize='15',
              backgroundcolor='grey',
              color='white',fontweight="bold")  
    
    # Saving the image
    plt.savefig(os.path.join(image_folder))
    fig.set_size_inches((20, 9), forward=False)
    plt.savefig(os.path.join(image_folder,'Top_two_classifier_'+image_str+'.png'))


### Bar plot for AVG RESULT COMPARISON -- BAR GRAPH
###########################################################

def avg_bar_plot(dataframe,image_folder,col_list =['S_acc_bal','acc_bal'],x_y_metrics='Accuracy', image_str='Accuracy Avg'):
    
    labels = dataframe.loc[:,'model']
    
    avg_adapt = dataframe.loc[:,col_list[0]]
    avg_simple = dataframe.loc[:,col_list[1]]
    
    # Getting max and min=> For better plot
    all_acc = list(avg_adapt) + list(avg_simple)
    max_plot = max(all_acc)+ 5
    min_plot = min(all_acc)- 5
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25)
    
    ax.barh(x - width/2, avg_adapt, width, label='SMOTE 1.0',color='dodgerblue')
    ax.barh(x + width/2, avg_simple, width, label='Without SMOTE',color='lime')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('{} Scores'.format(x_y_metrics))
    ax.set_title('{} comparison with SMOTE oversampling'.format(x_y_metrics))
    ax.set_yticks(x)
    
    ax.set_xlim(left=min_plot, right=max_plot)
    
    ax.set_yticklabels(labels)
    ax.legend()
    
    plt.vlines(x=np.arange((min_plot//5)*5,(max_plot//5)*5,5), ymin=0, ymax=len(labels), colors='orangered', ls=':', lw=2)
    plt.show()
    
    fig.set_size_inches((20, 10), forward=False)
    plt.savefig(os.path.join(image_folder,'Result_'+image_str+'.png'))


# =============================================================================
# Each classifier blot as bar plot
# =============================================================================
def each_classifier_plot(dataframe,variable_str,var_dict,image_str,image_folder, model_list1):
        
    grouped = dataframe.groupby(variable_str)
    
    names, vals,vals1,xs,xs1 = [], [] ,[],[],[]
    
    for i, (name, subdf) in enumerate(grouped):
        names.append(name)
        vals.append(subdf['acc_bal'].tolist())
        vals1.append(subdf['S_acc_bal'].tolist())
    

    
    # This makes the plot order on x-axis consistent with the other plots
    vals11,vals22 = [],[]
    for i,order_name in enumerate(list(var_dict.values())):
        names_series = pd.Series(names)
        name_ind = names_series[names_series==order_name].index[0]
    
        vals11.append(vals[name_ind])
        vals22.append(vals1[name_ind])
        
        
    vals, vals1, names = vals11, vals22, list(var_dict.values())
    
    
    left_plot_points=np.linspace(0, 20, len(names))
    right_plot_points=np.linspace(0, 20, len(names))+0.5
        
    #####
    for i in left_plot_points:
        xs.append(np.random.normal(i, 0.04, subdf.shape[0]))
    
    for i in right_plot_points:
        xs1.append(np.random.normal(i, 0.04, subdf.shape[0]))
        
    #plt.figure('All classifier accuracy plot')
    fig, ax = plt.subplots()
    plt.violinplot(vals,positions=left_plot_points)
    
    plt.violinplot(vals,positions=left_plot_points,showmeans=False, showmedians=False,showextrema=False)
    plt.violinplot(vals1,positions=right_plot_points, showmeans=False, showmedians=False,showextrema=False)

    
    plt.xticks((2*np.linspace(0, 20, len(names))+0.5)/2, names)
    ######################################
    clevels = np.linspace(0., 1., len(vals[0]))
    ######################################
    vals_dash= np.transpose(vals)
    vals1_dash= np.transpose(vals1)
    
    xs_dash =  np.transpose(xs)
    xs1_dash =  np.transpose(xs1)
    
    i=-1
    for x, val, clevel in zip(xs_dash, vals_dash, clevels):
        i=i+1
        plt.scatter(x, val, c=np.tile(cm.prism(clevel),(len(val),1)), alpha=0.4,label =model_list1[i],marker='o',s=60)
        
    for x, val, clevel in zip(xs1_dash, vals1_dash, clevels):
        plt.scatter(x, val, c=np.tile(cm.prism(clevel),(len(val),1)), alpha=0.4,marker='D',s=60)

        
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([0.04, box.y0, box.width * 0.92, box.height])
        
    plt.ylabel('Classifier Accuracy (each color represents different classifier)')
    plt.grid(visible=True, axis='y',linestyle='--')
    plt.title('Variation for Balanced accuracy for all classifiers',fontsize='15',
              backgroundcolor='grey',
              color='white',fontweight="bold")
    
    plt.ylabel('Accuracy Score')
    plt.legend(loc='upper right')
    plt.title('Left: Balanced Accuracy   ||     Right:  SMOTE Balanced Accuracy',fontsize='15',
              backgroundcolor='grey',
              color='white',fontweight="bold")    
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
    plt.show()
    
    fig.set_size_inches((20, 9), forward=False)
    plt.savefig(os.path.join(image_folder,'All_classifier_'+image_str+'.png'))

def multi_bar_plot(color_list,image_folder, names, modi_frame, color_map, img_str, alpha_val=0.8): 
    width = 0.1 # Define the width of each bar
    space = 0.5 # Define the space between each group of bars
    n_cases =  len(names)
    n_hues = len(names)


    plt.figure(figsize=(12,6))
    # Loop over the cases and hues and plot each bar with a different color
    for i in range(n_cases):
        # Get the color of the bar based on the hue
        color = color_list[i*n_cases:(i+1)*n_cases]    
        for j in range(n_hues):
            # Get the value of the data point
            value = modi_frame.loc[i * n_hues + j, 'roc_auc']
            
            # Get the position of the bar based on the case and hue
            # Add some offset to create space between each group of bars
            position = i + j * width + i * space
            
            # Plot the bar with plt.bar
            plt.bar(position, value, width=width, color=color[j], alpha=alpha_val)
    
    plt.xlabel('Cases')
    plt.ylabel('ROC-AUC value')
    plt.title(img_str + ' ROC values')
    plt.xticks([])
    
    # Create a list of handles and labels for the legend
    handles = [plt.bar([], [], color=color_map[j]) for j in range(len(names))]
    
    # add major grid lines with light gray color and dashed linestyle
    #ax.grid(axis='y', color='lightgray', linestyle='--')
    # make the grid lines appear behind the bars
    #ax.set_axisbelow(True)
    
    # Use the legend function to manually put the legend and colors
    leg = plt.legend(handles, names,bbox_to_anchor=(1,0.8) )
    
    for i in range(n_cases):
        leg.legend_handles[i].set_color(color_map[i])
        leg.legend_handles[i].set_alpha(alpha_val)
        
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, img_str + ' multi bar plot - Supplementary' + '.png'), dpi=300)

