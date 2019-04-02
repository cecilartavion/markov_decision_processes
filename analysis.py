import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from itertools import product
from sklearn.random_projection import SparseRandomProjection
from helpers import reconstructionError
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

path = 'C:/Users/John/Dropbox/Dalton State College/georgia_tech/CSE7641/markov_decision_processes/'
#path = 'C:/Users/jasplund/Dropbox/Dalton State College/georgia_tech/CSE7641/unsupervised_learning_and_dim_reduction/'


#######################################
#######################################
#######################################
#######################################
#######################################
disc_values = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
rew_values = ['-0.5', '-0.9', '+0.1', '+0.5', '+0.9','-0.1']
rew_compare = []
for rew in ['less_05', 'less_09', 'more_01', 'more_05', 'more_09']:
    disc_avg = []    
    for disc in disc_values:
        df_test = pd.read_csv('frozen_lake_{}_0.1_0_0.5_0.0001_{}_optimal.csv'.format(rew,disc))
        disc_avg.append(np.average(df_test['reward']))
    rew_compare.append(disc_avg)

disc_avg = []
for disc in disc_values:
    df_test = pd.read_csv('frozen_lake_0.1_0_0.5_0.0001_{}_optimal.csv'.format(disc))
    disc_avg.append(np.average(df_test['reward']))
rew_compare.append(disc_avg)
rew_compare
f = plt.figure()
for rew in range(len(rew_values)):
    plt.plot(['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'],rew_compare[rew],label=rew_values[rew])
plt.title('Avg. Reward Frozen Lake 8x8',fontsize=20) 
plt.xlabel('Discount Factor',fontsize=20)
plt.ylabel('Reward',fontsize=20)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
f.savefig(path + 'discount_reward_fz.pdf')


rew_compare = []
for rew in ['less_05', 'less_09', 'more_01', 'more_05', 'more_09']:
    disc_avg = []    
    for eps in [0.1,0.3,0.5]:
        df_test = pd.read_csv('frozen_lake_{}_0.1_0_{}_0.0001_0.8_optimal.csv'.format(rew,eps))
        print(np.average(df_test['reward']))
    rew_compare.append(disc_avg)
        
for alpha in ['0.1','0.5','0.9']:
    df_test = pd.read_csv('frozen_lake_less_05_{}_0_0.1_0.0001_0.8_optimal.csv'.format(alpha))
    print(np.average(df_test['reward']))


base_wine = pd.read_csv(path + 'BASE/wine_acc.csv',sep=',')
ica_wine = pd.read_csv(path + 'ICA/wine_acc.csv',sep=',')
pca_wine = pd.read_csv(path + 'PCA/wine_acc.csv',sep=',')
rf_wine = pd.read_csv(path + 'RF/wine_acc.csv',sep=',')
rp_wine = pd.read_csv(path + 'RP/wine_acc.csv',sep=',')
base_adult = pd.read_csv(path + 'BASE/adult_acc.csv',sep=',')
ica_adult = pd.read_csv(path + 'ICA/adult_acc.csv',sep=',')
pca_adult = pd.read_csv(path + 'PCA/adult_acc.csv',sep=',')
rf_adult = pd.read_csv(path + 'RF/adult_acc.csv',sep=',')
rp_adult = pd.read_csv(path + 'RP/adult_acc.csv',sep=',')

base_wine = base_wine.T.reset_index()
base_wine.columns = ['clusters','GMM','Kmeans']
base_wine = base_wine.drop(0,axis=0).reset_index(drop=True)
ica_wine = ica_wine.T.reset_index()
ica_wine.columns = ['clusters','GMM','Kmeans']
ica_wine = ica_wine.drop(0,axis=0).reset_index(drop=True)
pca_wine = pca_wine.T.reset_index()
pca_wine.columns = ['clusters','GMM','Kmeans']
pca_wine = pca_wine.drop(0,axis=0).reset_index(drop=True)
rf_wine = rf_wine.T.reset_index()
rf_wine.columns = ['clusters','GMM','Kmeans']
rf_wine = rf_wine.drop(0,axis=0).reset_index(drop=True)
rp_wine = rp_wine.T.reset_index()
rp_wine.columns = ['clusters','GMM','Kmeans']
rp_wine = rp_wine.drop(0,axis=0).reset_index(drop=True)
base_adult = base_adult.T.reset_index()
base_adult.columns = ['clusters','GMM','Kmeans']
base_adult = base_adult.drop(0,axis=0).reset_index(drop=True)
ica_adult = ica_adult.T.reset_index()
ica_adult.columns = ['clusters','GMM','Kmeans']
ica_adult = ica_adult.drop(0,axis=0).reset_index(drop=True)
pca_adult = pca_adult.T.reset_index()
pca_adult.columns = ['clusters','GMM','Kmeans']
pca_adult = pca_adult.drop(0,axis=0).reset_index(drop=True)
rf_adult = rf_adult.T.reset_index()
rf_adult.columns = ['clusters','GMM','Kmeans']
rf_adult = rf_adult.drop(0,axis=0).reset_index(drop=True)
rp_adult = rp_adult.T.reset_index()
rp_adult.columns = ['clusters','GMM','Kmeans']
rp_adult = rp_adult.drop(0,axis=0).reset_index(drop=True)

####################################
####################################
####################################
####################################
####################################

f = plt.figure()
plt.plot(list(base_wine['clusters']),list(base_wine['GMM']), label='BASE GMM')
plt.plot(list(base_wine['clusters']),list(base_wine['Kmeans']), label='BASE KM')
plt.plot(list(ica_wine['clusters']),list(ica_wine['GMM']), label='ICA GMM')
plt.plot(list(ica_wine['clusters']),list(ica_wine['Kmeans']), label='ICA KM')
plt.plot(list(pca_wine['clusters']),list(pca_wine['GMM']), label='PCA GMM')
plt.plot(list(pca_wine['clusters']),list(pca_wine['Kmeans']), label='PCA KM')
plt.plot(list(rf_wine['clusters']),list(rf_wine['GMM']), label='RF GMM')
plt.plot(list(rf_wine['clusters']),list(rf_wine['Kmeans']), label='RF KM')
plt.plot(list(rp_wine['clusters']),list(rp_wine['GMM']), label='RP GMM')
plt.plot(list(rp_wine['clusters']),list(rp_wine['Kmeans']), label='RP KM')
#plt.ylim(0.5, 1.1)
plt.title('Wine Accuracy for DR',fontsize=20) 
plt.xlabel('Number of Clusters',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
f.savefig('all_cluster_acc_num1_wine.pdf')

#######################################
#######################################
#######################################
#######################################
#######################################

f = plt.figure()
plt.plot(list(base_adult['clusters']),list(base_adult['GMM']), label='BASE GMM')
plt.plot(list(base_adult['clusters']),list(base_adult['Kmeans']), label='BASE KM')
plt.plot(list(ica_adult['clusters']),list(ica_adult['GMM']), label='ICA GMM')
plt.plot(list(ica_adult['clusters']),list(ica_adult['Kmeans']), label='ICA KM')
plt.plot(list(pca_adult['clusters']),list(pca_adult['GMM']), label='PCA GMM')
plt.plot(list(pca_adult['clusters']),list(pca_adult['Kmeans']), label='PCA KM')
plt.plot(list(rf_adult['clusters']),list(rf_adult['GMM']), label='RF GMM')
plt.plot(list(rf_adult['clusters']),list(rf_adult['Kmeans']), label='RF KM')
plt.plot(list(rp_adult['clusters']),list(rp_adult['GMM']), label='RP GMM')
plt.plot(list(rp_adult['clusters']),list(rp_adult['Kmeans']), label='RP KM')
#plt.ylim(0.5, 1.1)
plt.title('Adult Accuracy for DR',fontsize=20) 
plt.xlabel('Number of Clusters',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
#f.savefig('all_cluster_acc_num1_adult.pdf')

####################################
####################################
####################################
####################################
####################################

f = plt.figure()
pca2 = PCA(random_state=5)
pca2.fit(wineX)
tmp2 = pd.Series(data = pca2.explained_variance_ratio_,index = range(11))
plt.plot(tmp2, label='Wine')
plt.title('PCA Explained Variance for Wine',fontsize=20) 
plt.xlabel('Eigenvalue Index',fontsize=20)
plt.ylabel('Explained Variance',fontsize=20)
#plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
f.savefig('wine_pca_explained_var.pdf')

f = plt.figure()
pca1 = PCA(random_state=5)
pca1.fit(adultX)
tmp1 = pd.Series(data = pca1.explained_variance_ratio_,index = range(106))
plt.plot(tmp1, label='Wine')
plt.title('PCA Explained Variance for Adult',fontsize=20) 
plt.xlabel('Eigenvalue Index',fontsize=20)
plt.ylabel('Explained Variance',fontsize=20)
#plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
#f.savefig('adult_pca_explained_var.pdf')

####################################
####################################
####################################
####################################
####################################

f = plt.figure()
dims_w = [2,3,4,5,6,7,8,9,10]
ica = FastICA(random_state=5)
kurt = {}
for dim in dims_w:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(wineX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()
kurt = pd.Series(kurt) 
plt.plot(kurt, label='Wine')
plt.title('ICA Kurtosis for Wine',fontsize=20) 
plt.xlabel('Number of Features',fontsize=20)
plt.ylabel('Kurtosis',fontsize=20)
#plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
f.savefig('wine_ica_kurtosis.pdf')


f = plt.figure()
dims_a = [2,5,10,20,30,40,50,60]
ica = FastICA(random_state=5)
kurt = {}
for dim in dims_a:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(adultX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()
kurt = pd.Series(kurt) 
plt.plot(kurt, label='Adult')
plt.title('ICA Kurtosis for Adult',fontsize=20) 
plt.xlabel('Number of Features',fontsize=20)
plt.ylabel('Kurtosis',fontsize=20)
#plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
f.savefig('adult_ica_kurtosis.pdf')

####################################
####################################
####################################
####################################
####################################

f = plt.figure()
tmp2 = defaultdict(dict)
for i,dim in product(range(10),dims_a):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(adultX)    
    tmp2[dim][i] = reconstructionError(rp, adultX)
tmp2 =pd.DataFrame(tmp2).T
tmp1 = defaultdict(dict)
for i,dim in product(range(10),dims_w):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(wineX)  
    tmp1[dim][i] = reconstructionError(rp, wineX)
tmp1 =pd.DataFrame(tmp1).T
plt.plot(tmp2, label='Adult')
plt.plot(tmp1, label='Wine')
plt.title('Reconstruction Error for Wine and Adult',fontsize=20) 
plt.xlabel('Number of Features',fontsize=20)
plt.ylabel('Reconstruction Error',fontsize=20)
#plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
f.savefig('adult_rp_recon_error.pdf')


####################################
####################################
####################################
####################################
####################################


rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
fs_wine = rfc.fit(wineX,wineY).feature_importances_ 
tmp = pd.Series(np.sort(fs_wine)[::-1])
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(wineX.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
f = plt.figure()
plt.title("Feature Importances for Wine Data", fontsize=20)
plt.bar(range(wineX.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(wineX.shape[1]), indices)
plt.xlim([-1, wineX.shape[1]])
plt.xlabel('Feature',fontsize=20)
plt.ylabel('Feature Importance',fontsize=20)
plt.tight_layout()
plt.show()
f.savefig('wine_rf_feature_importance.pdf')

####################################
####################################
####################################
####################################
####################################


rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
fs_adult = rfc.fit(adultX,adultY).feature_importances_ 
tmp = pd.Series(np.sort(fs_adult)[::-1])
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(adultX.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
f = plt.figure()
plt.title("Feature Importances for Adult")
plt.bar(range(adultX.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(adultX.shape[1]), indices)
plt.xlim([-1, adultX.shape[1]])
plt.xlabel('Feature',fontsize=20)
plt.ylabel('Feature Importance',fontsize=20)
plt.tight_layout()
plt.show()
f.savefig('adult_rf_feature_importance.pdf')

####################################
####################################
####################################
####################################
####################################

base_wine_cluster = pd.read_csv(path + 'BASE/wine_dim_reduc4.csv',sep=',')
ica_wine_cluster = pd.read_csv(path + 'ICA/wine_dim_reduc4.csv',sep=',')
pca_wine_cluster = pd.read_csv(path + 'PCA/wine_dim_reduc4.csv',sep=',')
rf_wine_cluster = pd.read_csv(path + 'RF/wine_dim_reduc4.csv',sep=',')
rp_wine_cluster = pd.read_csv(path + 'RP/wine_dim_reduc4.csv',sep=',')
#ica_wine = pd.read_csv(path + 'ICA/wine_acc.csv',sep=',')
#pca_wine = pd.read_csv(path + 'PCA/wine_acc.csv',sep=',')
#rf_wine = pd.read_csv(path + 'RF/wine_acc.csv',sep=',')
#rp_wine = pd.read_csv(path + 'RP/wine_acc.csv',sep=',')
#base_adult = pd.read_csv(path + 'BASE/adult_acc.csv',sep=',')
#ica_adult = pd.read_csv(path + 'ICA/adult_acc.csv',sep=',')
#pca_adult = pd.read_csv(path + 'PCA/adult_acc.csv',sep=',')
#rf_adult = pd.read_csv(path + 'RF/adult_acc.csv',sep=',')
#rp_adult = pd.read_csv(path + 'RP/adult_acc.csv',sep=',')

base_cluster_kmeans1 = base_wine_cluster[
        base_wine_cluster['param_NN__activation']=='relu'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
base_cluster_kmeans2 = base_wine_cluster[
        base_wine_cluster['param_NN__activation']=='logistic'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
ica_cluster_kmeans1 = ica_wine_cluster[
        ica_wine_cluster['param_NN__activation']=='relu'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
ica_cluster_kmeans2 = ica_wine_cluster[
        ica_wine_cluster['param_NN__activation']=='logistic'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
pca_cluster_kmeans1 = pca_wine_cluster[
        pca_wine_cluster['param_NN__activation']=='relu'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
pca_cluster_kmeans2 = pca_wine_cluster[
        pca_wine_cluster['param_NN__activation']=='logistic'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
rf_cluster_kmeans1 = rf_wine_cluster[
        rf_wine_cluster['param_NN__activation']=='relu'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
rf_cluster_kmeans2 = rf_wine_cluster[
        rf_wine_cluster['param_NN__activation']=='logistic'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
rp_cluster_kmeans1 = rp_wine_cluster[
        rp_wine_cluster['param_NN__activation']=='relu'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
rp_cluster_kmeans2 = rp_wine_cluster[
        rp_wine_cluster['param_NN__activation']=='logistic'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]

x = base_cluster_kmeans1['param_NN__alpha']
y_relu = np.zeros(9)
y_relu = list(y_relu)
y_log = np.zeros(9)
y_log = list(y_log)
y_relu_ica = np.zeros(9)
y_relu_ica = list(y_relu_ica)
y_log_ica = np.zeros(9)
y_log_ica = list(y_log_ica)
y_relu_pca = np.zeros(9)
y_relu_pca = list(y_relu_pca)
y_log_pca = np.zeros(9)
y_log_pca = list(y_log_pca)
y_relu_rf = np.zeros(9)
y_relu_rf = list(y_relu_rf)
y_log_rf = np.zeros(9)
y_log_rf = list(y_log_rf)
y_relu_rp = np.zeros(9)
y_relu_rp = list(y_relu_rp)
y_log_rp = np.zeros(9)
y_log_rp = list(y_log_rp)
for i in range(9):
  y_relu[i] = base_cluster_kmeans1['mean_test_score'].values[i::9]
  y_log[i] = base_cluster_kmeans2['mean_test_score'].values[i::9]
  y_relu_ica[i] = ica_cluster_kmeans1['mean_test_score'].values[i::9]
  y_log_ica[i] = ica_cluster_kmeans2['mean_test_score'].values[i::9]
  y_relu_pca[i] = pca_cluster_kmeans1['mean_test_score'].values[i::9]
  y_log_pca[i] = pca_cluster_kmeans2['mean_test_score'].values[i::9]
  y_relu_rf[i] = rf_cluster_kmeans1['mean_test_score'].values[i::9]
  y_log_rf[i] = rf_cluster_kmeans2['mean_test_score'].values[i::9]
  y_relu_rp[i] = rp_cluster_kmeans1['mean_test_score'].values[i::9]
  y_log_rp[i] = rp_cluster_kmeans2['mean_test_score'].values[i::9]
alphas = list(np.unique(base_cluster_kmeans1['param_NN__alpha']))
hidden_layer_names = np.unique(base_cluster_kmeans1['param_NN__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    if i == 0 and j == 1:
        df_relu = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu[i+3*j])],axis=1)
        df_relu.columns = ['alpha','alphas']
        df_relu = df_relu.set_index('alpha')        
        df_relu['alphas'].sort_index().plot(ax=axarr[i][j],label='BASE relu')
        df_log = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log[i+3*j])],axis=1)
        df_log.columns = ['alpha','alphas']
        df_log = df_log.set_index('alpha')
        df_log['alphas'].sort_index().plot(ax=axarr[i][j], label='BASE sigmoid')
        df_relu_ica = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_ica[i+3*j])],axis=1)
        df_relu_ica.columns = ['alpha','alphas']
        df_relu_ica = df_relu_ica.set_index('alpha')        
        df_relu_ica['alphas'].sort_index().plot(ax=axarr[i][j],label='ICA relu')
        df_log_ica = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_ica[i+3*j])],axis=1)
        df_log_ica.columns = ['alpha','alphas']
        df_log_ica = df_log_ica.set_index('alpha')
        df_log_ica['alphas'].sort_index().plot(ax=axarr[i][j], label='ICA sigmoid')
        df_relu_pca = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_pca[i+3*j])],axis=1)
        df_relu_pca.columns = ['alpha','alphas']
        df_relu_pca = df_relu_pca.set_index('alpha')        
        df_relu_pca['alphas'].sort_index().plot(ax=axarr[i][j],label='PCA relu')
        df_log_pca = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_pca[i+3*j])],axis=1)
        df_log_pca.columns = ['alpha','alphas']
        df_log_pca = df_log_pca.set_index('alpha')
        df_log_pca['alphas'].sort_index().plot(ax=axarr[i][j], label='PCA sigmoid')
        df_relu_rf = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_rf[i+3*j])],axis=1)
        df_relu_rf.columns = ['alpha','alphas']
        df_relu_rf = df_relu_rf.set_index('alpha')        
        df_relu_rf['alphas'].sort_index().plot(ax=axarr[i][j],label='RF relu')
        df_log_rf = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_rf[i+3*j])],axis=1)
        df_log_rf.columns = ['alpha','alphas']
        df_log_rf = df_log_rf.set_index('alpha')
        df_log_rf['alphas'].sort_index().plot(ax=axarr[i][j], label='RF sigmoid')
        df_relu_rp = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_rp[i+3*j])],axis=1)
        df_relu_rp.columns = ['alpha','alphas']
        df_relu_rp = df_relu_rp.set_index('alpha')        
        df_relu_rp['alphas'].sort_index().plot(ax=axarr[i][j],label='RP relu')
        df_log_rp = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_rp[i+3*j])],axis=1)
        df_log_rp.columns = ['alpha','alphas']
        df_log_rp = df_log_rp.set_index('alpha')
        df_log_rp['alphas'].sort_index().plot(ax=axarr[i][j], label='RP sigmoid')
#        df_relu_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_train[i+3*j])],axis=1)
#        df_relu_train.columns = ['step size','alphas']
#        df_relu_train = df_relu_train.set_index('step size')        
#        df_relu_train['alphas'].sort_index().plot(ax=axarr[i][j],label='relu train')
#        df_log_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_train[i+3*j])],axis=1)
#        df_log_train.columns = ['step size','alphas']
#        df_log_train = df_log_train.set_index('step size')
#        df_log_train['alphas'].sort_index().plot(ax=axarr[i][j], label='sigmoid train')
        axarr[i,j].legend(loc='lower left',fontsize=8)
    else:
        df_relu = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu[i+3*j])],axis=1)
        df_relu.columns = ['alpha','alphas']
        df_relu = df_relu.set_index('alpha')        
        df_relu['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log[i+3*j])],axis=1)
        df_log.columns = ['alpha','alphas']
        df_log = df_log.set_index('alpha')
        df_log['alphas'].sort_index().plot(ax=axarr[i][j])
        df_relu_ica = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_ica[i+3*j])],axis=1)
        df_relu_ica.columns = ['alpha','alphas']
        df_relu_ica = df_relu_ica.set_index('alpha')        
        df_relu_ica['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log_ica = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_ica[i+3*j])],axis=1)
        df_log_ica.columns = ['alpha','alphas']
        df_log_ica = df_log_ica.set_index('alpha')
        df_log_ica['alphas'].sort_index().plot(ax=axarr[i][j])
        df_relu_pca = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_pca[i+3*j])],axis=1)
        df_relu_pca.columns = ['alpha','alphas']
        df_relu_pca = df_relu_pca.set_index('alpha')        
        df_relu_pca['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log_pca = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_pca[i+3*j])],axis=1)
        df_log_pca.columns = ['alpha','alphas']
        df_log_pca = df_log_pca.set_index('alpha')
        df_log_pca['alphas'].sort_index().plot(ax=axarr[i][j])
        df_relu_rf = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_rf[i+3*j])],axis=1)
        df_relu_rf.columns = ['alpha','alphas']
        df_relu_rf = df_relu_rf.set_index('alpha')        
        df_relu_rf['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log_rf = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_rf[i+3*j])],axis=1)
        df_log_rf.columns = ['alpha','alphas']
        df_log_rf = df_log_rf.set_index('alpha')
        df_log_rf['alphas'].sort_index().plot(ax=axarr[i][j])
        df_relu_rp = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_rp[i+3*j])],axis=1)
        df_relu_rp.columns = ['alpha','alphas']
        df_relu_rp = df_relu_rp.set_index('alpha')        
        df_relu_rp['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log_rp = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_rp[i+3*j])],axis=1)
        df_log_rp.columns = ['alpha','alphas']
        df_log_rp = df_log_rp.set_index('alpha')
        df_log_rp['alphas'].sort_index().plot(ax=axarr[i][j])
#        df_relu_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_train[i+3*j])],axis=1)
#        df_relu_train.columns = ['step size','alphas']
#        df_relu_train = df_relu_train.set_index('step size')        
#        df_relu_train['alphas'].sort_index().plot(ax=axarr[i][j])
#        df_log_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_train[i+3*j])],axis=1)
#        df_log_train.columns = ['step size','alphas']
#        df_log_train = df_log_train.set_index('step size')
#        df_log_train['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
#    axarr[i,j].set_xscale('log')
#fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
#fig.subplots_adjust(bottom = 0.256)
#fig.legend(loc='lower left')
plt.tight_layout()
fig.get_figure()
fig.savefig('validation_wine_all_dim_4.pdf')


####################################
####################################
####################################
####################################
####################################

base_wine_cluster = pd.read_csv(path + 'BASE/wine_dim_reduc5.csv',sep=',')
ica_wine_cluster = pd.read_csv(path + 'ICA/wine_dim_reduc5.csv',sep=',')
pca_wine_cluster = pd.read_csv(path + 'PCA/wine_dim_reduc5.csv',sep=',')
rf_wine_cluster = pd.read_csv(path + 'RF/wine_dim_reduc5.csv',sep=',')
rp_wine_cluster = pd.read_csv(path + 'RP/wine_dim_reduc5.csv',sep=',')
#ica_wine = pd.read_csv(path + 'ICA/wine_acc.csv',sep=',')
#pca_wine = pd.read_csv(path + 'PCA/wine_acc.csv',sep=',')
#rf_wine = pd.read_csv(path + 'RF/wine_acc.csv',sep=',')
#rp_wine = pd.read_csv(path + 'RP/wine_acc.csv',sep=',')
#base_adult = pd.read_csv(path + 'BASE/adult_acc.csv',sep=',')
#ica_adult = pd.read_csv(path + 'ICA/adult_acc.csv',sep=',')
#pca_adult = pd.read_csv(path + 'PCA/adult_acc.csv',sep=',')
#rf_adult = pd.read_csv(path + 'RF/adult_acc.csv',sep=',')
#rp_adult = pd.read_csv(path + 'RP/adult_acc.csv',sep=',')

base_cluster_kmeans1 = base_wine_cluster[
        base_wine_cluster['param_NN__activation']=='relu'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
base_cluster_kmeans2 = base_wine_cluster[
        base_wine_cluster['param_NN__activation']=='logistic'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
ica_cluster_kmeans1 = ica_wine_cluster[
        ica_wine_cluster['param_NN__activation']=='relu'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
ica_cluster_kmeans2 = ica_wine_cluster[
        ica_wine_cluster['param_NN__activation']=='logistic'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
pca_cluster_kmeans1 = pca_wine_cluster[
        pca_wine_cluster['param_NN__activation']=='relu'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
pca_cluster_kmeans2 = pca_wine_cluster[
        pca_wine_cluster['param_NN__activation']=='logistic'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
rf_cluster_kmeans1 = rf_wine_cluster[
        rf_wine_cluster['param_NN__activation']=='relu'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
rf_cluster_kmeans2 = rf_wine_cluster[
        rf_wine_cluster['param_NN__activation']=='logistic'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
rp_cluster_kmeans1 = rp_wine_cluster[
        rp_wine_cluster['param_NN__activation']=='relu'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]
rp_cluster_kmeans2 = rp_wine_cluster[
        rp_wine_cluster['param_NN__activation']=='logistic'][['mean_test_score',
                                             'param_NN__alpha',
                                             'param_NN__hidden_layer_sizes']]

x = base_cluster_kmeans1['param_NN__alpha']
y_relu = np.zeros(9)
y_relu = list(y_relu)
y_log = np.zeros(9)
y_log = list(y_log)
y_relu_ica = np.zeros(9)
y_relu_ica = list(y_relu_ica)
y_log_ica = np.zeros(9)
y_log_ica = list(y_log_ica)
y_relu_pca = np.zeros(9)
y_relu_pca = list(y_relu_pca)
y_log_pca = np.zeros(9)
y_log_pca = list(y_log_pca)
y_relu_rf = np.zeros(9)
y_relu_rf = list(y_relu_rf)
y_log_rf = np.zeros(9)
y_log_rf = list(y_log_rf)
y_relu_rp = np.zeros(9)
y_relu_rp = list(y_relu_rp)
y_log_rp = np.zeros(9)
y_log_rp = list(y_log_rp)
for i in range(9):
  y_relu[i] = base_cluster_kmeans1['mean_test_score'].values[i::9]
  y_log[i] = base_cluster_kmeans2['mean_test_score'].values[i::9]
  y_relu_ica[i] = ica_cluster_kmeans1['mean_test_score'].values[i::9]
  y_log_ica[i] = ica_cluster_kmeans2['mean_test_score'].values[i::9]
  y_relu_pca[i] = pca_cluster_kmeans1['mean_test_score'].values[i::9]
  y_log_pca[i] = pca_cluster_kmeans2['mean_test_score'].values[i::9]
  y_relu_rf[i] = rf_cluster_kmeans1['mean_test_score'].values[i::9]
  y_log_rf[i] = rf_cluster_kmeans2['mean_test_score'].values[i::9]
  y_relu_rp[i] = rp_cluster_kmeans1['mean_test_score'].values[i::9]
  y_log_rp[i] = rp_cluster_kmeans2['mean_test_score'].values[i::9]
alphas = list(np.unique(base_cluster_kmeans1['param_NN__alpha']))
hidden_layer_names = np.unique(base_cluster_kmeans1['param_NN__hidden_layer_sizes'])
fig, axarr = plt.subplots(3, 3, figsize=(12, 8),sharex='col', sharey='row')
plt.xscale('log')
for i in range(3):
  for j in range(3):
    if i == 0 and j == 0:
        df_relu = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu[i+3*j])],axis=1)
        df_relu.columns = ['alpha','alphas']
        df_relu = df_relu.set_index('alpha')        
        df_relu['alphas'].sort_index().plot(ax=axarr[i][j],label='BASE relu')
        df_log = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log[i+3*j])],axis=1)
        df_log.columns = ['alpha','alphas']
        df_log = df_log.set_index('alpha')
        df_log['alphas'].sort_index().plot(ax=axarr[i][j], label='BASE sigmoid')
        df_relu_ica = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_ica[i+3*j])],axis=1)
        df_relu_ica.columns = ['alpha','alphas']
        df_relu_ica = df_relu_ica.set_index('alpha')        
        df_relu_ica['alphas'].sort_index().plot(ax=axarr[i][j],label='ICA relu')
        df_log_ica = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_ica[i+3*j])],axis=1)
        df_log_ica.columns = ['alpha','alphas']
        df_log_ica = df_log_ica.set_index('alpha')
        df_log_ica['alphas'].sort_index().plot(ax=axarr[i][j], label='ICA sigmoid')
        df_relu_pca = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_pca[i+3*j])],axis=1)
        df_relu_pca.columns = ['alpha','alphas']
        df_relu_pca = df_relu_pca.set_index('alpha')        
        df_relu_pca['alphas'].sort_index().plot(ax=axarr[i][j],label='PCA relu')
        df_log_pca = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_pca[i+3*j])],axis=1)
        df_log_pca.columns = ['alpha','alphas']
        df_log_pca = df_log_pca.set_index('alpha')
        df_log_pca['alphas'].sort_index().plot(ax=axarr[i][j], label='PCA sigmoid')
        df_relu_rf = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_rf[i+3*j])],axis=1)
        df_relu_rf.columns = ['alpha','alphas']
        df_relu_rf = df_relu_rf.set_index('alpha')        
        df_relu_rf['alphas'].sort_index().plot(ax=axarr[i][j],label='RF relu')
        df_log_rf = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_rf[i+3*j])],axis=1)
        df_log_rf.columns = ['alpha','alphas']
        df_log_rf = df_log_rf.set_index('alpha')
        df_log_rf['alphas'].sort_index().plot(ax=axarr[i][j], label='RF sigmoid')
        df_relu_rp = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_rp[i+3*j])],axis=1)
        df_relu_rp.columns = ['alpha','alphas']
        df_relu_rp = df_relu_rp.set_index('alpha')        
        df_relu_rp['alphas'].sort_index().plot(ax=axarr[i][j],label='RP relu')
        df_log_rp = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_rp[i+3*j])],axis=1)
        df_log_rp.columns = ['alpha','alphas']
        df_log_rp = df_log_rp.set_index('alpha')
        df_log_rp['alphas'].sort_index().plot(ax=axarr[i][j], label='RP sigmoid')
#        df_relu_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_train[i+3*j])],axis=1)
#        df_relu_train.columns = ['step size','alphas']
#        df_relu_train = df_relu_train.set_index('step size')        
#        df_relu_train['alphas'].sort_index().plot(ax=axarr[i][j],label='relu train')
#        df_log_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_train[i+3*j])],axis=1)
#        df_log_train.columns = ['step size','alphas']
#        df_log_train = df_log_train.set_index('step size')
#        df_log_train['alphas'].sort_index().plot(ax=axarr[i][j], label='sigmoid train')
        axarr[i,j].legend(loc='lower left',fontsize=8)
    else:
        df_relu = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu[i+3*j])],axis=1)
        df_relu.columns = ['alpha','alphas']
        df_relu = df_relu.set_index('alpha')        
        df_relu['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log[i+3*j])],axis=1)
        df_log.columns = ['alpha','alphas']
        df_log = df_log.set_index('alpha')
        df_log['alphas'].sort_index().plot(ax=axarr[i][j])
        df_relu_ica = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_ica[i+3*j])],axis=1)
        df_relu_ica.columns = ['alpha','alphas']
        df_relu_ica = df_relu_ica.set_index('alpha')        
        df_relu_ica['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log_ica = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_ica[i+3*j])],axis=1)
        df_log_ica.columns = ['alpha','alphas']
        df_log_ica = df_log_ica.set_index('alpha')
        df_log_ica['alphas'].sort_index().plot(ax=axarr[i][j])
        df_relu_pca = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_pca[i+3*j])],axis=1)
        df_relu_pca.columns = ['alpha','alphas']
        df_relu_pca = df_relu_pca.set_index('alpha')        
        df_relu_pca['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log_pca = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_pca[i+3*j])],axis=1)
        df_log_pca.columns = ['alpha','alphas']
        df_log_pca = df_log_pca.set_index('alpha')
        df_log_pca['alphas'].sort_index().plot(ax=axarr[i][j])
        df_relu_rf = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_rf[i+3*j])],axis=1)
        df_relu_rf.columns = ['alpha','alphas']
        df_relu_rf = df_relu_rf.set_index('alpha')        
        df_relu_rf['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log_rf = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_rf[i+3*j])],axis=1)
        df_log_rf.columns = ['alpha','alphas']
        df_log_rf = df_log_rf.set_index('alpha')
        df_log_rf['alphas'].sort_index().plot(ax=axarr[i][j])
        df_relu_rp = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_rp[i+3*j])],axis=1)
        df_relu_rp.columns = ['alpha','alphas']
        df_relu_rp = df_relu_rp.set_index('alpha')        
        df_relu_rp['alphas'].sort_index().plot(ax=axarr[i][j])
        df_log_rp = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_rp[i+3*j])],axis=1)
        df_log_rp.columns = ['alpha','alphas']
        df_log_rp = df_log_rp.set_index('alpha')
        df_log_rp['alphas'].sort_index().plot(ax=axarr[i][j])
#        df_relu_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_relu_train[i+3*j])],axis=1)
#        df_relu_train.columns = ['step size','alphas']
#        df_relu_train = df_relu_train.set_index('step size')        
#        df_relu_train['alphas'].sort_index().plot(ax=axarr[i][j])
#        df_log_train = pd.concat([pd.DataFrame(alphas),pd.DataFrame(y_log_train[i+3*j])],axis=1)
#        df_log_train.columns = ['step size','alphas']
#        df_log_train = df_log_train.set_index('step size')
#        df_log_train['alphas'].sort_index().plot(ax=axarr[i][j])
    axarr[i, j].set_title(hidden_layer_names[i+3*j],fontsize= 20)
    axarr[i, j].set_ylim([0,1])
    axarr[i,j].set_xscale('log')
#    axarr[i,j].set_xscale('log')
#fig.suptitle('Rectified Linear Unit Function',fontsize= 20)
fig.text(0.06, 0.5, 'Accuracy', va='center', rotation='vertical',fontsize= 15)
#fig.subplots_adjust(bottom = 0.256)
#fig.legend(loc='lower left')
fig.get_figure()
fig.savefig('validation_wine_all_dim_5.pdf')

####################################
####################################
####################################
####################################
####################################
####################################

fig = plt.figure()
base_wine_cluster = pd.read_csv(path + 'BASE/wine_dim_reduc_test_results4.csv',sep=',')
ica_wine_cluster = pd.read_csv(path + 'ICA/wine_dim_reduc_test_results4.csv',sep=',')
pca_wine_cluster = pd.read_csv(path + 'PCA/wine_dim_reduc_test_results4.csv',sep=',')
rf_wine_cluster = pd.read_csv(path + 'RF/wine_dim_reduc_test_results4.csv',sep=',')
rp_wine_cluster = pd.read_csv(path + 'RP/wine_dim_reduc_test_results4.csv',sep=',')
#ica_wine = pd.read_csv(path + 'ICA/wine_acc.csv',sep=',')
#pca_wine = pd.read_csv(path + 'PCA/wine_acc.csv',sep=',')
#rf_wine = pd.read_csv(path + 'RF/wine_acc.csv',sep=',')
#rp_wine = pd.read_csv(path + 'RP/wine_acc.csv',sep=',')
#base_adult = pd.read_csv(path + 'BASE/adult_acc.csv',sep=',')
#ica_adult = pd.read_csv(path + 'ICA/adult_acc.csv',sep=',')
#pca_adult = pd.read_csv(path + 'PCA/adult_acc.csv',sep=',')
#rf_adult = pd.read_csv(path + 'RF/adult_acc.csv',sep=',')
#rp_adult = pd.read_csv(path + 'RP/adult_acc.csv',sep=',')

base_cluster = base_wine_cluster[['param_NN__max_iter','test acc']]
ica_cluster = ica_wine_cluster[['param_NN__max_iter','test acc']]
pca_cluster = pca_wine_cluster[['param_NN__max_iter','test acc']]
rf_cluster = rf_wine_cluster[['param_NN__max_iter','test acc']]
rp_cluster = rp_wine_cluster[['param_NN__max_iter','test acc']]

plt.plot(list(base_cluster['param_NN__max_iter']),list(base_cluster['test acc']), label='BASE')
plt.plot(list(ica_cluster['param_NN__max_iter']),list(ica_cluster['test acc']), label='ICA')
plt.plot(list(pca_cluster['param_NN__max_iter']),list(pca_cluster['test acc']), label='PCA')
plt.plot(list(rf_cluster['param_NN__max_iter']),list(rf_cluster['test acc']), label='RF')
plt.plot(list(rp_cluster['param_NN__max_iter']),list(rp_cluster['test acc']), label='RP')

plt.title("Accuracy Scores for Test Data with DR methods",fontsize=20)
plt.xscale('log')
plt.xlim([-1, 3400])
plt.xlabel('Number of Max Iterations',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.tight_layout()
plt.legend(loc='lower right',fontsize=15)
plt.show()
fig.get_figure()
fig.savefig('final_accuracy_4.pdf')


####################################
####################################
####################################
####################################
####################################
####################################

fig = plt.figure()
base_wine_cluster = pd.read_csv(path + 'BASE/wine_dim_reduc_test_results5.csv',sep=',')
ica_wine_cluster = pd.read_csv(path + 'ICA/wine_dim_reduc_test_results5.csv',sep=',')
pca_wine_cluster = pd.read_csv(path + 'PCA/wine_dim_reduc_test_results5.csv',sep=',')
rf_wine_cluster = pd.read_csv(path + 'RF/wine_dim_reduc_test_results5.csv',sep=',')
rp_wine_cluster = pd.read_csv(path + 'RP/wine_dim_reduc_test_results5.csv',sep=',')
base_wine_cluster_gmm = pd.read_csv(path + 'BASE/wine_dim_reduc_test_results5_gmm.csv',sep=',')
ica_wine_cluster_gmm = pd.read_csv(path + 'ICA/wine_dim_reduc_test_results5_gmm.csv',sep=',')
pca_wine_cluster_gmm = pd.read_csv(path + 'PCA/wine_dim_reduc_test_results5_gmm.csv',sep=',')
rf_wine_cluster_gmm = pd.read_csv(path + 'RF/wine_dim_reduc_test_results5_gmm.csv',sep=',')
rp_wine_cluster_gmm = pd.read_csv(path + 'RP/wine_dim_reduc_test_results5_gmm.csv',sep=',')
#ica_wine = pd.read_csv(path + 'ICA/wine_acc.csv',sep=',')
#pca_wine = pd.read_csv(path + 'PCA/wine_acc.csv',sep=',')
#rf_wine = pd.read_csv(path + 'RF/wine_acc.csv',sep=',')
#rp_wine = pd.read_csv(path + 'RP/wine_acc.csv',sep=',')
#base_adult = pd.read_csv(path + 'BASE/adult_acc.csv',sep=',')
#ica_adult = pd.read_csv(path + 'ICA/adult_acc.csv',sep=',')
#pca_adult = pd.read_csv(path + 'PCA/adult_acc.csv',sep=',')
#rf_adult = pd.read_csv(path + 'RF/adult_acc.csv',sep=',')
#rp_adult = pd.read_csv(path + 'RP/adult_acc.csv',sep=',')

base_cluster = base_wine_cluster[['param_NN__max_iter','test acc']]
ica_cluster = ica_wine_cluster[['param_NN__max_iter','test acc']]
pca_cluster = pca_wine_cluster[['param_NN__max_iter','test acc']]
rf_cluster = rf_wine_cluster[['param_NN__max_iter','test acc']]
rp_cluster = rp_wine_cluster[['param_NN__max_iter','test acc']]
base_cluster_gmm = base_wine_cluster_gmm[['param_NN__max_iter','test acc']]
ica_cluster_gmm = ica_wine_cluster_gmm[['param_NN__max_iter','test acc']]
pca_cluster_gmm = pca_wine_cluster_gmm[['param_NN__max_iter','test acc']]
rf_cluster_gmm = rf_wine_cluster_gmm[['param_NN__max_iter','test acc']]
rp_cluster_gmm = rp_wine_cluster_gmm[['param_NN__max_iter','test acc']]

plt.plot(list(base_cluster['param_NN__max_iter']),list(base_cluster['test acc']), label='BASE KM')
plt.plot(list(ica_cluster['param_NN__max_iter']),list(ica_cluster['test acc']), label='ICA KM')
plt.plot(list(pca_cluster['param_NN__max_iter']),list(pca_cluster['test acc']), label='PCA KM')
plt.plot(list(rf_cluster['param_NN__max_iter']),list(rf_cluster['test acc']), label='RF KM')
plt.plot(list(rp_cluster['param_NN__max_iter']),list(rp_cluster['test acc']), label='RP KM')
plt.plot(list(base_cluster_gmm['param_NN__max_iter']),list(base_cluster_gmm['test acc']), label='BASE GMM')
plt.plot(list(ica_cluster_gmm['param_NN__max_iter']),list(ica_cluster_gmm['test acc']), label='ICA GMM')
plt.plot(list(pca_cluster_gmm['param_NN__max_iter']),list(pca_cluster_gmm['test acc']), label='PCA GMM')
plt.plot(list(rf_cluster_gmm['param_NN__max_iter']),list(rf_cluster_gmm['test acc']), label='RF GMM')
plt.plot(list(rp_cluster_gmm['param_NN__max_iter']),list(rp_cluster_gmm['test acc']), label='RP GMM')


plt.title("Accuracy Scores for Test Data with DR methods",fontsize=20)
plt.xscale('log')
plt.xlim([-1, 3400])
plt.xlabel('Number of Max Iterations',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(loc='lower right',fontsize=15)
plt.tight_layout()
plt.show()
fig.get_figure()
fig.savefig('final_accuracy_5.pdf')

####################################
####################################
####################################
####################################
####################################
####################################

#base_train_LC = pd.read_csv('BASE/LC_train_4.csv',sep=',')
#base_test_LC = pd.read_csv('BASE/LC_test_4.csv',sep=',')
ica_train_LC = pd.read_csv('ICA/LC_train_4.csv',sep=',')
ica_test_LC = pd.read_csv('ICA/LC_test_4.csv',sep=',')
rp_train_LC = pd.read_csv('RP/LC_train_4.csv',sep=',')
rp_test_LC = pd.read_csv('RP/LC_test_4.csv',sep=',')
#base_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
#base_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
ica_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
ica_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
rp_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
rp_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
#base_train_LC = base_train_LC.set_index('training_sizes')
#base_test_LC = base_test_LC.set_index('training_sizes')
ica_train_LC = ica_train_LC.set_index('training_sizes')
ica_test_LC = ica_test_LC.set_index('training_sizes')
rp_train_LC = rp_train_LC.set_index('training_sizes')
rp_test_LC = rp_test_LC.set_index('training_sizes')


#mean_train_size_score = np.mean(base_train_LC,axis=1)
#mean_test_size_score = np.mean(base_test_LC,axis=1)
ica_mean_train_size_score = np.mean(ica_train_LC,axis=1)
ica_mean_test_size_score = np.mean(ica_test_LC,axis=1)
rp_mean_train_size_score = np.mean(rp_train_LC,axis=1)
rp_mean_test_size_score = np.mean(rp_test_LC,axis=1)
f = plt.figure()
#plt.plot(list(mean_train_size_score.index),list(mean_train_size_score), label='BASE Train')
#plt.plot(list(mean_test_size_score.index),list(mean_test_size_score), label = 'BASE Test')
plt.plot(list(ica_mean_train_size_score.index),list(ica_mean_train_size_score), label='ICA Train')
plt.plot(list(ica_mean_test_size_score.index),list(ica_mean_test_size_score), label = 'ICA Test')
plt.plot(list(rp_mean_train_size_score.index),list(rp_mean_train_size_score), label='RP Train')
plt.plot(list(rp_mean_test_size_score.index),list(rp_mean_test_size_score), label = 'RP Test')
#plt.xlim(0,3400)
plt.ylim(0, 1.1)
plt.title('Learning Curves for Wine Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(loc='upper left',fontsize=15)
plt.tight_layout()
plt.show()
f.savefig('LC_all_dim_4a.pdf')


####################################
####################################
####################################
####################################
####################################
####################################

base_train_LC = pd.read_csv('RF/LC_train_4.csv',sep=',')
base_test_LC = pd.read_csv('RF/LC_test_4.csv',sep=',')
ica_train_LC = pd.read_csv('PCA/LC_train_4.csv',sep=',')
ica_test_LC = pd.read_csv('PCA/LC_test_4.csv',sep=',')

base_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
base_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
ica_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
ica_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']

base_train_LC = base_train_LC.set_index('training_sizes')
base_test_LC = base_test_LC.set_index('training_sizes')
ica_train_LC = ica_train_LC.set_index('training_sizes')
ica_test_LC = ica_test_LC.set_index('training_sizes')
mean_train_size_score = np.mean(base_train_LC,axis=1)
mean_test_size_score = np.mean(base_test_LC,axis=1)
ica_mean_train_size_score = np.mean(ica_train_LC,axis=1)
ica_mean_test_size_score = np.mean(ica_test_LC,axis=1)
f = plt.figure()
plt.plot(list(mean_train_size_score.index),list(mean_train_size_score), label='RF Train')
plt.plot(list(mean_test_size_score.index),list(mean_test_size_score), label = 'RF Test')
plt.plot(list(ica_mean_train_size_score.index),list(ica_mean_train_size_score), label='PCA Train')
plt.plot(list(ica_mean_test_size_score.index),list(ica_mean_test_size_score), label = 'PCA Test')
#plt.xlim(0,3400)
plt.ylim(0, 1.1)
plt.title('Learning Curves for Wine Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(loc='upper left',fontsize=15)
plt.tight_layout()
plt.show()
f.savefig('LC_all_dim_4b.pdf')


####################################
####################################
####################################
####################################
####################################
####################################

#base_train_LC = pd.read_csv('BASE/LC_train_5.csv',sep=',')
#base_test_LC = pd.read_csv('BASE/LC_test_5.csv',sep=',')
ica_train_LC = pd.read_csv('ICA/LC_train_5.csv',sep=',')
ica_test_LC = pd.read_csv('ICA/LC_test_5.csv',sep=',')
rp_train_LC = pd.read_csv('RP/LC_train_5.csv',sep=',')
rp_test_LC = pd.read_csv('RP/LC_test_5.csv',sep=',')
#base_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
#base_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
ica_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
ica_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
rp_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
rp_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
#base_train_LC = base_train_LC.set_index('training_sizes')
#base_test_LC = base_test_LC.set_index('training_sizes')
ica_train_LC = ica_train_LC.set_index('training_sizes')
ica_test_LC = ica_test_LC.set_index('training_sizes')
rp_train_LC = rp_train_LC.set_index('training_sizes')
rp_test_LC = rp_test_LC.set_index('training_sizes')

#mean_train_size_score = np.mean(base_train_LC,axis=1)
#mean_test_size_score = np.mean(base_test_LC,axis=1)
ica_mean_train_size_score = np.mean(ica_train_LC,axis=1)
ica_mean_test_size_score = np.mean(ica_test_LC,axis=1)
rp_mean_train_size_score = np.mean(rp_train_LC,axis=1)
rp_mean_test_size_score = np.mean(rp_test_LC,axis=1)
f = plt.figure()
#plt.plot(list(mean_train_size_score.index),list(mean_train_size_score), label='BASE Train')
#plt.plot(list(mean_test_size_score.index),list(mean_test_size_score), label = 'BASE Test')
plt.plot(list(ica_mean_train_size_score.index),list(ica_mean_train_size_score), label='ICA Train')
plt.plot(list(ica_mean_test_size_score.index),list(ica_mean_test_size_score), label = 'ICA Test')
plt.plot(list(rp_mean_train_size_score.index),list(rp_mean_train_size_score), label='RP Train')
plt.plot(list(rp_mean_test_size_score.index),list(rp_mean_test_size_score), label = 'RP Test')
#plt.xlim(0,3400)
plt.ylim(0, 1.1)
plt.title('Learning Curves for Wine Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(loc='lower right',fontsize=15)
plt.tight_layout()
plt.show()
f.savefig('LC_all_dim_5a.pdf')


####################################
####################################
####################################
####################################
####################################
####################################

base_train_LC = pd.read_csv('RF/LC_train_5.csv',sep=',')
base_test_LC = pd.read_csv('RF/LC_test_5.csv',sep=',')
ica_train_LC = pd.read_csv('PCA/LC_train_5.csv',sep=',')
ica_test_LC = pd.read_csv('PCA/LC_test_5.csv',sep=',')

base_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
base_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
ica_train_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']
ica_test_LC.columns = ['training_sizes','fold1','fold2','fold3','fold4','fold5']

base_train_LC = base_train_LC.set_index('training_sizes')
base_test_LC = base_test_LC.set_index('training_sizes')
ica_train_LC = ica_train_LC.set_index('training_sizes')
ica_test_LC = ica_test_LC.set_index('training_sizes')


mean_train_size_score = np.mean(base_train_LC,axis=1)
mean_test_size_score = np.mean(base_test_LC,axis=1)
ica_mean_train_size_score = np.mean(ica_train_LC,axis=1)
ica_mean_test_size_score = np.mean(ica_test_LC,axis=1)

f = plt.figure()
plt.plot(list(mean_train_size_score.index),list(mean_train_size_score), label='RF Train')
plt.plot(list(mean_test_size_score.index),list(mean_test_size_score), label = 'RF Test')
plt.plot(list(ica_mean_train_size_score.index),list(ica_mean_train_size_score), label='PCA Train')
plt.plot(list(ica_mean_test_size_score.index),list(ica_mean_test_size_score), label = 'PCA Test')
#plt.xlim(0,3400)
plt.ylim(0, 1.1)
plt.title('Learning Curves for Wine Data',fontsize=20) 
plt.xlabel('Number of training examples',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.legend(loc='upper left',fontsize=15)
plt.tight_layout()
plt.show()
f.savefig('LC_all_dim_5b.pdf')

###############################
###############################
###############################
###############################
###############################

base_sse = pd.read_csv('BASE/SSE.csv',sep=',')
base_sse.columns = ['clusters', 'Adult SSE', 'Wine SSE']

f = plt.figure()
plt.plot(list(base_sse['clusters']),list(base_sse['Wine SSE']), label='Wine SSE')


#plt.xlim(0,3400)
#plt.ylim(0, 1.1)
plt.title('SSE for Wine Data',fontsize=20) 
plt.xlabel('Number of clusters',fontsize=20)
plt.ylabel('SSE',fontsize=20)
#plt.legend(loc='upper left',fontsize=15)
plt.tight_layout()
plt.show()
f.savefig('wine_sse.pdf')

f = plt.figure()
plt.plot(list(base_sse['clusters']),list(base_sse['Adult SSE']), label='Adult SSE')
#plt.xlim(0,3400)
#plt.ylim(0, 1.1)
plt.title('SSE for Adult Data',fontsize=20) 
plt.xlabel('Number of clusters',fontsize=20)
plt.ylabel('SSE',fontsize=20)
#plt.legend(loc='upper left',fontsize=15)
plt.tight_layout()
plt.show()
f.savefig('adult_sse.pdf')