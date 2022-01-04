from sklearn.linear_model import Ridge
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from sklearn.model_selection import cross_val_score,KFold,cross_validate
#from Feature_selection import Normalize1D

import pandas as pd

data=pd.read_csv('testdata\\testdata.csv')
X_total=data.iloc[:,1:-1]
S=StandardScaler()
X_total=S.fit_transform(X_total)
#X_total=normalize(X_total)
y_total=data['target']
# S2=Normalize1D()
# S2.fit(y_total)
# y_total=S2.transform(y_total)


Et=ExtraTreeRegressor()
RI=Ridge(tol=0.00001,alpha=0.00001,random_state=0)
ABR=AdaBoostRegressor(base_estimator=Et,random_state=0)
GBR=GradientBoostingRegressor(n_estimators=2048,learning_rate=0.1,random_state=0)
ETR=ExtraTreesRegressor(random_state=0,n_jobs=8,n_estimators=128)
RFR=RandomForestRegressor(random_state=0,n_jobs=8,n_estimators=128)
KRR=KNeighborsRegressor(n_jobs=8)
S=SVR(C=6)
LS=LinearSVR(C=6,random_state=65,max_iter=2500,loss='squared_epsilon_insensitive')

kfold=KFold(n_splits=5)
model_list=[GBR,S]
model_name=['GBR','SVR']
score_list=['r2','neg_mean_absolute_error','neg_mean_squared_error']
for i in range(len(model_list)):
    s_list=[]
    ten_fold=pd.DataFrame()
    ten_fold['n']=range(1,6)
    for score in score_list:
        #s=cross_val_score(model_list[i],X_total,y_total,scoring=score,cv=kfold,n_jobs=8)
        s=cross_validate(model_list[i],X_total,y_total,scoring=score,cv=kfold,n_jobs=8,return_train_score=True)
        if score=='r2':
            ten_fold['R2_test']=s['test_score']
            ten_fold['R2_train']=s['train_score']
        elif score=='neg_mean_absolute_error':
            ten_fold['MAE_test']=[-i for i in s['test_score']]
            ten_fold['MAE_train']=[-i for i in s['train_score']]
        else:
            ten_fold['MSE_test']=[-i for i in s['test_score']]
            ten_fold['MSE_train']=[-i for i in s['train_score']]
    ten_fold.to_csv(model_name[i]+'withNormalize_5fold.csv')

        

       
    