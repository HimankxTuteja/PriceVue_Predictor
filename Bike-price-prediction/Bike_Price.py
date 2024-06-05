import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
os.chdir('E:\\prasad\\practice\\My Working Projects\\Completed\\Bike Price Prediction')
df=pd.read_csv('bike.csv')
df.head(3)
df.isnull().sum()
df['name'].unique()
name_map=df['name'].value_counts()
df['name']=df['name'].map(name_map)
df.head()
df['name'].unique()
df[df.loc[:,'name']==27]['ex_showroom_price'].replace(np.nan,df[df.loc[:,'name']==27]['ex_showroom_price'].median())
df_median=df.copy()
for var in df['name'].unique():
    df_median.update(df[df.loc[:,'name']==var]['ex_showroom_price'].replace(np.nan,df[df.loc[:,'name']==var]['ex_showroom_price'].median()))
df_median.isnull().sum()
df_median
mean=df_median['ex_showroom_price'].mean()
mean
df_median['ex_showroom_price'].isnull().sum()
df_median['ex_showroom_price']=df_median['ex_showroom_price'].fillna(mean)
df_median.isnull().sum()
df=df_median
df.isnull().sum()
df.head()
df['seller_type'].unique()
seller_type_map={'Individual':1, 'Dealer':2}
df['seller_type']=df['seller_type'].map(seller_type_map)
df.head()
df['owner'].unique()
owner_map={'1st owner':1, '2nd owner':2, '3rd owner':3, '4th owner':4}
df['owner']=df['owner'].map(owner_map)
df.head()
df.info()
df['current_year']=2020
df.head()
df['Num_of Year']=df['current_year']-df['year']
df.head(2)
df.drop(['year','current_year'],axis=1,inplace=True)
df.head(2)
X=df.drop('selling_price',axis=1)
y=df['selling_price']
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,confusion_matrix,classification_report,accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
ms=MinMaxScaler()
X_train_ms=ms.fit_transform(X_train)
X_test_ms=ms.fit_transform(X_test)
X_train_ms=pd.DataFrame(X_train_ms,columns=X_train.columns)
X_test_ms=pd.DataFrame(X_test_ms,columns=X_test.columns)
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
def check_model(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print('r2_score:',r2_score(y_test,y_pred))
check_model(LinearRegression(),X_train,X_test,y_train,y_test)
check_model(RandomForestRegressor(),X_train,X_test,y_train,y_test)
check_model(DecisionTreeRegressor(),X_train,X_test,y_train,y_test)
check_model(KNeighborsRegressor(),X_train,X_test,y_train,y_test)
rf=RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print('r2_score:',r2_score(y_test,y_pred))
cv=cross_val_score(RandomForestRegressor(n_estimators=250),X_train,y_train,cv=5)
print(np.average(cv))
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
clf= GridSearchCV(RandomForestRegressor(), {
    'n_estimators': [100,150,200,250,300]},cv=5,return_train_score=False)
clf.fit(X_train,y_train)
clf
clf.cv_results_
clf.best_score_
clf.score(X_test,y_test)
df=pd.DataFrame(clf.cv_results_)
df
df[['param_n_estimators','params','mean_test_score']]
clf.best_params_
clf.best_score_
rf=RandomForestRegressor(n_estimators=250)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print('r2_score:',r2_score(y_test,y_pred))
print('MSE:',mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('MAE:',mean_absolute_error(y_test,y_pred))
cv=cross_val_score(RandomForestRegressor(n_estimators=250),X_train,y_train,cv=5)
print(np.average(cv))
rf.score(X_test,y_test)
rf.score(X_train,y_train)
# ### Save Model in Pickle & Joblib
import pickle
import joblib
pickle.dump(rf,open('rf_model.pkl','wb'))
joblib.dump(rf,'rf_model.joblib')
# #### load pickle Model
model=pickle.load(open('rf_model.pkl','rb'))
y_pred=model.predict(X_test)
model.score(X_train,y_train)
model.score(X_test,y_test)
# #### load joblib Model
model_jb=joblib.load('rf_model.joblib')
model_jb.score(X_train,y_train)
model_jb.score(X_test,y_test)
model.predict([[7,1,1,75000,79432.0,7]])
X_test.head(2)
sns.scatterplot(y_test,y_pred)
sns.distplot(y_test-y_pred)
plt.show()