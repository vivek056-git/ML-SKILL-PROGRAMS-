from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE=Path(__file__).resolve().parent.parent
df=pd.read_csv(BASE/'dataset'/'heart_disease.csv')
X=df.drop(columns=['target']); y=df.target
num=['age','trestbps','chol','thalach','oldpeak']; cat=['sex','cp','fbs','restecg','exang','slope','ca','thal']
pre=ColumnTransformer([('num',SimpleImputer(strategy='median'),num),
 ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',drop='first'))]),cat)])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
for name,est in [
 ('XGBoost',XGBClassifier(n_estimators=160,max_depth=4,learning_rate=.05,subsample=.85,colsample_bytree=.85,random_state=42,n_jobs=1,eval_metric='logloss')),
 ('LightGBM',LGBMClassifier(n_estimators=160,max_depth=5,num_leaves=20,learning_rate=.05,subsample=.85,colsample_bytree=.85,random_state=42,n_jobs=1,verbosity=-1))]:
    pipe=Pipeline([('preprocessor',pre),('model',est)]); pipe.fit(X_train,y_train)
    p=pipe.predict(X_test); pr=pipe.predict_proba(X_test)[:,1]
    print(name,'Accuracy=',accuracy_score(y_test,p),'ROC-AUC=',roc_auc_score(y_test,pr))
