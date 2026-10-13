from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

BASE=Path(__file__).resolve().parent.parent
df=pd.read_csv(BASE/'dataset'/'heart_disease.csv')
X=df.drop(columns=['target']); y=df['target']
num=['age','trestbps','chol','thalach','oldpeak']
cat=['sex','cp','fbs','restecg','exang','slope','ca','thal']
pre=ColumnTransformer([
 ('num',SimpleImputer(strategy='median'),num),
 ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                  ('onehot',OneHotEncoder(handle_unknown='ignore',drop='first'))]),cat)
])
model=Pipeline([('preprocessor',pre),('classifier',RandomForestClassifier(
 n_estimators=300,max_depth=8,min_samples_leaf=3,class_weight='balanced',
 random_state=42,n_jobs=-1))])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
model.fit(X_train,y_train)
p=model.predict(X_test); proba=model.predict_proba(X_test)[:,1]
print('Accuracy:',accuracy_score(y_test,p))
print('Precision:',precision_score(y_test,p))
print('Recall:',recall_score(y_test,p))
print('F1:',f1_score(y_test,p))
print('ROC-AUC:',roc_auc_score(y_test,proba))
