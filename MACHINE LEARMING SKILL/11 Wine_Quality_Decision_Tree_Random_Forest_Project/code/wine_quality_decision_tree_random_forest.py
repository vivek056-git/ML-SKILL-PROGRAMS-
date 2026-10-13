from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
BASE=Path(__file__).resolve().parent.parent
df=pd.read_csv(BASE/'dataset'/'winequality_red.csv')
X=df.drop(columns=['quality']); y=df.quality
pre=ColumnTransformer([('num',SimpleImputer(strategy='median'),X.columns.tolist())])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
models={'Decision Tree':DecisionTreeClassifier(max_depth=5,min_samples_leaf=8,class_weight='balanced',random_state=42),
'Random Forest':RandomForestClassifier(n_estimators=250,max_depth=10,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=1)}
for name,est in models.items():
    pipe=Pipeline([('preprocessor',pre),('model',est)]); pipe.fit(X_train,y_train)
    p=pipe.predict(X_test); pr=pipe.predict_proba(X_test)
    print(name,'Accuracy=',accuracy_score(y_test,p),'Macro ROC-AUC=',roc_auc_score(y_test,pr,multi_class='ovr',average='macro'))
