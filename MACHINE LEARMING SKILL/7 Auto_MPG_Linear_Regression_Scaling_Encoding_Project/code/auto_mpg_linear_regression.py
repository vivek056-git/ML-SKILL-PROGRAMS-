from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE=Path(__file__).resolve().parent.parent
df=pd.read_csv(BASE/'dataset'/'auto_mpg.csv')
num=['cylinders','displacement','horsepower','weight','acceleration','model_year']
cat=['origin','make']
X=df[num+cat]; y=df['mpg']
pre=ColumnTransformer([
 ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),
 ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',drop='first'))]),cat)
])
model=Pipeline([('preprocessor',pre),('regressor',LinearRegression())])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
model.fit(X_train,y_train)
p=model.predict(X_test)
print('MAE:',mean_absolute_error(y_test,p))
print('RMSE:',mean_squared_error(y_test,p)**0.5)
print('R2:',r2_score(y_test,p))
