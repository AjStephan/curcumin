
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys, AtomPairs
from rdkit.Chem import PandasTools
from rdkit import RDConfig
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Importing necessary libraries
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif , f_regression

#booster
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.pipeline import Pipeline
import time

start = time.time()

data = pd.read_csv('/Users/stephan/Documents/Machine_learning_France/curcumin/codes/curcumin/HOP.csv')
print('data read')
# 55000 data = pd.read_csv('https://raw.githubusercontent.com/AjStephan/havard-smile-opv/main/Non-fullerene%20small-molecules%20acceptors.csv')
print(data.dtypes)
print(data.isna().sum())
data

data = data.dropna()
print(data.isna().sum())

print(data.columns)

ez33 = 'InChI=1S/C23H23BF2O6/c1-27-20-11-7-16(13-22(20)29-3)5-9-18-15-19(32-24(25,26)31-18)10-6-17-8-12-21(28-2)23(14-17)30-4/h5-15H,1-4H3/b9-5+,10-6+'
ad1022 = 'InChI=1S/C23H23BF2O6/c1-27-18-9-11-22(29-3)16(13-18)5-7-20-15-21(32-24(25,26)31-20)8-6-17-14-19(28-2)10-12-23(17)30-4/h5-15H,1-4H3/b7-5+,8-6+'
ad11 = 'InChI=1S/C21H19BF2O6/c1-27-20-11-14(5-9-18(20)25)3-7-16-13-17(30-22(23,24)29-16)8-4-15-6-10-19(26)21(12-15)28-2/h3-13,25-26H,1-2H3/b7-3+,8-4+'
ad10 = 'InChI=1S/C23H23BF2O6/c1-27-18-9-5-16(22(14-18)29-3)7-11-20-13-21(32-24(25,26)31-20)12-8-17-6-10-19(28-2)15-23(17)30-4/h5-15H,1-4H3/b11-7+,12-8+'
ad9 = 'InChI=1S/C21H19BF2O4/c1-25-20-9-5-3-7-16(20)11-13-18-15-19(28-22(23,24)27-18)14-12-17-8-4-6-10-21(17)26-2/h3-15H,1-2H3/b13-11+,14-12+'
ADMeO3 ='InChI=1S/C25H27BF2O8/c1-29-18-12-22(31-3)20(23(13-18)32-4)9-7-16-11-17(36-26(27,28)35-16)8-10-21-24(33-5)14-19(30-2)15-25(21)34-6/h7-15H,1-6H3/b9-7+,10-8+'
AD14 = 'InChI=1S/C23H25BF2N2O2/c1-27(2)20-11-5-18(6-12-20)9-15-22-17-23(30-24(25,26)29-22)16-10-19-7-13-21(14-8-19)28(3)4/h5-17H,1-4H3/b15-9+,16-10+'
substitute_listinchi = (ADMeO3,ad9,ad10,ad11,ad1022,ez33,AD14)
molecules_without = [Chem.MolFromInchi(smiles) for smiles in substitute_listinchi]
curcumin = [Chem.RDKFingerprint(mol) for mol in molecules_without]
curcumin = np.array(curcumin)


print('curcumin loaded ')
# Generate PubChem fingerprints
# Generate Morgan fingerprints
mfull = [Chem.MolFromSmiles(x) for x in data['smiles']]
pubchem_fps = [Chem.RDKFingerprint(mol) for mol in mfull]
pubchem_fps = np.array(pubchem_fps)


print('smile  loaded ')
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(pubchem_fps, data['Gap'], test_size=0.2, random_state=42)

# Define a list of algorithms
algorithms = [
    ('Polynomial Regression', PolynomialFeatures(degree=2)),
    ('Support Vector Regression', SVR(kernel='linear', C=1.0)),
    ('Decision Tree', DecisionTreeRegressor(max_depth=5)),
    ('Random Forest', RandomForestRegressor(n_estimators=100)),
    ('Gaussian Process', GaussianProcessRegressor(alpha=0.1))
]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaledcurcumin = scaler.transform(curcumin)
# Create customized regressors
gbm = GradientBoostingRegressor(n_estimators=100, max_depth=3)
xgb = XGBRegressor(learning_rate=0.1)
lgbm = LGBMRegressor(num_leaves=31)
catboost = CatBoostRegressor(iterations=100)
histgbm = HistGradientBoostingRegressor(l2_regularization=0.1)

# Fit and predict using the customized regressors
gbm.fit(X_train_scaled, y_train)
y_pred_gbm = gbm.predict(X_test_scaled)
y_pred_gbmcur = gbm.predict(curcumin)

xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)
y_pred_xgbcur = xgb.predict(curcumin)

lgbm.fit(X_train_scaled, y_train)
y_pred_lgbm = lgbm.predict(X_test_scaled)
y_pred_lgbmcur = lgbm.predict(curcumin)

catboost.fit(X_train_scaled, y_train)
y_pred_catboost = catboost.predict(X_test_scaled)
y_pred_catboostcur = catboost.predict(curcumin)


histgbm.fit(X_train_scaled, y_train)
y_pred_histgbm = histgbm.predict(X_test_scaled)
y_pred_histgbmcur = histgbm.predict(curcumin)
print(y_pred_histgbmcur)
print(y_pred_lgbmcur)
print(y_pred_xgbcur)
print(y_pred_gbmcur)



# Calculate evaluation metrics
mse_gbm = mean_squared_error(y_test, y_pred_gbm)
mae_gbm = mean_absolute_error(y_test, y_pred_gbm)
r2_gbm = r2_score(y_test, y_pred_gbm)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
r2_lgbm = r2_score(y_test, y_pred_lgbm)

mse_catboost = mean_squared_error(y_test, y_pred_catboost)
mae_catboost = mean_absolute_error(y_test, y_pred_catboost)
r2_catboost = r2_score(y_test, y_pred_catboost)

mse_histgbm = mean_squared_error(y_test, y_pred_histgbm)
mae_histgbm = mean_absolute_error(y_test, y_pred_histgbm)
r2_histgbm = r2_score(y_test, y_pred_histgbm)

print("Gradient Boosting Regressor:")
print(f"Mean Squared Error: {mse_gbm}")
print(f"Mean Absolute Error: {mae_gbm}")
print(f"R-squared: {r2_gbm}")

print()

print("XGBoost Regressor:")
print(f"Mean Squared Error: {mse_xgb}")
print(f"Mean Absolute Error: {mae_xgb}")
print(f"R-squared: {r2_xgb}")
print()

print("LightGBM Regressor:")
print(f"Mean Squared Error: {mse_lgbm}")
print(f"Mean Absolute Error: {mae_lgbm}")
print(f"R-squared: {r2_lgbm}")
print()

print("CatBoost Regressor:")
print(f"Mean Squared Error: {mse_catboost}")
print(f"Mean Absolute Error: {mae_catboost}")
print(f"R-squared: {r2_catboost}")
print()

print("HistGradientBoosting Regressor:")
print(f"Mean Squared Error: {mse_histgbm}")
print(f"Mean Absolute Error: {mae_histgbm}")
print(f"R-squared: {r2_histgbm}")
# Train and evaluate each algorithm
for name, algorithm in algorithms:
    if name == 'Support Vector Regression':
        algorithm.set_params(C=1.0, epsilon=0.1)  # Set SVR hyperparameters as needed
    if name == 'Polynomial Regression':
        # Preprocess the data for polynomial regression
        poly_features = algorithm
        X_train_transformed = poly_features.fit_transform(X_train)
        X_test_transformed = poly_features.transform(X_test)
        algorithm = LinearRegression()  # Use linear regression as the model

    else:
        # Preprocess the data using standard scaler

        X_train_transformed = scaler.fit_transform(X_train)
        X_test_transformed = scaler.transform(X_test)

    # Train the algorithm
    algorithm.fit(X_train_transformed, y_train)

    # Evaluate the algorithm
    y_pred = algorithm.predict(X_test_transformed)
    y_predcur = algorithm.predict(scaledcurcumin)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    print(f"Algorithm: {name}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")
    print(f"explained_variance_score: {evs}")
    print(y_predcur)
end = time.time()

print("time =  ", end-start )


