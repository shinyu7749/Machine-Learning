#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install xgboost')


# In[1]:


### Source: model.py from https://github.com/CodingWZL/AI4LiS/blob/main/DOL/model

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from xgboost.sklearn import XGBRegressor
from sklearn.inspection import permutation_importance
import joblib


X = np.loadtxt("feature.txt")
#scaler = MinMaxScaler().fit(X)
#X = scaler.transform(X)
Y = np.loadtxt("energy.txt")
kfold = KFold(n_splits=5, shuffle=True)

score = []
def main():
    for i in range(1, 2):
        fold = 1
        for train, test in kfold.split(X, Y):
            reg_1 = KNeighborsRegressor(n_neighbors=2)
            reg_2 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, subsample=0.9, max_depth=5)
            reg_3 = XGBRegressor(n_estimators=300, learning_rate=0.06, subsample=0.9, max_depth=5)
            reg = VotingRegressor(estimators=[('KNN', reg_1), ('GBDT', reg_2), ('XGBoost', reg_3)], weights = [2, 2, 3])
            reg.fit(X[train], Y[train])

            # save models
            joblib.dump(reg, "model/"+str(fold)+".model")

            # calculate the mae
            mae = mean_absolute_error(Y[test], reg.predict(X[test]))
            score.append(mae)
            #mse = mean_squared_error(Y[test], reg.predict(X[test]))
            #accuracy = accuracy_score(Y[test], eclf.predict(X[test]))
            print(Y[test].T, reg.predict(X[test]).T)
            print(mae)

            # get the feature S6-importance by permutation S6-importance method
            result = permutation_importance(reg, X[train], Y[train], n_repeats=5, random_state=42)
            np.savetxt("importance/" + str(fold) + "_importance.txt", result.importances)
            fold = fold + 1


if __name__ == '__main__':
    main()


# In[6]:


### Basic visualization of trained result

import matplotlib.pyplot as plt

# === Plot true vs predicted values across all folds ===
# Collect predictions for all data points
y_true_all = []
y_pred_all = []

for train, test in kfold.split(X, Y):
    reg_1 = KNeighborsRegressor(n_neighbors=2)
    reg_2 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, subsample=0.9, max_depth=5)
    # remove XGB if you don't use it
    reg = VotingRegressor(estimators=[('KNN', reg_1), ('GBDT', reg_2)], weights=[1, 20])
    reg.fit(X[train], Y[train])

    y_true_all.extend(Y[test])
    y_pred_all.extend(reg.predict(X[test]))

# Convert to numpy arrays
y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# Scatter plot: True vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_true_all, y_pred_all, alpha=0.7, edgecolor='k')
plt.plot([y_true_all.min(), y_true_all.max()],
         [y_true_all.min(), y_true_all.max()],
         'r--', lw=2)  # perfect prediction line
plt.xlabel("True Y (E0 eV)")
plt.ylabel("Predicted Y (E0 eV)")
plt.title("True vs Predicted Energy")
plt.grid(True)
plt.tight_layout()
plt.savefig("true_vs_pred.png", dpi=300)
plt.show()


# In[ ]:




