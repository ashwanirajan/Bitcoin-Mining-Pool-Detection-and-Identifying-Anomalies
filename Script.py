import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest 
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix
#from sklearn.utils.fixes import signature
from sklearn.metrics import accuracy_score
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
    
data = pd.read_csv("miningdata.csv")


data.drop(labels=['stddev_output_idle_time','stddev_input_idle_time'], axis=1, inplace=True)

#get rid of non-numeric features
features = data.drop(labels=['is_miner','address'], axis=1)
target = data['is_miner'].values
indices = range(len(features))
com = features.corr()

#Train test split
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, target, indices,  test_size=0.4, random_state = 1)

#training the model
model = XGBClassifier()
model = XGBClassifier(learning_rate=0.1,n_estimators=100)
model.fit(X_train, y_train)

#predicting results
y_pred = model.predict(X_test) #

#accuracy
cnf_matrix = confusion_matrix(y_test, y_pred)
print("confusion matrix for miner classification\n")
print(cnf_matrix)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy for miner classification\n")
print(accuracy)

x_pos = np.arange(len(features.columns))
btc_importances = model.feature_importances_

inds = np.argsort(btc_importances)[::-1]
btc_importances = btc_importances[inds]
cols = features.columns[inds]
bar_width = .8

#how many features to plot?
n_features=20
x_pos = x_pos[:n_features][::-1]
btc_importances = btc_importances[:n_features]

#plot
plt.figure(figsize=(18,10))
plt.barh(x_pos, btc_importances, bar_width, label='BTC model')
plt.yticks(x_pos, cols, rotation=0, fontsize=12)
plt.xlabel('feature importance', fontsize=14)
plt.title('Mining Pool Detector', fontsize=20)
plt.savefig("feature_imp")
#plt.tight_layout()
#plt.show()




#Only the miners from test data

test_feature_data = data.loc[ indices_test , : ]
test_feature_data["predicted_miner"] = y_pred

miner_data = test_feature_data.loc[test_feature_data['predicted_miner'] == False]
target_data = miner_data['predicted_miner']
miner_data.drop(labels=['is_miner','address','predicted_miner'], axis=1, inplace=True)






percent=0.02

# train isolation forest
model2 =  IsolationForest(contamination=percent)
model2.fit(miner_data) 

#predict anomalies
y_pred2 = model2.fit_predict(miner_data)



############plots############

fig, ax = plt.subplots(nrows=2, ncols=2)
plt.tight_layout()




plt.subplot(2, 2, 1)
plt.scatter(miner_data['minimum_monthly_output'],miner_data['total_out_tx_count'], c = y_pred2)
plt.tick_params(axis='both', labelsize=15)
plt.ylabel('total output transactions count', fontsize = 17)
plt.xlabel('minimum_monthly_output' , fontsize = 17)

plt.subplot(2, 2, 2)
plt.scatter(miner_data['mean_tx_input_value'],miner_data['total_out_tx_count'], c = y_pred2)
plt.tick_params(axis='both', labelsize=15)
plt.ylabel('total output transactions count', fontsize = 17)
plt.xlabel('mean input transactions value' , fontsize = 17)


plt.subplot(2, 2, 3)
plt.scatter(miner_data['minimum_monthly_output'],miner_data['output_active_months'], c = y_pred2)
plt.tick_params(axis='both', labelsize=15)
plt.ylabel('total output acive months', fontsize = 17)
plt.xlabel('minimum_monthly_output' , fontsize = 17)

plt.subplot(2, 2, 4)
plt.scatter(miner_data['total_out_tx_count'],miner_data['total_out_tx_value'], c = y_pred2)
plt.tick_params(axis='both', labelsize=15)
plt.ylabel('total output transactions count', fontsize = 17)
plt.xlabel('total output transactions value' , fontsize = 17)


fig = plt.gcf()
fig.set_size_inches(14,12)
plt.savefig("anomaly.png")
#plt.show()