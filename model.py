#Data Wrangling

import pandas as pd
import numpy as np

#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#Data visualization
import matplotlib.pyplot as plt # plotting library
from dataprep.eda import plot
import seaborn as sns # additional functionality and enhancements

#Data preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

#Regression models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor
from xgboost import XGBClassifier
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('C:\\Users\\chuan\\Downloads\\Maternal Health Risk Data Set.csv.xls')
print("\033[1mDataset of Martenal Health Data Set : \033[0m")
data

# Check the head of the dataset
print("\033[1mFirst 10 rows of the data : \033[0m")
data.head(10)

# Check the tail of the dataset
print("\033[1mLast 10 rows of the data : \033[0m")
data.tail(10)

print("\033[1mInformation of the data :\033[0m ")
data.info()

# Describe the data
# Count, mean, min, max, 25%, 50%, 75%, std
print("\033[1mDescription of Data : \033[0m")
data.describe().T

# Check dimensionally of the DataFrame
print("\033[1mShape of Data :\033[0m")
data.shape

# Plot pairwise relationships
# Pair of Target with all other features in dataset
sns.pairplot(data, hue='RiskLevel')
plt.show()

# Plot heatmap
# making visualization of patterns, to identify correlations between variable
plt.figure(figsize=(15,12))
sns.heatmap(data.corr(), annot=True)
plt.show()

# Counts of RiskLevel for each Age
plt.figure(figsize=(10, 6))  # Increase the figure size

sns.countplot(x='Age', hue='RiskLevel', data=data)
plt.title('Counts of RiskLevel for each Age')  # Add a title to the plot
plt.xlabel('Age')  # Set the x-axis label
plt.ylabel('Count')  # Set the y-axis label

# Adjust the appearance of the plot
sns.set(style="whitegrid")  # Set the plot style
sns.despine()  # Remove the top and right spines

plt.tight_layout()  # Adjust the spacing of the plot
plt.show()

correlation = data[['RiskLevel', 'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'HeartRate']].corr()
correlation

data["RiskLevel"].value_counts()
# Count the unique value of the RiskLevel
# Unique Value : low risk, mid risk, high risk

# Show Pie Chart and Bar Chart of RiskLevel
fig = plt.figure(figsize=(5, 5))

# Define custom colors for the pie chart
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

# Generate the pie chart with custom colors and shadow
plt.pie(data['RiskLevel'].value_counts(), labels=list(data['RiskLevel'].unique()), autopct='%1.1f%%',
        colors=colors, shadow=True)

# Add a title with a larger font size and bold text
plt.title('RiskLevel Distribution', fontsize=16, fontweight='bold')

# Add a legend with smaller font size and adjust its position
plt.legend(loc='upper right', fontsize=10)

# Adjust the appearance of the plot
plt.axis('equal')
plt.tight_layout()  # Improve spacing between elements

plt.show()

plt.figure(figsize=(6, 8))
sns.countplot(x='RiskLevel', data=data, palette='viridis')
plt.xlabel('Risk Level')
plt.ylabel('Count')
plt.title('Distribution of Risk Levels')
plt.xticks(rotation=0)
plt.show()

# replace the target variable from string to numeric
data['RiskLevel'] = data['RiskLevel'].replace({'low risk':0, 'mid risk':1, 'high risk':2})
data
# Risk Level Values are changed to 0(low risk), 1(mid risk) ,2(high risk)

# Data type has been changed from object to int64 (for RiskLevel)
data.info()

# Convert the BodyTemp (Body Temperature) from F (Fahrenheit) to C (Celcius)
data['BodyTemp_C'] = (data['BodyTemp'] - 32) * 5/9
data['BodyTemp'] = data['BodyTemp_C']
data = data.drop(columns=['BodyTemp_C']) # drop the extra columns to prevent duplicate data
data

data.hist()

# Check correlation between all feature after conversion of target variable RiskLevel to numeric
plt.figure(figsize=(15,12))
sns.heatmap(data.corr(), annot=True)
plt.show()

# using dataprep class to show Statistics, Histogram, KDE plot, Normal Q-Q plot, Box Plot, Value Tabale
plot(data,'Age')
# We have data of age ranges from 10 years old to 70 years old

plot(data, 'BS')
# We have the BS ranges from 6 to 19
# Value after 8 consider high risk, they are valid and not consider outlier

plot(data, 'BodyTemp')
# Body Temp range of 36.88 to 39

plot(data,'DiastolicBP')
# The range of DiastolicBP is from 49 to 100

plot(data,'HeartRate')

import seaborn as sns
import matplotlib.pyplot as plt

sns.lineplot(x='DiastolicBP', y='SystolicBP', hue='RiskLevel', data=data, palette=['green', 'orange', 'red'])
plt.show()
# indicates when DBP and SBP is high then Risk is high

# Line Charts to provide the relantionship of SBP and DBP for different Risk Level
grid = sns.FacetGrid(data, col="RiskLevel", hue="RiskLevel", col_wrap=3, palette=['green', 'orange', 'red'])
grid.map(sns.lineplot, "DiastolicBP", "SystolicBP")
grid.add_legend()
plt.show()

# below scatter plot indicates total count of high risklevel in BS is high compared to low RiskLevel
# So BS is impactful factor for high risk
sns.scatterplot(x='RiskLevel', y='BS', hue='RiskLevel', data=data, palette=['green', 'orange', 'red'])
plt.show()

# As BS increases, Risk also increases significantly in all ages and vice versa
sns.lineplot(x='Age', y='BS', hue='RiskLevel', data=data, palette=['green', 'orange', 'red'])
plt.show()

# Count Missing Value
print("\033[1mMissing values by Column : \033[0m")
print("-"*30)
print(data.isna().sum())
print("-"*30)
print("Total Missing Values: ",data.isna().sum().sum())

# No missing values

# Split data into feature and target
X = data.drop('RiskLevel',axis = 1)
y = data['RiskLevel']
print("X Shape : ", X.shape)
print("y Shape : ", y.shape)

normal = MinMaxScaler()
standard = StandardScaler()

normalised_features = normal.fit_transform(X)
normalised_data = pd.DataFrame(normalised_features, index = X.index, columns = X.columns)

standardised_features = standard.fit_transform(X)
standardised_data = pd.DataFrame(standardised_features, index = X.index, columns = X.columns)

# Create subplots
fig, ax = plt.subplots(1, 3, figsize = (21, 5))

# Original
sns.boxplot(x = 'variable', y = 'value', data = pd.melt(data[X.columns]), ax = ax[0])
ax[0].set_title('Original')

# MinMaxScaler
sns.boxplot(x = 'variable', y = 'value', data = pd.melt(normalised_data[X.columns]), ax = ax[1])
ax[1].set_title('MinMaxScaler')

# StandardScaler
sns.boxplot(x = 'variable', y = 'value', data = pd.melt(standardised_data[X.columns]), ax = ax[2])
ax[2].set_title('StandardScaler')

knn = KNeighborsRegressor()
svr = SVR()
tree = DecisionTreeRegressor(max_depth = 10, random_state = 42)
xgb = XGBRegressor()
catb = CatBoostRegressor()
linear = LinearRegression()
sdg = SGDRegressor()
rfr = RandomForestRegressor(max_depth=2, random_state=0)
gb = GradientBoostingRegressor(random_state=0)
br = BayesianRidge()

scalers = [normal, standard]

# split the data into testing set and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

knn_rmse = []

# Without feature scaling
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
knn_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using KNN
for scaler in scalers:
    pipe = make_pipeline(scaler, knn)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    knn_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
knn_df = pd.DataFrame({'Root Mean Squared Error': knn_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
knn_df

svr_rmse = []

# Without feature scaling
svr.fit(X_train, y_train)
pred = svr.predict(X_test)
svr_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using SVR
for scaler in scalers:
    pipe = make_pipeline(scaler, svr)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    svr_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
svr_df = pd.DataFrame({'Root Mean Squared Error': svr_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
svr_df

xgb_rmse = []

# Without feature scaling
xgb.fit(X_train, y_train)
pred = xgb.predict(X_test)
xgb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using XGB
for scaler in scalers:
    pipe = make_pipeline(scaler, xgb)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    xgb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
xgb_df = pd.DataFrame({'Root Mean Squared Error': xgb_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
xgb_df

catb_rmse = []

# Without feature scaling
catb.fit(X_train, y_train)
pred = catb.predict(X_test)
catb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using CatBoost
for scaler in scalers:
    pipe = make_pipeline(scaler, catb)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    catb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
catb_df = pd.DataFrame({'Root Mean Squared Error': catb_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
catb_df

linear_rmse = []

# Without feature scaling
linear.fit(X_train, y_train)
pred = linear.predict(X_test)
linear_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using linear
for scaler in scalers:
    pipe = make_pipeline(scaler, linear)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    linear_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
linear_df = pd.DataFrame({'Root Mean Squared Error': linear_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
linear_df

sdg_rmse = []

# Without feature scaling
sdg.fit(X_train, y_train)
pred = sdg.predict(X_test)
sdg_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using SDG
for scaler in scalers:
    pipe = make_pipeline(scaler, sdg)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    sdg_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
sdg_df = pd.DataFrame({'Root Mean Squared Error': sdg_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
sdg_df

rfr_rmse = []

# Without feature scaling
rfr.fit(X_train, y_train)
pred = rfr.predict(X_test)
rfr_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using RandomForest
for scaler in scalers:
    pipe = make_pipeline(scaler, rfr)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rfr_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
rfr_df = pd.DataFrame({'Root Mean Squared Error': rfr_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
rfr_df

gb_rmse = []

# Without feature scaling
gb.fit(X_train, y_train)
pred = gb.predict(X_test)
gb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using GradientBoosting
for scaler in scalers:
    pipe = make_pipeline(scaler, gb)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    gb_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
gb_df = pd.DataFrame({'Root Mean Squared Error': gb_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
gb_df

br_rmse = []

# Without feature scaling
br.fit(X_train, y_train)
pred = br.predict(X_test)
br_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Apply different scaling techniques and make predictions using BayesianRidge
for scaler in scalers:
    pipe = make_pipeline(scaler, br)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    br_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))

# Show results
br_df = pd.DataFrame({'Root Mean Squared Error': br_rmse}, index = ['Original', 'MinMaxScaler', 'StandardScaler'])
br_df

data=pd.DataFrame([
    ['KNN', 'MinMaxScaler', knn_rmse[1]], ['KNN', 'StandardScaler', knn_rmse[2]],
    ['SVR', 'MinMaxScaler', svr_rmse[1]], ['SVR', 'StandardScaler', svr_rmse[2]],
    ['XGBoost', 'MinMaxScaler', xgb_rmse[1]], ['XGBoost', 'StandardScaler',  xgb_rmse[2]],
    ['CatBoost', 'MinMaxScaler', catb_rmse[1]], ['CatBoost', 'StandardScaler',  catb_rmse[2]],
    ['Linear', 'MinMaxScaler', linear_rmse[1]], ['Linear', 'StandardScaler',  linear_rmse[2]],
    ['SGD', 'MinMaxScaler', sdg_rmse[1]], ['SGD', 'StandardScaler', sdg_rmse[2]],
    ['RF', 'MinMaxScaler', rfr_rmse[1]], ['RF', 'StandardScaler',  rfr_rmse[2]],
    ['GB', 'MinMaxScaler', gb_rmse[1]], ['GB', 'StandardScaler',  gb_rmse[2]],
    ['BR', 'MinMaxScaler', br_rmse[1]],  ['BR', 'StandardScaler',  br_rmse[2]]
], columns=['Models', 'Scalers', 'Root Mean Squared Error'])

data

normalised_features

# Splitting data into train and test data
#X_train, X_test, y_train, y_test = train_test_split(normalised_features, y, test_size = 0.30, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Normal scaling of training dataset
X_train = normal.fit_transform(X_train)
X_test = normal.transform(X_test)

from imblearn.over_sampling import RandomOverSampler
over_sampler = RandomOverSampler(random_state = 42)
X_res, y_res = over_sampler.fit_resample(X_train, y_train)
y_res.value_counts()

#The LogisticRegression class can be configured for multinomial logistic regression
#by setting the “multi_class” argument to “multinomial” and the “solver” argument to a solver
#that supports multinomial logistic regression, such as “lbfgs“.
lr = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')
lr.fit(X_train, y_train)
score_lr_without_Kfold = lr.score(X_test, y_test)
score_lr_without_Kfold

# one-vs-one (‘ovo’) is used for multi-class strategy.
svm = SVC(decision_function_shape='ovo')
svm.fit(X_train, y_train)
score_svm_without_Kfold = svm.score(X_test, y_test)
score_svm_without_Kfold

# n_estimators is a parameter for the number of trees in the forest, which is 40
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
score_rf_without_Kfold = rf.score(X_test, y_test)
score_rf_without_Kfold

# n_neighbors is a parameter for Number of neighbors, which is 6
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
score_knn_without_Kfold = knn.score(X_test, y_test)
score_knn_without_Kfold

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
score_xgb_without_Kfold = xgb.score(X_test, y_test)
score_xgb_without_Kfold

score_lr_with_Kfold_imbalance = cross_val_score(LogisticRegression(solver='lbfgs',multi_class='multinomial'),
                                                X_train, y_train, cv=3)
print("Evaluation metric scores for each fold : ",score_lr_with_Kfold_imbalance)
print("Avg :",np.average(score_lr_with_Kfold_imbalance))

score_svm_with_Kfold_imbalance = cross_val_score(SVC(decision_function_shape='ovo'), X_train, y_train, cv=3)
print("Evaluation metric scores for each fold : ",score_svm_with_Kfold_imbalance)
print("Avg :",np.average(score_svm_with_Kfold_imbalance))

score_rf_with_Kfold_imbalance = cross_val_score(RandomForestClassifier(n_estimators=40), X_train, y_train, cv=10)
print("Evaluation metric scores for each fold : ",score_rf_with_Kfold_imbalance)
print("Avg :",np.average(score_rf_with_Kfold_imbalance))

# Check cross val scores of KNeighborsClassifier with with K-fold as 10.
score_knn_with_Kfold_imbalance = cross_val_score(KNeighborsClassifier(n_neighbors=6), X_train, y_train, cv=10)
print("Evaluation metric scores for each fold : ",score_knn_with_Kfold_imbalance)
print("Avg :",np.average(score_knn_with_Kfold_imbalance))

# Check cross val scores of XGBClassifier with with K-fold as 3.
score_xgb_with_Kfold_imbalance = cross_val_score(XGBClassifier(), X_train, y_train, cv=3)
print("Evaluation metric scores for each fold : ",score_xgb_with_Kfold_imbalance)
print("Avg :",np.average(score_xgb_with_Kfold_imbalance))

# With imbalance dataset, score of RandomForestClassifier is high
# hence reverified with differnt estimators but n_estimators=40 gives good score
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),X_train, y_train, cv=10)
print("Avg Score for Estimators=5 and CV=10 :",np.average(scores1))
scores2 = cross_val_score(RandomForestClassifier(n_estimators=10), X_train, y_train, cv=10)
print("Avg Score for Estimators=10 and CV=10 :",np.average(scores1))
scores3 = cross_val_score(RandomForestClassifier(n_estimators=20),X_train, y_train, cv=10)
print("Avg Score for Estimators=20 and CV=10 :",np.average(scores1))
scores4 = cross_val_score(RandomForestClassifier(n_estimators=30), X_train, y_train, cv=10)
print("Avg Score for Estimators=30 and CV=10 :",np.average(scores1))

# cross validation scores with balance dataset
score_lr_with_Kfold_balance = cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_res, y_res, cv=3)
print(score_lr_with_Kfold_balance)
print("Avg :",np.average(score_lr_with_Kfold_balance))
score_svm_with_Kfold_balance = cross_val_score(SVC(gamma='auto'), X_res, y_res, cv=3)
print(score_svm_with_Kfold_balance)
print("Avg :",np.average(score_svm_with_Kfold_balance))
score_rf_with_Kfold_balance = cross_val_score(RandomForestClassifier(n_estimators=40),X_res, y_res, cv=10)
print(score_rf_with_Kfold_balance)
print("Avg :",np.average(score_rf_with_Kfold_balance))
score_knn_with_Kfold_balance = cross_val_score(KNeighborsClassifier(n_neighbors=6), X_res, y_res, cv=10)
print(score_knn_with_Kfold_balance)
print("Avg :",np.average(score_knn_with_Kfold_balance))
score_xgb_with_Kfold_balance = cross_val_score(XGBClassifier(), X_res, y_res, cv=10)
print(score_xgb_with_Kfold_balance)
print("Avg :",np.average(score_xgb_with_Kfold_balance))

# With balance dataset, score of RandomForestClassifier is high
# hence reverified with differnt estimators but n_estimators=40 gives good score
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),X_res, y_res, cv=10)
print("Avg Score for Estimators=5 and CV=10 :",np.average(scores1))
scores2 = cross_val_score(RandomForestClassifier(n_estimators=10),X_res, y_res, cv=10)
print("Avg Score for Estimators=10 and CV=10 :",np.average(scores1))
scores3 = cross_val_score(RandomForestClassifier(n_estimators=20),X_res, y_res, cv=10)
print("Avg Score for Estimators=20 and CV=10 :",np.average(scores1))
scores4 = cross_val_score(RandomForestClassifier(n_estimators=30),X_res, y_res, cv=10)
print("Avg Score for Estimators=30 and CV=10 :",np.average(scores1))

# Bar subplots for checking differnce between original, k-folded imbalanced and k-folded balanced data for differnt models
data = pd.DataFrame([['LogisticRegression', 'without_Kfold', score_lr_without_Kfold],
                   ['LogisticRegression', 'with_Kfold_imbalance', score_lr_with_Kfold_imbalance],
                   ['LogisticRegression', 'with_Kfold_balance', score_lr_with_Kfold_balance],
                   ['SVM', 'without_Kfold', score_svm_without_Kfold],
                   ['SVM', 'with_Kfold_imbalance', score_svm_with_Kfold_imbalance],
                   ['SVM', 'with_Kfold_balance', score_svm_with_Kfold_balance],
                   ['RandomForest', 'without_Kfold', score_rf_without_Kfold],
                   ['RandomForest', 'with_Kfold_imbalance', score_rf_with_Kfold_imbalance],
                   ['RandomForest', 'with_Kfold_balance', score_rf_with_Kfold_balance],
                   ['KNN', 'without_Kfold', score_knn_without_Kfold],
                   ['KNN', 'with_Kfold_imbalance', score_knn_with_Kfold_imbalance],
                   ['KNN', 'with_Kfold_balance', score_knn_with_Kfold_balance],
                   ['XGBoost', 'without_Kfold', score_xgb_without_Kfold],
                   ['XGBoost', 'with_Kfold_imbalance', score_xgb_with_Kfold_imbalance],
                   ['XGBoost', 'with_Kfold_balance', score_xgb_with_Kfold_balance]],
                  columns=['Models', 'Processes', 'Cross Validation Scores'])

data = data.explode('Cross Validation Scores')
data['Cross Validation Scores'] = data['Cross Validation Scores'].astype('float') * 100

# plot with seaborn barplot
plt.figure(figsize=(18, 8))
ax = sns.barplot(data=data, x ='Processes', y ='Cross Validation Scores', hue ='Models', ci = None)

for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3)

# Fitting balanced data into RandomForestClassifier
RF = RandomForestClassifier(criterion='gini')
RF.fit(X_res, y_res)
# Predicting unseen data with RandomForestClassifier
pred= RF.predict(X_test)
# Check mean_squared_error
print(np.sqrt(mean_squared_error(y_test, pred)))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))

# Fitting balanced data into KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_res, y_res)
# Predicting unseen data with KNeighborsClassifier
pred= knn.predict(X_test)
# Check mean_squared_error
print(np.sqrt(mean_squared_error(y_test, pred)))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))

xgb = XGBClassifier()
xgb.fit(X_res, y_res)
# Predicting unseen data with XGBClassifier
pred= xgb.predict(X_test)
# Check mean_squared_error
print(np.sqrt(mean_squared_error(y_test, pred)))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))

# Fitting balanced data into DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_res, y_res)
# Predicting unseen data with DecisionTreeClassifier
pred= model.predict(X_test)
# Check mean_squared_error
print(np.sqrt(mean_squared_error(y_test, pred)))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))

# Fitting balanced data into SVC
svm = SVC(decision_function_shape='ovo')
svm.fit(X_res, y_res)
# Predicting unseen data with SVC
pred = svm.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))

# Fitting balanced data into SVM RBF
svm_rbf = SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo')
svm_rbf.fit(X_res, y_res)
# Predicting unseen data with SVM RBF
pred = svm_rbf.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# Fitting balanced data into QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0, store_covariance=False, tol=0.0001)
qda.fit(X_res, y_res)
# Predicting unseen data with QuadraticDiscriminantAnalysis
pred = qda.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))

from sklearn.naive_bayes import GaussianNB
# Fitting balanced data into GaussianNB
gnb = GaussianNB()
gnb.fit(X_res, y_res)
# Predicting unseen data with GaussianNB
pred = gnb.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))

# Fitting balanced data into LogisticRegression
lr = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')
lr.fit(X_res, y_res)
# Predicting unseen data with LogisticRegression
pred = lr.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, pred)))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))

from sklearn.model_selection import GridSearchCV
# Initialising list of paramaters for selection of best params for XGBoost Model
param_grid = {
    "learning_rate": [0.5, 1, 3, 5],
    "reg_lambda": [0, 1, 5, 10, 20]
}

# Applying param_grid , k_fold as 3 and training the model
# Computations can be run in parallel by using the keyword n_jobs=-1
grid = GridSearchCV(xgb, param_grid, cv=3, n_jobs=-1)
grid.fit(X_res, y_res)

grid.best_params_

# Applying Best params to XGBoost Model
xgb = XGBClassifier(colsample_bytree= 1, gamma=0, learning_rate=1, max_depth=3, subsample=0.8, reg_lambda=1)
xgb.fit(X_res, y_res)
pred= xgb.predict(X_test)

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test,pred),annot=True)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, pred))

# Initialising list of paramaters for selection of best params for KNeighborsClassifier Model
# Applying param_grid and training the model
param_grid={'n_neighbors': [1,2,3,4,5,6,7,8]}
gridsearchcv = GridSearchCV(knn, param_grid)
gridsearchcv.fit(X_res, y_res)

gridsearchcv.best_params_

# Applying Best Params to KNeighborsClassifier Model
knn2 = KNeighborsClassifier(n_neighbors=1)
knn2.fit(X_res, y_res)
pred = knn2.predict(X_test)

cm = confusion_matrix(y_test, pred)
print("Confusion Matrix:")
print(cm)

report = classification_report(y_test, pred)
print("Classification Report:")
print(report)

import pandas as pd
data = pd.read_csv('Maternal Health Risk Data Set.csv')

X = data.drop('RiskLevel', axis = 1)
Y = data['RiskLevel']

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
fn = data.columns[0:6]
cn = data["RiskLevel"].unique().tolist()
dataTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
dataTree.fit(X,Y)
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize= (20,10), dpi=300)
tree.plot_tree(dataTree, feature_names= fn, class_names= cn, filled = True)
plt.show()

data.drop(columns="SystolicBP", axis=1, inplace=True)
data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

normal = MinMaxScaler()
X_train_features = normal.fit_transform(X_train)
X_test_features = normal.transform(X_test)

over_sampler = RandomOverSampler(random_state = 42)
X_res, y_res = over_sampler.fit_resample(X_train, y_train)
y_res.value_counts()


RF2= RandomForestClassifier(criterion='gini', max_depth=20, max_features='log2', n_estimators=50)
RF2.fit(X_res, y_res)
pred= RF2.predict(X_test)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(y_test, pred)

# Create a heatmap plot of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

from sklearn.metrics import classification_report

report = classification_report(y_test, pred)
print("Classification Report:")
print(report)

age = int(input("Enter Age: "))
systolicBP = int(input("Enter Systolic BP: "))
diastolicBP = int(input("Enter Diastolic BP: "))
bloodSugar = float(input("Enter BS: "))
bodyTemp = float(input("Enter Body Temperature(in F): "))
heartRate = int(input("Enter Heart Rate: "))

# Create a new data point based on user inputs
new_data = [[age, systolicBP, diastolicBP, bloodSugar, bodyTemp, heartRate]]
new_data_scaled = scaler.transform(new_data)

# Reshape the data to be 2-dimensional
new_data_scaled = np.reshape(new_data_scaled, (1, 6))

# Make a prediction using the trained model
prediction = xgb.predict(new_data_scaled)

# Print the prediction
print("Predicted Risk Level:", prediction)


























