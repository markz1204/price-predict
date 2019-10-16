#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats

#Load the training data
df_train=pd.read_csv('train.csv')

#Check the column headers
print(df_train.columns)

#Descriptive statistics summary
df_train['SalePrice'].describe()

#histogram
#sns.distplot(df_train['SalePrice'])

#correlation matrix
#corrmat = df_train.corr()
#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrmat, vmax=.8, square=True)

#check if this variable is not relevant to the sale price
#var = 'EnclosedPorch'
#data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
#data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

#check duplicated rows
df_train[df_train.duplicated()==True]

#check column data types
res = df_train.dtypes
print(res[res == np.dtype('int64')])
print(res[res == np.dtype('bool')])
print(res[res == np.dtype('object')])
print(res[res == np.dtype('float64')])

#standardize
print(df_train["LotConfig"].unique())

# feature scaling, only apply to numeric data
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(df_train[["GrLivArea","SalePrice"]])

#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

#df_train.isnull().sum().max()

#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

#histogram and normal probability plot
#sns.distplot(df_train['GrLivArea'], fit=norm)
#fig = plt.figure()
#res = stats.probplot(df_train['GrLivArea'], plot=plt)

df_train['SalePrice'] = np.log(df_train['SalePrice'])

#sns.distplot(df_train['SalePrice'], fit=norm)
#fig = plt.figure()
#res = stats.probplot(df_train['SalePrice'], plot=plt)

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

#sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
#fig = plt.figure()
#res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

#plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);
#plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);

#identify the outliers
fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(16, 4))
axes = np.ravel(axes)
col_name = ['OverallQual','YearBuilt', 'YearRemodAdd', '1stFlrSF', 'GrLivArea', 'FullBath','TotRmsAbvGrd','GarageArea']
#for i, c in zip(range(8), col_name):
#    df_train.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='r')

#delete outliers
print(df_train.shape)
df_train = df_train[df_train['GrLivArea'] < 8]
df_train = df_train[df_train['1stFlrSF'] < 2500]
df_train = df_train[df_train['GarageArea'] < 1000]

for i, c in zip(range(8), col_name):
    df_train.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='b')

#Now feature selection part
print(df_train.shape) #still 76 features remain

#check distribution of all the inputs
#df_train.hist(figsize=(20, 20), bins=20)
#plt.show()

#Based on the distribution results, check suspicious inputs
#3SsnPorch has too many zeros
#df_train['3SsnPorch'].describe()

#BedroomAbvGr is not normal distributed
#np.unique(df_train['BedroomAbvGr'].values)
#df_train.groupby('BedroomAbvGr').count()['Id']

#Two Basement bathroom variables, can be merged together
#df_train.groupby('BsmtFullBath').count()['Id']
#df_train.groupby('BsmtHalfBath').count()['Id']
#df_train['Bathroom']=df_train['BsmtFullBath']+ df_train['BsmtHalfBath']

#Four basement area variables. Three can be dropped
#df_train[['TotalBsmtSF', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF']].head()

# Three more porch related variables. We can merge them togher or just keep one.
#df_train[['OpenPorchSF', 'EnclosedPorch', 'ScreenPorch']].describe()

#Garage area and cars must be correlated. Use area or cars?
#df_train.corr()['GarageArea']['GarageCars']

#garage year built can be also dropped because we have a varaible: house year built.
#df_train.corr()['GarageYrBlt']['YearBuilt']

#KitchenAbvGr can be dropped as there are too many 1s
#df_train['KitchenAbvGr'].describe()

#Lot area has a small proportion houses which have large area, need to be filtered.
#sns.distplot(df_train['LotArea'], bins=100) #can filter Lot Area above 50000 to 50001

#Now let's check the correlation matrix
df_train.corr()['SalePrice'].sort_values()

#YearBuilt and YearRemodAdd seems correlated
#df_train.corr()['YearBuilt']['YearRemodAdd']

#Only select numeric variables (including SalePrice)
num_attrs = df_train.select_dtypes([np.int64, np.float64]).columns.values
df_train_num= df_train[num_attrs]

#Merge two bathroom variables
#df_train_num.loc['Bath']= df_train_num['BsmtFullBath'] + df_train_num['BsmtHalfBath']

#Remove the above variables, removed 'GarageYrBlt'
df_train_num.corr()['GarageCars']['GarageArea']
df_train_num.corr()['YearBuilt']['YearRemodAdd']

df_train_num=df_train_num.drop(['Id','GarageCars'],axis=1)

""""
#Get the correlation matrix
corr = df_train_num.corr()
corr = corr.applymap(lambda x : 1 if x > 0.5 else -1 if x < -0.5 else 0)
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, vmax=1, center=0,vmin=-1 ,  square=True, linewidths=.005)
plt.show()
"""

#Identify two correlated variables
#df_train_num=df_train_num.drop(['TotRmsAbvGrd','1stFlrSF'],axis=1)

#select correlation >0.5
df_train_num=df_train_num[df_train_num.columns[df_train_num.corr()['SalePrice']>0.5]]
#df_train_num.columns

cols = ['OverallQual','YearBuilt', 'YearRemodAdd', '1stFlrSF', 'GrLivArea', 'FullBath','TotRmsAbvGrd','GarageArea']

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

x = df_train_num[cols].values
y = df_train_num['SalePrice'].values
X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = RandomForestRegressor(n_estimators=400)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
reg = clf

#Check the training dataset prediction performance
from sklearn import metrics
#Mean Absolute Error
print('MAE:', metrics.mean_absolute_error(y_test, preds))
#Mean Squared Error
print('MSE:', metrics.mean_squared_error(y_test, preds))
#Root Mean Squared Error
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))

#Plot the predictions and actuals
#plt.scatter(y_test,preds)

""""
#Build the model
from sklearn import linear_model
reg = linear_model.LinearRegression()

#Split the input and output
df_train_num_x=df_train_num.drop('SalePrice',axis=1)
df_train_num_y=df_train_num['SalePrice']

#Train the model
reg.fit(df_train_num_x, df_train_num_y)

#Check the model coefficients
print('Coefficients: \n', reg.coef_)

#Get the prediction based on the training dataset
preds = reg.predict(df_train_num_x)

#Check the training dataset prediction performance
from sklearn import metrics
#Mean Absolute Error
print('MAE:', metrics.mean_absolute_error(df_train_num_y, preds))
#Mean Squared Error
print('MSE:', metrics.mean_squared_error(df_train_num_y, preds))
#Root Mean Squared Error
print('RMSE:', np.sqrt(metrics.mean_squared_error(df_train_num_y, preds)))

#Plot the predictions and actuals
plt.scatter(df_train_num_y,preds)

#Check the error
sns.distplot((df_train_num_y-preds),bins=35)
"""""

#Load the test data
df_test=pd.read_csv('test.csv')
df_test_num= df_test[['OverallQual','YearBuilt', 'YearRemodAdd', '1stFlrSF', 'GrLivArea', 'FullBath','TotRmsAbvGrd','GarageArea','Id']]

#IMPORTANT: All the feature engineering & data cleaning steps we have done to the training variables, we have to do the same for the test dataset!!
#Before we can feed the data into our model, we have to check missing values again. Otherwise the code will give you an error.
#df_test_num.loc['TotalBsmtSF']=df_test_num['TotalBsmtSF'].fillna(np.mean(df_test_num['TotalBsmtSF']))

df_test_num.loc[df_test_num['GarageArea'].isnull(), 'GarageArea'] = np.mean(df_test_num['GarageArea']).round(2)
df_test_num.loc[df_test_num['GrLivArea'].notnull(),'GrLivArea']=np.log(df_test_num['GrLivArea'])

#Predict the results for test dataset
submit= pd.DataFrame()
submit['Id'] = df_test_num.Id
#select features
preds_out = np.exp(pd.DataFrame(reg.predict(df_test_num[cols].values)))

submit['SalePrice'] = preds_out
#final submission
submit.to_csv('test_submit.csv', index=False)

#Check output
#check yearly alignment
df_train['preds'] = preds_out
df_yearly=df_train[['SalePrice','preds','YearBuilt']].groupby('YearBuilt').mean()
sns.lineplot(data=df_yearly)

#check Rates the overall material and finish of the house
df_yearly1=df_train[['SalePrice','preds','OverallQual']].groupby('OverallQual').mean()
sns.lineplot(data=df_yearly1)

#check Rates the overall condition of the house
df_yearly2=df_train[['SalePrice','preds','OverallCond']].groupby('OverallCond').mean()
sns.lineplot(data=df_yearly2)

#check Bedrooms
df_yearly3=df_train[['SalePrice','preds','BedroomAbvGr']].groupby('BedroomAbvGr').mean()
sns.lineplot(data=df_yearly3)
