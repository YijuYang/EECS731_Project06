import numpy as np
import pandas

from pandas           import read_csv
from matplotlib       import pyplot
from matplotlib.dates import DateFormatter, WeekdayLocator, drange, RRuleLocator, YEARLY, rrulewrapper, MonthLocator
from sklearn.pipeline import make_pipeline
from sklearn          import preprocessing
from sklearn.metrics  import explained_variance_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

dataset = pandas.read_csv('data/realTweets/realTweets/Twitter_volume_AAPL.csv', parse_dates=['timestamp'], infer_datetime_format=True)
print(dataset.head(10))

dataset.plot(x='timestamp', figsize=(10,5))
pyplot.show()

data = pandas.DataFrame()
data['value'] = dataset.value
data['time'] = dataset.timestamp.apply(lambda t:t.time())
data['date'] = dataset.timestamp.apply(lambda t:t.date())
print(data.head(10))


data_pre = pandas.DataFrame()
data_pre['value'] = dataset.value
data_pre['year'] = dataset.timestamp.apply(lambda t:t.year)
data_pre['month'] = dataset.timestamp.apply(lambda t:t.month)
data_pre['day'] = dataset.timestamp.apply(lambda t:t.day)
data_pre['hour'] = dataset.timestamp.apply(lambda t:t.hour)
data_pre['minute'] = dataset.timestamp.apply(lambda t:t.minute)
print(data_pre.head(10))



IF = IsolationForest(max_samples=100)
IF.fit(data_pre)

# predictions
y_pred_train = IF.predict(data_pre)

data_pre['timestamp_int'] = dataset.timestamp.apply(lambda t:t.timestamp())
data_pre['timestamp'] = dataset.timestamp
#data_pre['timestamp'] = data_ori['timestamp'].timestamp()
data_pre['normal0'] = y_pred_train
print(data_pre.head(10))

data_nor = data_pre[data_pre.normal0 == 1]
data_abn = data_pre[data_pre.normal0 == -1]

ax = pyplot.gca()
data_nor.plot(x='timestamp_int', y='value', ax=ax, color='blue')
data_abn.plot(kind='scatter', x='timestamp_int', y='value', ax = ax, marker='x', color='r')

pyplot.show()



data_pre = data_pre.drop(['timestamp'], axis=1)
data_pre = data_pre.drop(['timestamp_int'], axis=1)

RF = RandomForestClassifier()
RF.fit(data_pre,y_pred_train)

# predictions
y_pred_train = RF.predict(data_pre)

data_pre['timestamp_int'] = dataset.timestamp.apply(lambda t:t.timestamp())
data_pre['timestamp'] = dataset.timestamp
#data_pre['timestamp'] = data_ori['timestamp'].timestamp()
data_pre['normal1'] = y_pred_train
print(data_pre.head(10))

data_nor = data_pre[data_pre.normal1 == 1]
data_abn = data_pre[data_pre.normal1 == -1]

ax = pyplot.gca()
data_nor.plot(x='timestamp_int', y='value', ax=ax,color='blue',marker='o')
data_abn.plot(kind='scatter', x='timestamp_int', y='value', ax = ax, marker='x', color='r')

pyplot.show()

data_pre = data_pre.drop(['timestamp'], axis=1)

min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data_pre)
# train one class SVM
model =  OneClassSVM(nu=0.95 * 0.01)
data = pandas.DataFrame(np_scaled)
model.fit(data)

data_pre['normal2'] = pandas.Series(model.predict(data))
data_pre['normal2'] = data_pre['normal2'].map( {1: 0, -1: 1} )
print(data_pre['normal2'].value_counts())

fig, ax = pyplot.subplots()

a = data_pre.loc[data_pre['normal2'] == 1, ['timestamp_int', 'value']]

ax.plot(data_pre['timestamp_int'], data_pre['value'], color='blue',marker='.',linestyle=' ')
ax.scatter(a['timestamp_int'], a['value'], color='red',marker='x')
pyplot.show()
