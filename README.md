# Project 6 - D(St)reams of Anomalies
The real world does not slow down for bad data
1.  Set up a data science project structure in a new git repository in your GitHub account
2.  Download the benchmark data set from
https://www.kaggle.com/boltzmannbrain/nab  or
https://github.com/numenta/NAB/tree/master/data
3.  Load the  one  of the data set into panda data frames
4.  Formulate one or two ideas on how feature engineering would help the data set to establish additional value using exploratory data analysis
5.  Build one or more anomaly detection models to determine the  anomalies  using the other columns as features
6.  Document your process and results
7.  Commit your notebook, source code, visualizations and other supporting files to the git repository in GitHub

### Feature engineering
There are very few types of data in this data set, so it is difficult for us to combine them again. A good solution is to subdivide them to increase the diversity and difference of the data. For this, we split the time data into specific year, month, day, hour and minute. In this way, you can not only explore the timeliness of the data, but also observe their periodicity. For the only data, this is time-divisional.
