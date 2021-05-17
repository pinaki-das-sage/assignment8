import dash
import dash_core_components as dcc
import dash_html_components as html
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

import plotly.express as px
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn import metrics

dataList = pd.read_csv('data/Bengaluru_House_Data.csv')

# dataList.head()
# dataList.sort_values('price')
# dataList.shape

# check and drop null values
# dataList.isnull().sum()
dataList.dropna(inplace=True)

# dataList.isnull().sum()

# check the size column
dataList['size'].unique()

# lets extract the integer part from it

dataList['size'] = dataList['size'].apply(lambda x: str(x.split(' ')[0]))
dataList['isize'] = dataList['size'].apply(lambda x: int(x))
dataList.head()


# the sq ft column has some non numeric values, we will convert them
# https://stackoverflow.com/questions/44140489/get-non-numerical-rows-in-a-column-pandas-python/44140594
# dataList[pd.to_numeric(dataList['total_sqft'], errors='coerce').isnull()].head()
# dataList[pd.to_numeric(dataList['total_sqft'], errors='coerce').isnull()].shape


# cleanup the total_sqft column
def convert_sqft_to_num(x):
    tokens = x.split('-')
    # if the value is of format x - y, then we take an average
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2

    # in all other cases, we try to convert to float and add null if conversion is not possible
    try:
        return float(x)
    except:
        return np.nan


dataList['sqft'] = dataList['total_sqft'].apply(lambda x: convert_sqft_to_num(x))

# dataList.head()

# clean the null values
dataList.dropna(inplace=True)
# dataList.head()
# dataList.dtypes

# the current price is in 100000, lets change it to millions
dataList['price'] = dataList['price'] / 10

# add price per sq ft
dataList['price_per_sqft'] = dataList['price'] * 1000000 / dataList['sqft']
dataList.sort_values('price_per_sqft')

dataList['location'] = dataList['location'].apply(lambda x: x.strip())
dataList['location'].value_counts(ascending=False)

# remove areas with less than 10 entries
cleanedList = dataList.groupby('location').filter(lambda x: len(x) > 10)

# cleanedList.head()

# cleanedList.head()
# cleanedList.groupby('location').count()

# remove outliers manually
cleanedList = cleanedList[cleanedList['price_per_sqft'] < 25000]
cleanedList = cleanedList[cleanedList['sqft'] <= 4000]
# cleanedList = cleanedList[(cleanedList['size'] > 1) & (cleanedList['size'] < 5)]
cleanedList = cleanedList[(cleanedList['isize'] == 2) | (cleanedList['isize'] == 3)]

fig1 = px.scatter(cleanedList, x='price_per_sqft', y='sqft', trendline='ols', color='size')
fig1.update_layout(
    title='Price vs Square footage for 2 and 3 bedroom houses',
    showlegend=True)

# lets get only relevant columns
# cleanedList.head()

# @todo including 'location', throws an error on the scalar fit, need to investigate more
filteredList = cleanedList[['size', 'isize', 'price', 'sqft']]
# filteredList.head()

# standardize our variables

# instantiate the SKLearn class
std_scalar = StandardScaler()
# "fit" the scalar to our data & then transform the data to the new parameters
std_scalar.fit(filteredList)

# Now transform
transformedScalar = std_scalar.transform(filteredList)

# create a pandas dataframe from it
scaledList = pd.DataFrame(transformedScalar, columns=filteredList.columns)
# scaledList.head()

# scaledList.mean()
# scaledList.std()

x = scaledList.copy()
y = cleanedList['price_per_sqft'].copy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=12)

# Instantiate the SKlearn algorithm
mymodel = LinearRegression()
mymodel.fit(x_train, y_train)
y_preds = mymodel.predict(x_test)
[round(i, 2) for i in y_preds[:5]]

# Compare that to the actual prices
list(y_test[:5])

# evaluate
rmse = sqrt(metrics.mean_squared_error(y_test, y_preds))
# print(rmse)

avg_val = y_train.mean()
# print(avg_val)

comparison = np.full((len(y_test),), avg_val)
# print(comparison[:5])

sqrt(metrics.mean_squared_error(y_test, comparison))
r2 = metrics.r2_score(y_test, y_preds)
round(r2, 2)

# pickle the fitted scalar
filename = open('bengaluru.pkl', 'wb')
pickle.dump(mymodel, filename)
filename.close()

plotData = plotData = pd.concat([pd.DataFrame(y_preds)[0], pd.DataFrame(y_test.values)[0]], axis=1,
                                keys=["pred", "test"])

fig2 = px.scatter(plotData, x='pred', y='test', trendline='ols')
fig2.update_layout(
    title='Prediction versus actual values map',
    showlegend=True)

# import plotly.graph_objects as go
# fig2 = go.Figure()
# fig2 = fig2.add_trace(go.Scatter(x=y_preds, y=y_test))
# fig.show()

# Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "Linear regression - Train, test, split"

# Set up the layout
app.layout = html.Div(children=[
    html.H2('House price analysis for Bangalore'),
    html.Hr(),
    dcc.Graph(
        id='assignment7.1',
        figure=fig1
    ),
    html.Hr(),
    dcc.Graph(
        id='assignment7.2',
        figure=fig2
    ),
    html.A('Code on Github', href="https://github.com/pinaki-das-sage/assignment8"),
]
)

if __name__ == '__main__':
    app.run_server()
