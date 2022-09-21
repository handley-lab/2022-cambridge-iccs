#| # Example 1: Introduction to regression
#|
#| This notebook is designed to introduce the basic concepts of training models
#| using the numpy and scipy libraries alone.
#------------------------------------------------------------------------------
#| Import the key libraries and functions.
import numpy as np

#| Load the data.
data_x, data_y = np.loadtxt('example_1_data.csv').T

#| Plot the data
import matplotlib.pyplot as plt
plt.plot(data_x, data_y, '.');

#| Define our model function

def f(x, theta):
    i = np.arange(len(theta))
    x = np.array(x)
    return (theta * x[..., None]**i).sum(axis=-1)

#|- **Q1**: What does the `f` define?
#|- **Q2**: What does `None` slicing do?
#|- **Q3**: Why choose `...` rather than `:`?
#|- **Q4**: Why is `axis=-1` essential to the above?
#|- **(Optional exercise)**: Are you able to extend the above to `theta`?
#------------------------------------------------------------------------------

#| Now fit the model to the data by minimising a loss function
from scipy.optimize import minimize

def loss(theta):
    return ((f(data_x, theta) - data_y)**2).sum()

theta0 = np.zeros(4)
sol = minimize(loss, theta0)

#|- **Q5**: 
#| Plot the solution
theta = sol.x
plt.plot(data_x, data_y, '.')
x = np.linspace(-1,3,1001)
plt.plot(x, f(x, theta));

#| Consider models of increasing complexity
plt.plot(data_x, data_y, '.')
for n in [0, 1, 2, 3, 4]:
    sol = minimize(loss, np.zeros(n+1))
    theta = sol.x
    plt.plot(x, f(x, theta));
#| **Q6**: Where is the quadratic solution? (hint, try adding a legend)
#------------------------------------------------------------------------------
#| **Q7:** What's goes wrong with the following?
plt.plot(data_x, data_y, '.')
sol = minimize(loss, np.zeros(10))
theta = sol.x
plt.plot(x, f(x, theta));


#| **Q8:** Plot the loss as a function of `n`
#| **Q9:** How could we go about choosing n? Programme something if you have time
#------------------------------------------------------------------------------

#| This dataset was generated with the code below.
#| 
#| Try modifying the data and see how these observations change
np.random.seed(0)
x = np.random.uniform(-1,3,50)
y = 1 + (x-1)**3 -(x-1)
y += np.random.randn(len(y))
np.savetxt('example_1_data.csv', np.array([x, y]).T)
#| # BREAK
#| 
#| ***
#------------------------------------------------------------------------------
#| # Example 2: Machine Learning with scikit learn
#| 
#| https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data
#|
#| Can we predict the rain in London?
#------------------------------------------------------------------------------
#| ## Exploring the data and showcasing pandas
#| First import the data raw as downloaded from kaggle

import pandas as pd
data = pd.read_csv('london_weather.csv')
data.head()

#| We can also get some broad info, which is useful to show us how many non-null values there are

data.info()

#| The pandas describe method is useful for getting a quick overview of the stastistics of the data

data.describe()

#| Note that the counts don't agree -- this is due to missing data, mostly in the snow_depth
#| Let's drop them.

data.dropna(inplace=True)

#| We could also have also used an imputer to fill it with the median
#| **(Optional exercise)**
#|
#|`from sklearn.impute import SimpleImputer`
#------------------------------------------------------------------------------
#| The pandas hist methoda is a useful tool for showing an overview of the shape of the data (useful for eyeballing distributions & scales

data.hist();

#| Now look into correlations

corr_matrix = data.corr()
corr_matrix['sunshine'].sort_values(ascending=False)

#| Unsurprisingly sunshine correlates with temperatures and pressures, and
#| anticorrelates with rain and cloud cover
#------------------------------------------------------------------------------
#| We can also plot the full scatter matrix, which is much more involved,
#| showing the scatter plots of pairwise parameters. Note that down the
#| diagonal it shows the histograms as before

pd.plotting.scatter_matrix(data);

#| It's also relatively easy to define new variables in pandas, e.g. an integer detailing the month. Here we use the `to_datetime` functionality to convert to a column of datetimes, and the pandas `.dt.month` operation to extract the month.

data['month'] = pd.to_datetime(data.date, format='%Y%m%d').dt.month
corr_matrix = data.corr()
corr_matrix['precipitation'].sort_values(ascending=False)

#| Surprising -- quite weak correlations... but!

pd.plotting.scatter_matrix(data);

#| It's a non-linear relationship with month. Let's drop it for now
data.drop(columns='month', inplace=True)



#| ## Showcasing scikit-learn
#| Let's start by trying to predict the amount of temperature
y = data['sunshine']
X = data.drop(columns=['sunshine'])

#| Split into training and testing data using the scikit learn function
#|
#| **(Optional Exercise)**: Look into using sklearn.model_selection.StratifiedShuffleSplit

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#| Now we need to standardise the data by zero meaning and scaling to unit variance. Other options include `MaxMinScaler`, as well as custom transformations. Look into `sklearn.compose.CoumnTransformer`.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = StandardScaler()
pipeline.fit(X_train)
X_trans = pipeline.transform(X_train)

#| Could also run `X_trans = pipeline.fit_transform(X_train)`  to 'fit' the pipeline and and scale the data simultaneously
#|
#| Now let's train a model!
#|
#| Start with something embarrassingly simple -- linear regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_trans, y_train)

#| Done! Now let's test the predictions on the _training_ set (i.e. how well it has done in-sample):

from sklearn.metrics import mean_squared_error
import numpy as np
y_pred = lin_reg.predict(X_trans)
np.sqrt(mean_squared_error(y_train, y_pred))

#| Well that's only OK when compared to the spread in y itself:

y_train.std()

#| A scatter plot of predictions is also not great
import matplotlib.pyplot as plt
plt.scatter(y_train, y_pred);

#| We're clearly doing something though!
#| Let's try something different -- a decision tree

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_trans, y_train)
y_pred = tree_reg.predict(X_trans)
np.sqrt(mean_squared_error(y_train, y_pred))

#| Well that seem fishy...
plt.scatter(y_train, y_pred);

#| This is almost certainly an example of overfitting. At this point we would be tempted to look at the testing data, but it is better to hold that back until later.
#| Let's try some cross-validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, X_trans, y_train, cv=10, scoring="neg_mean_squared_error")
np.sqrt(-scores)

#| So out-of-sample apparently worse than the linear regression -- let's check for sure

scores = cross_val_score(lin_reg, X_trans, y_train, cv=10, scoring="neg_mean_squared_error")
np.sqrt(-scores)


#| Let's try a `RandomForestRegressor` --  tthis is effectively a stack of decision trees
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(X_trans, y_train)
y_pred = forest_reg.predict(X_trans)
np.sqrt(mean_squared_error(y_train, y_pred))

#| Seems reasonable.
plt.scatter(y_train, y_pred);

#| Out of sample?
scores = cross_val_score(forest_reg, X_trans, y_train, cv=10, scoring="neg_mean_squared_error")
np.sqrt(-scores)

#| So slightly better than the linear regression, and clearly overfit in sample
#------------------------------------------------------------------------------
#| To tune the hyperparameters we can run a grid search over the hyperparameters for the `RandomForestRegressor`

from sklearn.model_selection import GridSearchCV
param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(X_trans, y_train)

#| We can get the best params as an attribute
grid_search.best_params_

#| and the best model
grid_search.best_estimator_

#| As well as the total grid of results
for mean_score, params in zip(grid_search.cv_results_["mean_test_score"], grid_search.cv_results_["params"]):
    print(np.sqrt(-mean_score), params)

#| Also useful to look at which features are the most important
for feature, importance in zip(X_train.columns, grid_search.best_estimator_.feature_importances_):
    print(feature, importance)
#| global radiation and cloud cover seem to be the most important.



#| Now try it on the test set

final_model = grid_search.best_estimator_

X_trans = pipeline.transform(X_test)
y_pred = final_model.predict(X_trans)

plt.scatter(y_test, y_pred);

#| Let's compare to lineara regression

plt.scatter(y_test, lin_reg.predict(X_trans));
plt.scatter(y_test, tree_reg.predict(X_trans));


#| Interestingly it seems that the main reason the tree does better is dealing with those outliers

#| Exercises:
#| - **Q10** How do the above change if we try to predict rainfall? temperature?
#| - **Q11** Are any other of the out-of-the-box methods any better?
#| - **Q12** Do these results change if you add in any new features e.g. month/year?
#| - **Q13** Try this analysis out using other datasets from kaggle?
#| - **Q14** How do the results change if we remove those outliers?


#| # BREAK
#| 
#| ***
#------------------------------------------------------------------------------
#| # Example 3: Neural network training
#|
#| ## Keras TensorFlow sequential API
#| Let's specify an anatomy
from tensorflow import keras
model = keras.models.Sequential()

model.add(keras.layers.Input(X.shape[-1]))
model.add(keras.layers.Dense(30, activation="relu"))
model.add(keras.layers.Dense(30, activation="relu"))
model.add(keras.layers.Dense(30, activation="relu"))
model.add(keras.layers.Dense(1, activation="softplus"))

#| Alternatively we can do it in one go
#|```
#|model = keras.models.Sequential([
#|    keras.layers.Input(X.shape[-1]),
#|    keras.layers.Dense(30, activation="relu"),
#|    keras.layers.Dense(30, activation="relu"),
#|    keras.layers.Dense(1, activation="softplus") 
#|    ])
#|```

#| Inspect the model
model.summary()

model.summary()

#| Specify the model training procedures with 'compilation'
model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])

#| Split into validation and testing using sklearn's `test_train_split` again
X_trans_, X_valid, y_train_, y_valid = train_test_split(pipeline.transform(X_train), y_train, test_size=0.2, random_state=43)

#| Train!
history = model.fit(X_trans_, y_train_, epochs=50, validation_data=(X_valid, y_valid))

#| Plot the history
pd.DataFrame(history.history).plot()

#| Make predictions
plt.scatter(y_test, lin_reg.predict(pipeline.transform(X_test)));
plt.scatter(y_test, model.predict(pipeline.transform(X_test)));


#| ## Keras TensorFlow functional API
#| Now let's look at the functional API, implemented for a non-sequential 'wide and deep' network
input = keras.layers.Input(X.shape[-1])
hidden1 = keras.layers.Dense(30, activation="relu")(input)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input, hidden2])
output = keras.layers.Dense(1, activation="softplus")(concat)

model = keras.models.Model(inputs=[input], outputs=[output])

#| Now let's fit it
model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
history = model.fit(X_trans_, y_train_, epochs=50, validation_data=(X_valid, y_valid))
            
pd.DataFrame(history.history).plot()

#| Make predictions
plt.scatter(y_test, lin_reg.predict(pipeline.transform(X_test)));
plt.scatter(y_test, model.predict(pipeline.transform(X_test)));

#| ## Keras TensorFlow Subclssing API
#| This is useful for dynamic models. Here we implement the same 'wide and deep' network as above, but in principle this could change dynamically according to input
class WideAndDeepModel(keras.models.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.output_ = keras.layers.Dense(1, activation="softplus")

    def call(self, input):
        hidden1 = self.hidden1(input)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.Concatenate()([input, hidden2])
        output = self.output_(concat)
        return output

#| We define `model_` rather than `model` purely because we wish to return to the previous model earlier (subclassed models are more restrictive than functional or sequential models).
model_ = WideAndDeepModel()
model_.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
history = model_.fit(X_trans_, y_train_, epochs=50, validation_data=(X_valid, y_valid))


#| We can inspect any weights we choose
weights, biases = model.get_layer('dense_4').get_weights()
weights

#| Saving and restoring
model.save("my_keras_model.h5")


#| Tasks
#| - **Q15** Make a more systematic comparison of all the methods losses
#| - **Q15** Explore changing the neural network architectures/training mechanisms


#| ## PyTorch

#| Define a sequential model for now
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(X.shape[-1],30),
    torch.nn.ReLU(),
    torch.nn.Linear(30,30),
    torch.nn.ReLU(), 
    torch.nn.Linear(30,1),
    torch.nn.Softplus(), 
    )

#| Define a mean square error loss function
loss_fn = torch.nn.MSELoss()

#| Define a stochastic gradient descent optimiser with learning rate 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

#| (slightly verbose) mechanism for passing training data into a format that pytorch can understand. 
train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_trans_).float(), torch.tensor(y_train_.values).float()), batch_size=32)
valid_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_valid).float(), torch.tensor(y_valid.values).float()), batch_size=32)

#| Very neat package for progress bars
import tqdm


#| The meat of the code -- train! (will discuss)
pbar = tqdm.tqdm(np.arange(50))
for _ in pbar:

    sum_train_loss = 0
    for x_batch, y_batch in train_dataloader:
        pred = model(x_batch)[:, 0]
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        sum_train_loss += loss.item()

    sum_valid_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in valid_dataloader:
            pred = model(x_batch)[:, 0] 
            loss = loss_fn(pred, y_batch)
            sum_valid_loss += loss.item()

    pbar.set_description("Loss: (%.2f, %.2f)" % (sum_train_loss/len(train_dataloader),
                                                 sum_valid_loss/len(valid_dataloader)))

#| Plot the predictions
y_pred = model(torch.from_numpy(pipeline.transform(X_test)).float()).detach().numpy()
plt.scatter(y_test, y_pred);

#| Tasks
#| - **Q16** Compare and contrast the pytorch approach vs the keras approach. Which do you prefer?
