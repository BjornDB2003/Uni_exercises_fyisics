# Bjorn De Busschere, Bjorn.DeBusschere@student.uantwerpen.be, 3/04/2024
# =============================================================================
# Imports
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import torch
import random
import math
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from CombinedAttributesAdder import CombinedAttributesAdder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# Functions
# =============================================================================


def Cleaning_NaN(drop, df_cal):
    """
    Function for cleaning op NaN value fields in the DataSet California Housing
    Dataset. Options are: Dropping rows with NaN values, dropping columns with
    NaN values, filling in the NaN values with the median, or removing a
    specified attribute.

    parameters:
        - drop should be 'row', 'column', 'imputer' or the name of the column
          to drop in a string (e.g. ['total_bedrooms'])
        - df_cal should be the california housing DataFrame

    returns:
        - cleaned dataframe
    """
    # Initial checks
    if not isinstance(df_cal, pd.DataFrame):
        # Raising error
        raise TypeError('df_cal should be a DataFrame')
    if not isinstance(drop, (list, str)):
        # Raising error
        raise TypeError('drop should be a list or string')
    # Options to remove the row with NaN instances or removing the column with
    # NaN instances
    if drop == 'row':
        df_cal_cleaned = df_cal.dropna(axis=0)
        print('\nNumber of rows dropped:')
        print(len(df_cal) - len(df_cal_cleaned))
    elif drop == 'column':
        df_cal_cleaned = df_cal.dropna(axis=1)
        print('\nNumber of columns dropped:')
        print(len(df_cal.columns) - len(df_cal_cleaned.columns))
    # Option to fill in the NaN value fields with the median value of the
    # attribute using an imputer function
    elif drop == 'imputer':
        # Dropping non-numerical columns
        df_cal_num = df_cal.drop(columns=['ocean_proximity'])
        # Creating the imputer
        imp_median = SimpleImputer(strategy='median')
        # Fitting and using the imputer
        imp_median.fit(df_cal_num)
        print("\nThe median value of each numerical column is:")
        print(imp_median.statistics_)
        cal_cleaned = imp_median.transform(df_cal_num)
        # Transforming the data back to a DataFrame type
        df_cal_cleaned = pd.DataFrame(cal_cleaned, columns=df_cal_num.columns,
                                      index=df_cal_num.index)
        # Adding the ocean proximity column again
        df_cal_cleaned['ocean_proximity'] = df_cal['ocean_proximity']
    # Option to also just drop a column when given the name
    else:
        df_cal_cleaned = df_cal.drop(columns=drop)
        print(f'\nDropped the column(s): {drop}')
    return df_cal_cleaned


def OneHotEncoder_ocean_proximity(df_cal, show):
    """
    Function for turning the non-numerical field ocean_proximity into a
    numerical one with a OneHotEncoder.

    parameters:
        - show should be a boolean and should be True in order to show the
          new DataFrame
        - df_cal should be the california housing DataFrame

    returns:
        - new dataframe
        - the header of the OneHotEncoder part of the DataFrame
    """
    # Initial checks
    if not isinstance(df_cal, pd.DataFrame):
        # Raising error
        raise TypeError('df_cal should be a DataFrame')
    if not isinstance(show, bool):
        # Raising error
        raise TypeError('show should be a boolean')
    # Isolating the ocean_proximity column
    ocean_proximity = df_cal['ocean_proximity'].values.reshape(-1, 1)
    # Initiating the OneHotEncoder function
    prox_encoder = OneHotEncoder()
    # Fitting and using the function
    prox_encoder.fit(ocean_proximity)
    print("\nThe categories of ocean_proximity are:")
    print(list(prox_encoder.categories_[0]))
    ocean_proximity_hot = prox_encoder.transform(ocean_proximity)
    # Transforming the data back to a DataFrame type
    df_prox_hot = pd.DataFrame(ocean_proximity_hot.toarray(),
                               columns=list(prox_encoder.categories_[0]))
    # Adding the new columns back to the DataFrame
    df_cal_proxhot = (df_cal.drop(columns=['ocean_proximity'])
                      .join(df_prox_hot))
    if show:
        # Visualization of the data
        pd.set_option('display.max_columns', None)
        print(df_cal_proxhot.head())
        pd.reset_option('max_columns')
    return [df_cal_proxhot, list(prox_encoder.categories_[0])]


def Standard_scaler(to_drop, df_cal, show):
    """
    Function to standardscale the attributes of a DataFrame

    parameters:
        - to_drop should be the attributes you don't want to scale in a list
        - df_cal should be the california housing DataFrame
        - show should be a boolean and should be True in order to show the
          new DataFrame

    returns:
        - scaled dataframe
    """
    # Initial checks
    if not isinstance(df_cal, pd.DataFrame):
        # Raising error
        raise TypeError('df_cal should be a DataFrame')
    if not isinstance(show, bool):
        # Raising error
        raise TypeError('show should be a boolean')
    if not isinstance(to_drop, list):
        # Raising error
        raise TypeError('to_drop should be a list')
    df_hold = df_cal[to_drop]
    df_cal_num = df_cal.drop(columns=to_drop)
    # Creating the scaler object
    scaler = StandardScaler()
    # Fitting it and transforming the data
    scaler.fit(df_cal_num)
    cal_num = scaler.transform(df_cal_num)
    # Adding the removed attributes again
    df_cal_scaled = (pd.DataFrame(cal_num,
                                  columns=list(df_cal_num.columns))
                     .join(df_hold))
    if show:
        # Printing the new dataset
        pd.set_option('display.max_columns', None)
        print(df_cal_scaled.head())
        pd.reset_option('max_columns')
    return df_cal_scaled


class LinearRegression(torch.nn.Module):
    """
    Class that creates a LinearRegression model neural network that
    contains 1 linear layer and uses no dropout while training.
    """
    def __init__(self, input_size, output_size):
        """
        Initialises the layers of the linear regression model

        parameters:
            - Input_size should be the size of the linear layer
            - Output_size should be the size of the output
        """
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(in_features=input_size,
                                      out_features=output_size)
        # Dropout can be commented back in to use dropout during training
        # self.dropout = torch.nn.Dropout(p=0.05)

    def forward(self, x):
        """
        Forward pass function of the LinearRegression model, x should be
        a tensor containing the data.
        """
        out = self.linear(x)
        # out = self.dropout(out)
        return out


def train(dataloader, model, loss_function, optimizer):
    """
    Function to train the linear regression model and printing the progress
    over time.

    parameters:
        - dataloader: a dataloader object that provides the data to train
          the model on, with its labels
        - model: the model you want to train
        - loss_function: the loss function you want to use to evaluate the
          predicted labels compared to the real labels
        - optimizer: Optimizer function you want to use to set the weights
    """
    size = len(dataloader.dataset)
    # Putting the model in training mode
    model.train()
    print(f"training the model over all training data. Size = {size}")
    batch_count = 0
    total_loss = 0
    # Looping over all the batches that the dataloader creates
    for batch, (X, y) in enumerate(dataloader):
        # Resetting the gradients to 0
        optimizer.zero_grad()
        batch_count += 1
        X, y = X.to(device), y.to(device)
        # Predicting the labels with the model
        pred = model(X)
        # Calculating the loss
        loss = loss_function(pred, y)
        # Updating the weights of the model
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Keeping track of the total loss
        total_loss += loss.item()
        # Printing the process of the training every 50 batches
        if batch % 50 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    return total_loss / batch_count


def evaluate(dataloader, model, loss_function):
    """
    Function to evaluate the linear regression model

    parameters:
        - dataloader: a dataloader object that provides the data to train
          the model on, with its labels
        - model: the model you want to train
        - loss_function: the loss function you want to use to evaluate the
    """
    size = len(dataloader.dataset)
    # Put the model in evaluation mode
    model.eval()
    print(f"evaluating the model over all test data. Size = {size}")
    total_loss = 0
    # Putting the gradients to 0
    with torch.no_grad():
        # Looping over all the batches that the dataloader creates
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            # Predicting the labels with the model
            pred = model(X)
            # Calculating the loss
            loss = loss_function(pred, y)
            # Keeping track of the total loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    # =========================================================================
    # 1. Investigating and visualizing the data
    # =========================================================================
    # Initiating the cwd path
    cwd = Path.cwd()
    # importing the data in DataFrames
    df_ans = pd.read_csv(f'{cwd}/anscombe.csv', delimiter=',')
    df_cal = pd.read_csv(f'{cwd}/housing.csv', delimiter=',')
    # Printing the head to check the data
    print(df_cal.head())
    print('==================================================================='
          '====================')

    # A)_______________________________________________________________________
    # Printing the amount of instances of this dataset
    num_inst = len(df_cal)
    print("\nThe number of data instances are:", num_inst)

    # Printing the types of the value fields in the dataset
    data_types = df_cal.dtypes.to_string()
    print("\nThe data types of all the value fields in the dataset are:")
    print(data_types)
    data_types = df_cal.dtypes
    print()
    # Getting the non-numerical attributes and printing them
    print('following metrics have the a non-numerical value:')
    for item in data_types.items():
        if item[1] == 'object':
            print(item[0], 'has the value:', item[1])
    print('==================================================================='
          '====================')

    # B)_______________________________________________________________________
    # Plotting a histogram for all the attributes in the dataset
    df_cal.hist(figsize=(10, 10))
    plt.tight_layout()
    # plt.show()

    # C)_______________________________________________________________________
    # splitting the dataset in a training set and a test set for training the
    # regression model
    df_cal_train, df_cal_test = train_test_split(df_cal, test_size=0.2,
                                                 random_state=42)

    # D)_______________________________________________________________________
    # plotting the latitude and longitude in a scatter plot and also indicating
    # population of each district and the median house value
    marker_size = df_cal_train['population']/100
    fig = df_cal_train.plot.scatter(x='longitude', y='latitude', alpha=0.5,
                                    s=marker_size, label='population',
                                    c='median_house_value', colormap='jet')
    plt.title('Population and median house value of the districts of '
              'California')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # plt.show()

    # E)_______________________________________________________________________
    # Plotting the correlation matrix and also printing it to the terminal

    # Making it, so it can display the entire matrix by removing the limitation
    # on the amount of rows and columns it can print
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # Creating the correlation matrix
    correlation_matrix = df_cal.drop(columns=['ocean_proximity']).corr()
    print("\nCorrelation matrix of California housing information:")
    # Printing correlation matrix
    print(correlation_matrix)
    # Plotting correlation matrix
    plt.figure(figsize=(9, 9))
    sns.heatmap(correlation_matrix, cmap='coolwarm')
    plt.title('Correlation Matrix')
    pd.reset_option('max_columns')
    pd.reset_option('max_rows')
    print('==================================================================='
          '====================')
    # Plotting the scatter matrix can be done with this command, but it takes a
    # long time to load
    # pd.plotting.scatter_matrix(df_cal, diagonal='hist',  figsize=[10, 10])
    # plt.show()

    # =========================================================================
    # 2. Preprocessing the data (step by step)
    # =========================================================================

    # A)_______________________________________________________________________
    # printing the amount of NaN instances for each column
    amount_nan = df_cal.isna().sum().to_string()
    print("\nAmount of NaN instances for each column:")
    print(amount_nan)
    drop = 'imputer'  # ['total_bedrooms']
    df_cal_cleaned = Cleaning_NaN(drop, df_cal)
    print('==================================================================='
          '====================')

    # B)_______________________________________________________________________
    # Sampling 20 instances of the attribute ocean_proximity
    print('\nSample of 20 instances of the ocean_proximity column:')
    print(df_cal_cleaned['ocean_proximity'].sample(20))
    unique_categories = df_cal_cleaned['ocean_proximity'].unique()
    # Printing the different non-numerical options possible for this attribute
    print('\nThe different categories are:')
    print(unique_categories)
    [df_cal_cleaned_proxhot, OneHotHeader] = OneHotEncoder_ocean_proximity(
                                                            df_cal_cleaned,
                                                            True)
    print('==================================================================='
          '====================')

    # C)_______________________________________________________________________
    # Creating the object to add new attributes by combining other ones
    add_attributes = CombinedAttributesAdder()
    # Transforming the data
    df_cal_cleaned_proxhot_added = (add_attributes.
                                    transform(df_cal_cleaned_proxhot))

    # Plotting the correlation matrix again to see if these new attributes have
    # stronger correlation
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # Creating the correlation matrix
    correlation_matrix = df_cal_cleaned_proxhot_added.corr()
    print("\nCorrelation matrix of California housing information (extended):")
    # Printing the matrix
    print(correlation_matrix)
    # Plotting the matrix
    plt.figure(figsize=(9, 9))
    sns.heatmap(correlation_matrix, cmap='coolwarm')
    plt.title('Correlation Matrix')
    pd.reset_option('max_columns')
    pd.reset_option('max_rows')
    print('==================================================================='
          '====================')

    # D)_______________________________________________________________________
    print()
    # splitting the data to later remove the median_house_value because we
    # don't want to standardscale this value
    df_median_house_value = pd.DataFrame(
        df_cal_cleaned_proxhot_added['median_house_value'],
        columns=['median_house_value'])
    # Dropping the median_house_value column
    df_cal_to_scale = (df_cal_cleaned_proxhot_added
                       .drop(columns=['median_house_value']))
    df_cal_complete = Standard_scaler(OneHotHeader,
                                      df_cal_to_scale,
                                      True)
    # Printing the correlation matrix again to see if it changed or not
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    correlation_matrix = df_cal_complete.join(df_median_house_value).corr()
    print("\nCorrelation matrix of California housing information "
          "(normalised):")
    print(correlation_matrix)
    pd.reset_option('max_columns')
    pd.reset_option('max_rows')

    print('==================================================================='
          '====================')

    # =========================================================================
    # 3. Making a transformation pipeline
    # =========================================================================
    # initialising the estimators for the pipeline
    estimators = [('imp_median_pipe', SimpleImputer(strategy='median')),
                  ('add_attributes_pipe', CombinedAttributesAdder()),
                  ('scaler_pipe', StandardScaler())]
    # creating the pipeline
    pipe = Pipeline(estimators)
    # Stripping the dataframe of the data we don't want to process
    df_cal_stripped = df_cal.drop(columns=['median_house_value',
                                           'ocean_proximity'])
    # Fitting and transforming with the pipeline
    cal_processed = pipe.fit_transform(df_cal_stripped)
    # Transforming the data back to a dataframe
    df_cal_processed = pd.DataFrame(cal_processed,
                                    columns=list(df_cal_stripped.columns)
                                    + ['rooms_per_household',
                                       'bedrooms_per_room',
                                       'population_per_household'])
    print(df_cal_processed)
    # Checking if the DataFrame we get from processing it step by step and
    # with the pipeline are the same
    if df_cal_processed.equals(df_cal_complete.drop(
            columns=OneHotHeader)):
        print('\nThe DataFrames from the individual steps and the pipeline '
              'are the same.')
    print('==================================================================='
          '====================')

    # =========================================================================
    # Model training and evaluation
    # =========================================================================
    # 1) Data Conversion_______________________________________________________
    # Splitting the data in a training set and a test set
    df_train, df_test = train_test_split(df_cal_processed.
                                         join(df_median_house_value),
                                         test_size=0.2,
                                         random_state=42)
    # Splitting the labels and the data
    df_median_house_value = df_train[['median_house_value']]
    df_train = df_train.drop(columns=['median_house_value'])

    # Transforming the data to a tensor
    df_train = df_train.astype(np.float32)
    df_labels = df_median_house_value.astype(np.float32)
    tens_train = torch.from_numpy(df_train.values)

    # Scaling the labels with a MinMaxScaler and turning it in a tensor
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    scaler_fitted_train = scaler_train.fit(df_labels.values)
    fitted_labels_train = scaler_train.transform(df_labels.values)
    tens_labels = torch.from_numpy(fitted_labels_train)

    # Doing the same for the test data
    # Splitting the labels and the data
    df_median_house_value_test = df_test[['median_house_value']]
    df_test = df_test.drop(columns=['median_house_value'])

    # Transforming the data to a tensor
    df_test = df_test.astype(np.float32)
    df_labels_test = df_median_house_value_test.astype(np.float32)
    tens_test = torch.from_numpy(df_test.values)

    # Scaling the labels with a MinMaxScaler and turning it in a tensor
    scaler_test = MinMaxScaler(feature_range=(0, 1))
    scaler_fitted_test = scaler_test.fit(df_labels_test.values)
    fitted_labels_test = scaler_test.transform(df_labels_test.values)
    tens_labels_test = torch.from_numpy(fitted_labels_test)

    # This is some simple test data to test the linear regression model on
    """
    x_values = np.linspace(0, 100, 10000)
    y_values_true = 5 * x_values + 10
    std_dev = 0.1
    random_deviation = np.random.normal(0, std_dev, len(x_values))
    y_values_with_noise = y_values_true + random_deviation

    x_mean = np.mean(x_values)
    x_std = np.std(x_values)
    x_values_normalized = (x_values - x_mean) / x_std

    x_tensor = torch.tensor(x_values_normalized, 
                            dtype=torch.float32).reshape(-1, 1)
    y_tensor = torch.tensor(y_values_with_noise, 
                            dtype=torch.float32).reshape(-1, 1)
    dataset_train = TensorDataset(x_tensor, y_tensor)
    """
    # 2) Model creation________________________________________________________
    # initialising the sizes of the model neural network
    input_size = 11
    output_size = 1
    # determines the device (CPU or GPU) on which training and testing will be
    # executed.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initiate the model
    model = LinearRegression(input_size, output_size).to(device)

    # 3) Model Training________________________________________________________
    # Initiate the loss function
    loss_function = torch.nn.MSELoss()
    # Initiate the optimizer function and its parameters
    learning_rate = 0.00065
    momentum = 0.84
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
    #                             momentum=momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Initiate the schedular with its parameter
    gamma = 0.95
    scheduler = ExponentialLR(optimizer, gamma=gamma)

    # Printing the sizes of the training data
    print("Input data size:", tens_train.shape)
    print("Label data size:", tens_labels.shape)
    # Creating a TensorDataset from the training data and labels
    dataset_train = TensorDataset(tens_train, tens_labels)
    # Creating the DataLoader
    train_dataloader = DataLoader(dataset_train, batch_size=35, shuffle=True)
    print(len(tens_test))
    # Printing the sizes of the testing data
    print("Input data size:", tens_test.shape)
    print("Label data size:", tens_test.shape)
    # Creating a TensorDataset from the testing data and labels
    dataset_test = TensorDataset(tens_test, tens_labels_test)
    # Creating the DataLoader
    test_dataloader = DataLoader(dataset_test, batch_size=35, shuffle=True)

    # Initiating the number of epochs and other parameters
    num_epochs = 100
    epoch_plot = 5
    train_loss_list = []
    test_loss_list = []
    train_RMSE_list = []
    test_RMSE_list = []
    pred_list = []

    # Sampling some random rows to visualise the progress of the model later
    random_numbers = [random.randint(1, len(tens_test))-1 for _ in range(5)]
    five_rows = tens_test[random_numbers]
    five_rows_labels = torch.from_numpy(df_labels_test.values)[random_numbers]

    # Loop for training the model for an amount of epochs specified
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        # Training the model and getting the total loss
        train_loss = train(train_dataloader, model, loss_function, optimizer)
        scheduler.step()
        # Testing the model and getting the total loss
        test_loss = evaluate(test_dataloader, model, loss_function)
        # Adding the total losses to a list
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        # For some epochs keep track of the predictions of 5 random rows we
        # specified before
        if epoch % epoch_plot == 0:
            model.eval()
            with torch.no_grad():
                five_pred = model(five_rows.to(device))
            pred_list.append(scaler_test.inverse_transform(five_pred)
                             .squeeze().tolist())
        print()
    # 4) Model Evaluation______________________________________________________
    # Calculating the RMSE from the loss that just calculates the MSE
    train_RMSE_list = [math.sqrt(loss) for loss in train_loss_list]
    test_RMSE_list = [math.sqrt(loss) for loss in test_loss_list]
    x_values = range(1, num_epochs + 1)
    # Plotting the loss function for every epoch trained
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, train_loss_list, label='Train Loss')
    plt.plot(x_values, test_loss_list, label='Test Loss')
    plt.plot(x_values, train_RMSE_list, label='Train Loss RMSE')
    plt.plot(x_values, test_RMSE_list, label='Test Loss RMSE')
    plt.title('Training and Test Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # plt.show()
    # 5) Model Application_____________________________________________________
    # Plotting the predictions of the 5 random rows every couple epochs
    plt.figure()
    labels_list = five_rows_labels.squeeze().tolist()
    colors = ['blue', 'red', 'green', 'gold', 'chocolate']
    for i in range(5):
        # Initiating the x and y values for the plot
        x_values = [epoch_plot*(j+1) for j in range(len(pred_list))]
        y_values = [item[i] for item in pred_list]
        y2_values = [labels_list[i] for _ in range(len(pred_list))]
        # Plotting the predictions
        plt.plot(x_values, y_values, marker='o', color=colors[i],
                 label=f'prediction {i+1}')
        # Plotting the true label values
        plt.plot(x_values, y2_values, linestyle='--', color=colors[i],
                 label=f'label {i+1}')
    plt.title('Predictions for every 5th epoch')
    plt.xticks(x_values)
    plt.xlabel('Epoch')
    plt.ylabel('Mean_house_value')
    plt.legend()
    plt.show()
