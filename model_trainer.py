# Bjorn De Busschere, Bjorn.DeBusschere@student.uantwerpen.be, 15/04/2024
# =============================================================================
# Imports
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

# =============================================================================
# Functions
# =============================================================================


def model_trainer(model, model_name, x_train, y_train, x_test, y_test,
                  batch_size, epochs, summary=True, plot=True, show=True,
                  give_time=False):
    """
    Function for training and evaluating a convolutional neural network model.

    :param model: a convolutional neural network model
    :param model_name: name of the convolutional neural network model (for
                       plotting title)
    :param x_train: a np.array containing the training data
    :param y_train: a np.array containing the training labels
    :param x_test: a np.array containing the testing data
    :param y_test: a np.array containing the testing labels
    :param batch_size: an integer specifying the batch size for training the
                       model
    :param epochs: an integer specifying the amount of epochs for training the
                   model
    :param summary: a boolean to turn on the printing of the summary of the
                    model
    :param plot: a boolean to turn on the plotting of the performance metrics
                 of the model
    :param show: a boolean to turn on the showing of the plot of the
                 performance metrics of the model

    :param give_time: a boolean to turn on the printing of the time taken to
                      train the model
    :return: if specified, the performance metrics of the model
    """

    # Initial checks.
    if not isinstance(model_name, str):
        raise ValueError("model_name should be a string")
    if (not isinstance(x_train, np.ndarray)
            or not isinstance(y_train, np.ndarray)):
        raise ValueError("x_train and y_train should be numpy arrays")
    if (not isinstance(x_test, np.ndarray)
            or not isinstance(y_test, np.ndarray)):
        raise ValueError("x_test and y_test should be numpy arrays")
    if not (isinstance(batch_size, int) and batch_size > 0):
        raise ValueError("batch_size should be a positive integer")
    if not (isinstance(epochs, int) and epochs > 0):
        raise ValueError("epochs should be a positive integer")
    if (not isinstance(summary, bool) or not isinstance(plot, bool)
            or not isinstance(show, bool) or not isinstance(give_time, bool)):
        raise ValueError("summary, plot, show, and give_time should all be "
                         "booleans")

    # Printing the summary of the model if specified.
    if summary:
        model.summary()

    # Starting the timing of the training of the model.
    start_time = time.time()

    # Training the model.
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1
                        )

    # Ending the timing of the training of the model.
    end_time = time.time()

    # If specified print the time taken to train the model to the terminal.
    if give_time:
        print(f'Time taken to train the model: {round(end_time-start_time, 2)} '
              f'seconds')

    # Evaluating the model on the test set and printing it.
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # If specified plotting the performance metrics of the model.
    if plot:
        # Creating the x_axis of the plot to get the epochs in the plot right.
        x_values = range(1, epochs+1)
        pd.DataFrame(history.history, index=x_values).plot(figsize=(8, 5))
        plt.grid(True)
        plt.xticks(x_values)
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.title(f'loss and accuracy of {model_name}')
        # Set the vertical range of the plot to [0-3].
        plt.gca().set_ylim(0, 3)

    # When plotting is not specified, the performance metrics are returned in a
    # list.
    else:
        loss_list = history.history['loss']
        accuracy_list = history.history['accuracy']
        val_loss_list = history.history['val_loss']
        val_accuracy_list = history.history['val_accuracy']
        return [loss_list, accuracy_list, val_loss_list, val_accuracy_list]

    # If specified, show the plot of the performance metrics.
    if show:
        plt.show()
