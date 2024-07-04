# Bjorn De Busschere, Bjorn.DeBusschere@student.uantwerpen.be, 15/04/2024
# =============================================================================
# Imports
# =============================================================================
from tensorflow.keras import layers
from tensorflow import keras
from keras.optimizers import SGD

# =============================================================================
# Functions
# =============================================================================


def model_generator(n, best_lr=0.001, best_mom=0.9):
    """
    Function to generate a convolutional neural network model. A total of 13
    different models can be made with different amount of layers and
    hyperparameters.

    :param n: integer to indicate the type of model to generate.
    :param best_lr: best learning rate obtained from hyperparameter tuning.
    :param best_mom: best momentum value obtained from hyperparameter tuning.

    :return: the requested model with the specified hyperparameters.
    """

    # Initial checks.
    if not (isinstance(n, int) and 0 < n < 14):
        raise ValueError('n should be an integer from 1 to 13')
    if not (isinstance(best_lr, float) and best_lr > 0):
        raise TypeError('best_lr should be a positive float')
    if not (isinstance(best_mom, float) and best_mom > 0):
        raise TypeError('best_mom should be a positive float')

    # Initialising the different parameters for the different models for
    # hyperparameter tuning.
    lr_tuning = {4: 1.0, 5: 0.1, 6: 0.01, 7: 0.001, 8: 0.0001}
    mom_tuning = {9: 0.5, 10: 0.99}
    drop_tuning = {11: 0.2, 12: 0.5, 13: 0.8}

    # Defining some default parameters for our models.
    momentum = 0.9
    learning_rate = 0.001
    drop_out = 0.2

    # Changing the hyperparameters for some models.
    if n in range(4, 9):
        learning_rate = lr_tuning[n]
    elif n in range(9, 11):
        momentum = mom_tuning[n]
        learning_rate = best_lr
    elif n in range(11, 14):
        drop_out = drop_tuning[n]
        learning_rate = best_lr
        momentum = best_mom

    # Defining the input shape.
    input_shape = (32, 32, 3)

    # Defining the specified models given by the integer n.
    if n >= 1:
        inputs = keras.layers.Input(shape=input_shape)
        x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        if n >= 2:
            x = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Conv2D(128, kernel_size=(3, 3), activation="relu")(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            if n >= 3:
                x = layers.Dropout(drop_out)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        if n >= 3:
            x = layers.Dropout(drop_out)(x)
        output = layers.Dense(10, activation="softmax")(x)

    # Creating the model defined above.
    model = keras.models.Model(inputs=inputs, outputs=output)

    # Defining the optimizer for the model.
    opt = SGD(learning_rate=learning_rate, momentum=momentum)

    # Compiling the model with the categorical crossentropy for the loss, the
    # optimizer defined above and the accuracy as the metric.
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"]
                  )

    # Returning the created model
    return model
