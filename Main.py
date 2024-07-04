# Bjorn De Busschere, Bjorn.DeBusschere@student.uantwerpen.be, 19/05/2024
# =============================================================================
# Imports
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers
from keras.datasets import mnist
from keras.optimizers import Adam
import keras
import keras_tuner
from Image_Transformer import ImageTransformer
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# =============================================================================
# Functions
# =============================================================================


def show_number(image_data, label, ax=None, show=True, lines=True, title=True):
    """
    Function for plotting an image from the MNIST dataset.

    :param image_data: a np.array containing the image data of the image you
                       want to plot.

    :param label: an integer specifying the category of the image.

    :param show: a boolean to turn on the showing of the plot of the image

    :param ax: axis of the subplot you want to plot the image in

    :param lines: a boolean to turn on a line grid in the image

    :param title: a boolean to turn on addition of a title to the image
    """

    # Initial type checks.
    if not isinstance(image_data, np.ndarray):
        raise TypeError("image must be a numpy array")

    if not (isinstance(label, int) and 0 <= label <= 9):
        raise ValueError("label should be a positive integer from 0 to 9")

    if not isinstance(show, bool):
        raise ValueError("show should be a boolean")

    if not isinstance(lines, bool):
        raise ValueError("lines should be a boolean")

    if not isinstance(title, bool):
        raise ValueError("title should be a boolean")

    # Creating the figure for the plot.
    if ax is None:
        fig, ax = plt.subplots()

    # Plotting the image from the image_data.
    ax.imshow(image_data, cmap='gray_r', vmin=0, vmax=255)
    ax.axis('off')

    # Adding tile if specified.
    if title:
        ax.set_title(f' Image of the number: {label}')

    height, width = image_data.shape

    # Add a grid onto the image if specified.
    if lines:
        # Horizontal lines.
        ax.hlines(y=np.arange(0, height) + 0.5,
                  xmin=-0.5,
                  xmax=width - 0.5,
                  color='gray',
                  linewidth=1, alpha=0.3)
        # Vertical lines.
        ax.vlines(x=np.arange(0, width) + 0.5,
                  ymin=-0.5,
                  ymax=height - 0.5,
                  color='gray',
                  linewidth=1, alpha=0.3)

    # If specified, show the image.
    if show:
        plt.show()


def batch_show_image_transformation(batch, batch_label, amount=5, show=True,
                                    prob_translate=0.8, prob_scaling=0.8,
                                    prob_perspective=0.4, prob_rotate=0.9,
                                    prob_noise=0.9, prob_grad=0.75,
                                    prob_image_noise=0.8, prob_inv=0.4,
                                    prob_motion_blur=0.3, only=None):
    """
    Function for transforming a batch of images and showing them in a grid
    format.

    :param batch: np.array containing all the 28x28 images

    :param batch_label: labels of the images in the batch

    :param amount: integer specifying the amount of variations of each image
                   you want to plot.

    :param show: A boolean to turn on the showing of the plot of the image.

    :param prob_translate: float less or equal to 1 specifying the
                               probability of the translation transformation.

    :param prob_scaling: float less or equal to 1 specifying the
                         probability of the scaling transformation.

    :param prob_perspective: float less or equal to 1 specifying the
                             probability of the perspective transformation.

    :param prob_rotate: float less or equal to 1 specifying the
                        probability of the rotation transformation.

    :param prob_noise: float less or equal to 1 specifying the
                       probability of the noise transformation.

    :param prob_grad: float less or equal to 1 specifying the
                      probability of the gradient transformation.

    :param prob_image_noise: float less or equal to 1 specifying the
                             probability of the blur/colorjitter transformation

    :param prob_inv: float less or equal to 1 specifying the
                     probability of the inversion transformation.

    :param prob_motion_blur: float less or equal to 1 specifying the
                             probability of the motion blur transformation.

    :param only: a string of the name of a transformation, the probability of
                 this transformation will be made 1, while the probability of
                 all other transformations will be 0.
    """

    # Initial type checks.
    if not isinstance(batch, np.ndarray):
        raise TypeError("batch must be a numpy array")

    if batch.ndim != 3:
        raise ValueError("batch must be a 3D numpy array")

    if not isinstance(batch_label, np.ndarray) or batch_label.ndim != 1:
        raise ValueError("batch_label must be a 1D numpy array")

    if not isinstance(amount, int):
        raise ValueError("amount must be an integer")

    if not isinstance(show, bool):
        raise ValueError("show must be a boolean")

    if not all(0 <= p <= 1 for p in [prob_translate, prob_scaling,
                                     prob_perspective,
                                     prob_rotate, prob_noise,
                                     prob_grad, prob_image_noise,
                                     prob_inv, prob_motion_blur]):
        raise ValueError("the probabilities of the transformations must be "
                         "between 0 and 1")

    if only is not None and only not in ['translate', 'scaling', 'perspective',
                                         'rotate', 'noise', 'gradient',
                                         'image_noise', 'inv', 'motion_blur']:
        raise ValueError(
            "only must be a string of ['translate', 'scaling', 'perspective', "
            "'rotate', 'noise', 'gradient', 'image_noise', 'inv', "
            "'motion_blur'] or None")

    # Making all probabilities 0 if a transformation is passed in 'only'.
    if only is not None:
        probabilities = {'translate': 0, 'scaling': 0,
                         'perspective': 0, 'rotate': 0,
                         'noise': 0, 'gradient': 0,
                         'image_noise': 0, 'inv': 0,
                         'motion_blur': 0}

        # Making the specified probability 1.
        probabilities[only] = 1

    # If only is None (by default) set the probabilities accordingly.
    else:
        probabilities = {
            'translate': prob_translate,
            'scaling': prob_scaling,
            'perspective': prob_perspective,
            'rotate': prob_rotate,
            'noise': prob_noise,
            'gradient': prob_grad,
            'image_noise': prob_image_noise,
            'inv': prob_inv,
            'motion_blur': prob_motion_blur
        }

    # If batch has more than 10 images, raise an error.
    if len(batch) > 10:
        raise ValueError('size of batch is too big, max 10 images')

    # Initialising a transformation object.
    trans = ImageTransformer()

    # Creating the figure for the plot.
    fig, axes = plt.subplots(amount+1, len(batch))
    # Add a title for the figure.
    fig.suptitle('Comparison of images: before and after transformation',
                 fontsize=16)

    # Pass the batch to the transformer object and transform the images.
    for i in range(0, amount+1):
        im = trans.random_batch_transform(batch,
                                          prob_translate=probabilities['translate'],
                                          prob_scaling=probabilities['scaling'],
                                          prob_perspective=probabilities['perspective'],
                                          prob_rotate=probabilities['rotate'],
                                          prob_noise=probabilities['noise'],
                                          prob_grad=probabilities['gradient'],
                                          prob_image_noise=probabilities['image_noise'],
                                          prob_inv=probabilities['inv'],
                                          prob_motion_blur=probabilities['motion_blur']
                                          )

        # Defining a rectangle to be put around the original images.
        rect = patches.Rectangle((0, 0), 1, 1,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')

        # Plotting the images in the batch and the transformed images.
        for j in range(0, len(batch)):
            # Plotting the original images.
            if i == 0:
                show_number(batch[j], int(batch_label[j]), axes[i, j],
                            show=False, lines=False, title=False)
                ax = axes[i, j]
                # Adding the borders for the images.
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                rect = patches.Rectangle((xmin, ymin),
                                         xmax - xmin,
                                         ymax - ymin,
                                         linewidth=2,
                                         edgecolor='black',
                                         facecolor='none')
                ax.add_patch(rect)
            # Plotting the other images without border.
            else:
                show_number(im[j], int(y_test[j]), axes[i, j],
                            show=False, lines=False, title=False)

    # Show the images if specified.
    if show:
        plt.show()


class MyHyperModel(keras_tuner.HyperModel):
    """
    Class to build and fit a DNN model with variable hyperparameters for use in
    a keras.tuner.
    """
    def build(self, hp):
        """
        Builds a DNN model with variable hyperparameters.

        :param hp: a `HyperParameters` object containing the hyperparameters to
                   be tuned.

        :return: a compiled model with variable hyperparameters
        """

        # Defining the model in sequential mode to add layers to it.
        model = keras.Sequential()

        # Adding a flatten layer to add it pass the images into the DNN.
        model.add(layers.Flatten())

        # Tuning number of layers.
        for i in range(hp.Int("num_layers", min_value=1, max_value=5, step=1)):
            model.add(
                layers.Dense(
                    # Tuning the amount of neurons in each layer.
                    units=hp.Int(f"units_{i}", min_value=32, max_value=512,
                                 step=32),
                    # Tuning the activation function of the layers.
                    activation=hp.Choice("activation",
                                         values=["relu", "tanh"]),
                            )
                    )

            # Adding dropout layers to the model and tuning the drop out rate.
            model.add(
                layers.Dropout(rate=hp.Float("dropout_rate", min_value=0.0,
                                             max_value=0.5, step=0.1)
                               )
                     )

        # Adding the final dense layer with softmax activation function.
        model.add(layers.Dense(10, activation="softmax"))

        # Tuning the optimizer.
        optimizer_options = hp.Choice("optimizer", values=["adam", "sgd"])

        # Tuning the learning rate.
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2,
                                 sampling="log")

        # Tuning the momentum.
        momentum = hp.Float("momentum", min_value=0.0, max_value=0.8, step=0.1)

        # Tuning the amount of epochs.
        epochs = hp.Int("epochs", min_value=5, max_value=60, step=5)

        # Defining the optimizer based on the tuning parameters.
        if optimizer_options == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate,
                                             momentum=momentum)

        # Compile the model with the specified parameters.
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Return the compiled model.
        return model

    def fit(self, hp, model, *args, **kwargs):
        """
        Fits the model with the given data.

        :param hp: A `HyperParameters` object containing the hyperparameters

        :param model: A compiled Keras model

        :param args: other arguments to input in model.fit

        :param kwargs: other keyword arguments to input in model.fit

        :return: the history of the training of the model
                 ((val) loss and accuracy)
        """

        # Tuning the batch size.
        batch_size = hp.Choice("batch_size", [16, 32, 64])

        # Setting the amount of epochs from tuning.
        epochs = hp.get('epochs')

        # Fit the model with the defined parameters.
        return model.fit(
            *args,
            batch_size=batch_size,
            epochs=epochs,
            **kwargs,
        )


def confusion_matrix(model, x_test, y_test, show=False, give_metrics=True):
    """
    Function to plot the confusion matrix of a model when evaluated on a test
    set.

    :param model: a trained keras model

    :param x_test: the test dataset images

    :param y_test: the test dataset labels

    :param show: A boolean to turn on the showing of the plot of the matrix.

    :param give_metrics: A boolean to turn on the returning of the metrics:
                         precision, recall and F1 score for all the categories.

    :return: precision, recall and F1 score for all the categories in lists
    """

    # Initial type checks.
    if not isinstance(model, keras.Model):
        raise TypeError("model must be a trained Keras model")

    if not isinstance(x_test, (np.ndarray, tf.Tensor)):
        raise TypeError("x_test must be a numpy array or a TensorFlow tensor")

    if not isinstance(y_test, (np.ndarray, tf.Tensor)):
        raise TypeError("y_test must be a numpy array or a TensorFlow tensor")

    if not isinstance(show, bool):
        raise TypeError("show must be a boolean")

    if not isinstance(give_metrics, bool):
        raise TypeError("give_metrics must be a boolean")

    # Defining the predicted labels by the trained model.
    y_pred = np.argmax(model.predict(x_test), axis=1)

    # Defining the actual labels from the database.
    y_true = np.argmax(y_test, axis=1)

    # Creating the confusion matrix.
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    # Calculate the percentage of each prediction category.
    percentages = confusion_matrix / len(x_test)

    # Plotting the confusion matrix.
    cm_display = (metrics.
                  ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                         display_labels=list(range(10))))

    # Setting the colormap of the plot.
    cm_display.plot(cmap=plt.cm.GnBu, colorbar=False)

    # Adding the percentages of each value on the matrix, underneath the
    # numbers.
    for i in range(10):
        for j in range(10):
            if percentages[i, j] != 0:
                plt.text(j, i+0.2, f'{percentages[i, j] * 100:.2f}%',
                         ha='center', va='top', color='red', fontsize=8)

    # If specified, calculate the performance metrix of the model.
    if give_metrics:
        # Initialise empty lists.
        precision = []
        recall = []
        f1 = []

        # Calculating the performance metrix for each category.
        for i in range(10):
            precision.append(confusion_matrix[i, i]/(sum(confusion_matrix[:, i])))
            recall.append(confusion_matrix[i, i] / (sum(confusion_matrix[i, :])))
            f1.append((2 * precision[i] * recall[i])/(recall[i] + precision[i]))

        # Returning the metrics.
        return precision, recall, f1

    # If specified, show the image of the confusion matrix.
    if show:
        plt.show()


def ROC(model, x_test, y_test, show=False):
    """
    Function for plotting the ROC curve of a model when evaluated on a test set

    :param model: a trained keras model

    :param x_test: the test dataset images

    :param y_test: the test dataset labels

    :param show: A boolean to turn on the showing of the plot of the ROC curve.
    """

    # Initial type checks.
    if not isinstance(model, keras.Model):
        raise TypeError("model must be a trained Keras model")

    if not isinstance(x_test, (np.ndarray, tf.Tensor)):
        raise TypeError("x_test must be a numpy array or a TensorFlow tensor")

    if not isinstance(y_test, (np.ndarray, tf.Tensor)):
        raise TypeError("y_test must be a numpy array or a TensorFlow tensor")

    if not isinstance(show, bool):
        raise TypeError("show must be a boolean")

    # Computing the probabilities for each category for each image.
    y_pred_prob = model.predict(x_test)

    # Getting the true labels.
    y_true = np.argmax(y_test, axis=1)

    # Initialising empty dictionaries.
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Getting the fpr and tpr and auc for each category.
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int),
                                      y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Initialising the figure to plot the curves on.
    plt.figure(figsize=(8, 6))

    # Plotting the ROC curves in the figure.
    for i in range(10):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} '
                                       f'(AUC = {roc_auc[i]:.5f})')

    # Plotting the reference diagonal.
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    # Formatting the plot.
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend(loc="lower right")

    # If specified, show the image of the ROC curves.
    if show:
        plt.show()


def display_label(x_test, y_test, number_subplots=5, label_to_plot=1,
                  show=False):
    """
    Function of displaying only instances of a certain label from the dataset
    in a subplot.

    :param x_test: the test dataset images

    :param y_test: the test dataset labels

    :param number_subplots: an integer specifying the size of the grid to plot
                            the images in.

    :param label_to_plot: an integer specifying the label of the instances
                          to plot.

    :param show: A boolean to turn on the showing of the plot of the images.

    """

    # Initial type checks.
    if not isinstance(x_test, (np.ndarray, tf.Tensor)):
        raise TypeError("x_test must be a numpy array or a TensorFlow tensor")

    if not isinstance(y_test, (np.ndarray, tf.Tensor)):
        raise TypeError("y_test must be a numpy array or a TensorFlow tensor")

    if not (isinstance(number_subplots, int) and 0 <= number_subplots <= 10):
        raise ValueError("number_subplots should be a positive integer from 0 "
                         "to 9")

    if not (isinstance(label_to_plot, int) and 0 <= label_to_plot <= 9):
        raise ValueError("label_to_plot should be a positive integer from 0 "
                         "to 9")

    if not isinstance(show, bool):
        raise TypeError("show must be a boolean")

    # Initiating the subplot.
    fig, axes = plt.subplots(number_subplots, number_subplots)

    # Starting a count.
    count = 0

    # Looping over the different elements in the axes of the subplot.
    for i in range(number_subplots):
        for j in range(number_subplots):
            test = False

            # Looping over the y_test element to check whether the label
            # is label_to_plot.
            while not test:
                if int(y_test[count]) == label_to_plot:

                    # If the label is label_to_plot plot the number.
                    show_number(x_test[count], int(y_test[count]),
                                axes[i, j], show=False, title=False,
                                lines=False)

                    # Increase the count.
                    count += 1

                    # Make test True to stop the while loop.
                    test = True
                else:

                    # When the label is not label_to_plot, also increase the
                    # count.
                    count += 1

    if show:
        # Show the plot.
        plt.show()

# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Importing the dataset.
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Importing the MNIST dataset.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Training and testing options.
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Options for running the main function (only test/tune one model at
    # the time).
    transformations_show = False
    tune_DNN = False
    test_DNN = True
    train_set = 'original'  # 'transformed'
    train_CNN = False
    test_CNN = False

    # Option for printing only the image one (example in report)
    number_example = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Preprocessing.
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Making a plot of 10 images with and without transformations.
    if transformations_show:

        batch_show_image_transformation(x_test[516:526], y_test[516:526],
                                        amount=9, show=True)

    # plotting only a certain label of numbers in a subplot.
    if number_example:
        display_label(x_test, y_test, 10, 9,
                      True)

    # Initialising a transformer object.
    transformer = ImageTransformer()

    if tune_DNN or test_DNN or test_CNN or train_CNN:

        # Transforming the test dataset
        x_test_trans = transformer.random_batch_transform(x_test)

    if train_set == 'transformed' and (tune_DNN or train_CNN):

        # Converting the image dataset 2 times to increase the training size.
        x_train_trans1 = transformer.random_batch_transform(x_train)
        x_train_trans2 = transformer.random_batch_transform(x_train)

        # Concatenating the images into one array.
        x_train_trans_tot = np.concatenate((x_train_trans1,
                                           x_train_trans2),
                                           axis=0)

        # Printing the shape of the array for clarity.
        print(x_train_trans_tot.shape)

        # Concatenating the labels into one big array.
        y_train_tot = np.concatenate((y_train, y_train), axis=0)

        # Printing the shape of the array for clarity.
        print(y_train_tot.shape)

        # Splitting the training and label datasets into training and
        # validation sets.
        x_train, x_val, y_train, y_val = train_test_split(x_train_trans_tot,
                                                          y_train_tot,
                                                          test_size=0.2,
                                                          random_state=42)

    elif train_set == 'original' and (tune_DNN or train_CNN):

        # Splitting the training and label datasets into training and
        # validation sets.
        x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                          y_train,
                                                          test_size=0.2,
                                                          random_state=42)

    if tune_DNN or train_CNN:

        # Normalising the pixel values of all the images in the datasets.
        x_train = np.expand_dims(x_train, -1).astype("float32")/255.0
        x_val = np.expand_dims(x_val, -1).astype("float32")/255.0
        x_test = np.expand_dims(x_test, -1).astype("float32")/255.0
        x_test_trans = np.expand_dims(x_test_trans, -1).astype("float32")/255.0

        # Turning the labels into One Hot Encoded vectors.
        num_classes = 10
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_val = keras.utils.to_categorical(y_val, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    if test_CNN or test_DNN:

        # Normalising the pixel values of all the images in the datasets.
        x_test = np.expand_dims(x_test, -1).astype("float32")/255.0
        x_test_trans = np.expand_dims(x_test_trans, -1).astype("float32")/255.0

        # Turning the labels into One Hot Encoded vectors.
        num_classes = 10
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # DNN model.
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    if tune_DNN:

        # Initialising the tuner object for searching the search space.
        tuner = keras_tuner.RandomSearch(
            MyHyperModel(),
            objective="val_accuracy",
            max_trials=30,
            overwrite=True,
            directory="my_dir",
            project_name="tune_hypermodel",
        )

        # Plotting a summary of the search space.
        tuner.search_space_summary()

        # Searching the search space with the tuner.
        tuner.search(
            x=x_train,
            y=y_train,

            # Terminating the models early when progress stagnates to prevent
            # overfitting.
            callbacks=[tf.keras.callbacks.EarlyStopping('val_loss',
                                                        patience=3)],
            validation_data=(x_val, y_val)
        )

        # Printing the results of the searching.
        tuner.results_summary()

        # Get the best models found through tuning.
        models = tuner.get_best_models(num_models=2)

        # Print a summary of the best model.
        best_model = models[0]
        best_model.summary()

        # Initialising the MyHyperModel object.
        hypermodel = MyHyperModel()

        # Get the hyperparameters of the best model found when tuning.
        best_hp = tuner.get_best_hyperparameters()[0]

        # Build the optimal model using the optimal hyperparameters.
        model = hypermodel.build(best_hp)

        # Fit the optimal model and storing its progress.
        history = hypermodel.fit(
            hp=best_hp,
            model=model,
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping('val_loss',
                                                        patience=3)],
        )

        # Evaluating the model on the test dataset with transformed images.
        score = model.evaluate(x_test_trans, y_test, verbose=0)
        # printing the performance.
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        # Printing the history of the model training.
        print(pd.DataFrame(history.history))

        # Plotting the learning curves of the training of the model.
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.title(f'loss and accuracy of the best model')
        plt.gca().set_ylim(0, 1)

        # Save the optimal model as a keras file.
        file_path = Path(r"C:\Bjorn\fysica Bach 3\programmeren voor fysici\Examenopdracht\optimal_model_trans4.keras")
        model.save(file_path)

        # Showing the plot
        plt.show()

    if test_DNN:
        # Load the model from a stored Keras file.
        optimal_model = keras.models.load_model(r"C:\Bjorn\fysica Bach 3\programmeren voor fysici\Examenopdracht\optimal_model_trans3.keras")

        # Evaluating the model on the transformed dataset.
        score = optimal_model.evaluate(x_test_trans, y_test, verbose=0)

        # Printing a summary of the model.
        optimal_model.summary()

        # Printing the performance of the model when evaluated on the test set.
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        # Creating the confusion matrix plot and getting the performance
        # metrics when evaluated on the original test set.
        precision1, recall1, f1_1 = confusion_matrix(optimal_model,
                                                     x_test, y_test,
                                                     show=False,
                                                     give_metrics=True)

        # Creating the confusion matrix plot and getting the performance
        # metrics when evaluated on the transformed test set.
        precision2, recall2, f1_2 = confusion_matrix(optimal_model,
                                                     x_test_trans, y_test,
                                                     show=False,
                                                     give_metrics=True)

        # Printing the performance metrics.
        print('Performance metrics of original data:')
        df = pd.DataFrame({"Precision": precision1, "Recall": recall1,
                           "F1 Score": f1_1})
        print(df)

        print('Performance metrics of transformed data:')
        df = pd.DataFrame({"Precision": precision2, "Recall": recall2,
                           "F1 Score": f1_2})
        print(df)

        # Creating the ROC curve plot for the model when evaluated on the
        # original test set.
        ROC(optimal_model, x_test, y_test, False)

        # Creating the ROC curve plot for the model when evaluated on the
        # transformed test set.
        ROC(optimal_model, x_test_trans, y_test, False)

        # Showing all the plots.
        plt.show()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # CNN model.
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    if train_CNN:

        # Converting the datasets to tf.tensors.
        x_train = tf.convert_to_tensor(x_train)
        x_test = tf.convert_to_tensor(x_test)
        x_val = tf.convert_to_tensor(x_val)
        x_test_trans = tf.convert_to_tensor(x_test_trans)

        # Giving the grayscale images 3 color channels to fit with the input of
        # The ResNet50 architecture.
        x_train = tf.image.grayscale_to_rgb(x_train)
        x_test = tf.image.grayscale_to_rgb(x_test)
        x_val = tf.image.grayscale_to_rgb(x_val)
        x_test_trans = tf.image.grayscale_to_rgb(x_test_trans)

        # Changing the dimensions of the images to 32x32 to fit with the input
        # dimensions of the ResNet50 architecture.
        x_train = tf.image.resize(x_train, [32, 32])
        x_test = tf.image.resize(x_test, [32, 32])
        x_val = tf.image.resize(x_val, [32, 32])
        x_test_trans = tf.image.resize(x_test_trans, [32, 32])

    if test_CNN:
        # Converting the datasets to tf.tensors.
        x_test = tf.convert_to_tensor(x_test)
        x_test_trans = tf.convert_to_tensor(x_test_trans)

        # Giving the grayscale images 3 color channels to fit with the input of
        # The ResNet50 architecture.
        x_test = tf.image.grayscale_to_rgb(x_test)
        x_test_trans = tf.image.grayscale_to_rgb(x_test_trans)

        # Changing the dimensions of the images to 32x32 to fit with the input
        # dimensions of the ResNet50 architecture.
        x_test = tf.image.resize(x_test, [32, 32])
        x_test_trans = tf.image.resize(x_test_trans, [32, 32])

    if train_CNN:
        # Defining the input shape of the ResNet50 model.
        input_shape = tf.keras.Input(shape=(32, 32, 3))

        # Importing the ResNet50 model architecture with random weights and
        # without the fully connected layer at the top since we want to use
        # our own classifier.
        net = tf.keras.applications.ResNet50(include_top=False,
                                             weights=None,
                                             input_tensor=input_shape)

        # MaxPooling the output of our CNN model to input into the classifier
        # layer.
        gap = tf.keras.layers.GlobalMaxPooling2D()(net.output)

        # Creating the classifier layer with 10 outputs.
        output = tf.keras.layers.Dense(10, activation='softmax')(gap)

        # Creating the model with the architecture defined above.
        model_Net = tf.keras.Model(net.input, output)

        # Compiling the model.
        model_Net.compile(optimizer=Adam(),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        # Setting the amount of epochs for the training of the ResNet50 model.
        epochs = 5

        # Training the CNN model on the training set and getting her
        # performance while training.
        history = model_Net.fit(x_train, y_train,
                                batch_size=128,
                                epochs=epochs,
                                validation_data=(x_val, y_val))

        # Getting the x values for the plot from the amount of epochs trained.
        x_values = range(1, epochs + 1)

        # Printing the performance metrics while training.
        print(pd.DataFrame(history.history))

        # Plotting the performance metrics.
        pd.DataFrame(history.history, index=x_values).plot(figsize=(8, 5))

        # Formatting the plot.
        plt.grid(True)
        plt.xticks(x_values)
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.title(f'loss and accuracy of ResNet50 model')
        plt.gca().set_ylim(0, 3)

        # Evaluating the model on the test dataset.
        score = model_Net.evaluate(x_test, y_test, verbose=0)

        # Printing the performance metrics of the model on the test dataset.
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        # saving the CNN model as a keras file.
        file_path = Path(r"C:\Bjorn\fysica Bach 3\programmeren voor fysici\Examenopdracht\model_Net_trans_final2.keras")
        model_Net.save(file_path)

        # Showing the plot.
        plt.show()

    if test_CNN:
        # Loading the CNN model from a keras file.
        model_Net = keras.models.load_model(r"C:\Bjorn\fysica Bach 3\programmeren voor fysici\Examenopdracht\model_Net_trans_final2.keras")

        # Evaluating the model on the transformed dataset.
        score = model_Net.evaluate(x_test_trans, y_test, verbose=0)

        # Printing a summary of the model architecture.
        model_Net.summary()

        # Printing the performance metrics of the model on the transformed test
        # dataset.
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        # Creating the confusion matrix plot and getting the performance
        # metrics when evaluated on the original test set.
        precision1, recall1, f1_1 = confusion_matrix(model_Net,
                                                     x_test, y_test,
                                                     show=False,
                                                     give_metrics=True)

        # Creating the confusion matrix plot and getting the performance
        # metrics when evaluated on the transformed test set.
        precision2, recall2, f1_2 = confusion_matrix(model_Net,
                                                     x_test_trans, y_test,
                                                     show=False,
                                                     give_metrics=True)

        # Printing the performance metrics.
        print('Performance metrics of original data:')
        df = pd.DataFrame({"Precision": precision1, "Recall": recall1,
                           "F1 Score": f1_1})
        print(df)

        print('Performance metrics of transformed data:')
        df = pd.DataFrame({"Precision": precision2, "Recall": recall2,
                           "F1 Score": f1_2})
        print(df)

        # Creating the ROC curve plot for the model when evaluated on the
        # original test set.
        ROC(model_Net, x_test, y_test, False)

        # Creating the ROC curve plot for the model when evaluated on the
        # transformed test set.
        ROC(model_Net, x_test_trans, y_test, False)

        # Showing all the plots.
        plt.show()
