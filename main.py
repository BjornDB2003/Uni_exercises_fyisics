# Bjorn De Busschere, Bjorn.DeBusschere@student.uantwerpen.be, 15/04/2024
# =============================================================================
# Imports
# =============================================================================
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model_generator import model_generator
from keras.utils import to_categorical
from model_trainer import model_trainer

# =============================================================================
# Functions
# =============================================================================


def show_image(image_data, label, show=True):
    """
    Function for plotting an image from the CIFAR10 dataset.

    :param image_data: a np.array containing the image data of the image you
                       want to plot.
    :param label: an integer specifying the category of the image.
    :param show: a boolean to turn on the showing of the plot of the image
    """
    if not isinstance(x_train, np.ndarray):
        raise ValueError("image_data should be numpy arrays")
    if not (isinstance(label, int) and 0 <= label <= 9):
        raise ValueError("label should be a positive integer from 0 to 9")
    if not isinstance(show, bool):
        raise ValueError(" show should be a boolean")

    # Defining the category for each label.
    labels = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
              5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

    # Creating the figure for the plot
    plt.figure()

    # Plotting the image from the image_data
    plt.imshow(image_data)
    plt.axis('off')
    plt.title(f'CIFAR10 32x32 image of: {labels[label]} (label {label})')

    # If specified, show the image.
    if show:
        plt.show()


def compare_parameters(list_to_plot, para_list, para, para_abr, epoch=5,
                       ask=False):
    """
    Function for plotting the performance metrics of training data from a model
    that has been trained with different values of a certain hyperparameter.
    Thus function also returns the best hyperparameter value, this can be either
    inputted by hand or calculated automatically.

    :param list_to_plot: a list containing the performance metrics of the model
    :param para_list: a list containing the values of the hyperparameters the
                      model was trained on
    :param para: a string of the name of the hyperparameter (for plotting)
    :param para_abr: a string of the abbreviation of the hyperparameter (for
                     the legend of the plot)
    :param epoch: an integer specifying the amount of epochs the model was
                  trained for (default to 5)
    :param ask: a boolean to let the user input the best value for the
                hyperparameter

    :return: the hyperparameter value with the best achieved loss. This
             function does not take into account overfitting or other
             complications concerning the performance of the model
    """

    # Initial checks.
    if not isinstance(list_to_plot, list):
        raise ValueError("list_to_plot should be a list")
    if not isinstance(para_list, list):
        raise ValueError("para_list should be a list")
    if not isinstance(para, str):
        raise ValueError("para should be a string")
    if not isinstance(para_abr, str):
        raise ValueError("para_abr should be a string")
    if not (isinstance(epoch, int) and epoch > 0):
        raise ValueError("epoch should be a positive integer.")
    if not isinstance(ask, bool):
        raise ValueError("ask should be a boolean")

    # Initialising the colors for plotting the function.
    color = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']

    # initialising the x values of the plot based on the amount of epochs the
    # model was trained on.
    if para == 'Epoch':
        epochs = [range(1, ep+1) for ep in para_list]
    else:
        epochs = [range(1, epoch + 1) for _ in para_list]

    # Initialize an empty vector.
    loss_track = []

    # Creating a subplot object.
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Looping over all the values of the hyperparameter and plotting the loss
    # for that model.
    for i in range(0, len(para_list)):
        ax1.plot(epochs[i], list_to_plot[i][0],
                 label=f'Loss ({para_abr}={para_list[i]})', color=color[i])
        ax1.plot(epochs[i], list_to_plot[i][2],
                 label=f'Val loss ({para_abr}={para_list[i]})', color=color[i],
                 linestyle='--')
        # Keeping track of the final achieved loss of that model.
        loss_track.append(list_to_plot[i][0][-1])
    ax1.set_title(f'Loss for different {para}')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper left', fontsize='small')

    # Looping over all the values of the hyperparameter and plotting the
    # accuracy for that model.
    for i in range(0, len(para_list)):
        ax2.plot(epochs[i], list_to_plot[i][1],
                 label=f'Accuracy ({para_abr}={para_list[i]})', color=color[i])
        ax2.plot(epochs[i], list_to_plot[i][3],
                 label=f'Val Accuracy ({para_abr}={para_list[i]})',
                 color=color[i], linestyle='--')
    ax2.set_title(f'Accuracy for different {para}')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='upper left', fontsize='small')

    plt.tight_layout()

    # Ask for the best value of the hyperparameter or calculate it
    # automatically
    if ask:
        # user input for hyperparameter
        plt.show()
        best_para_str = input(f"What is the best {para}? ")
        # Test of the input
        try:
            best_para_fl = float(best_para_str)
            if best_para_fl == int(best_para_fl):
                best_para = int(best_para_fl)
            else:
                best_para = best_para_fl
        except ValueError:
            raise ValueError("best hyperparameter should be an integer or a "
                             "float")
    else:
        # Calculating the best value of the hyperparameter only based on the
        # loss.
        best_para = para_list[loss_track.index(min(loss_track))]

    # Returning the best hyperparameter.
    return best_para

# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    # Importing the data from keras.
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Concatenating the data we imported into one big array for preprocessing.
    img_data = np.concatenate((x_train, x_test), axis=0)
    img_labels = np.concatenate((y_train, y_test), axis=0)

    # Normalizing the image_data to the interval [0, 1].
    img_data = img_data.astype('float32') / 255

    # One hot encoding the labels to a binary vector.
    img_labels_hot = to_categorical(img_labels, num_classes=10)

    # Splitting the dataset into a train en test set.
    img_data_train, img_data_test, img_labels_train, img_labels_test = (
        train_test_split(img_data,
                         img_labels_hot,
                         test_size=0.2,
                         random_state=42)
    )

    # initialising some random integers to show example images.
    rand_int = [0, 45552, 30165, 16741, 20321]
    for i in rand_int:
        # Plotting the images with the show_image function.
        show_image(img_data[i], int(img_labels[i][0]), False)
    # Showing all the images at once.
    plt.show()

    # Loop for creating and training the first 3 models.
    for n in range(1, 4):

        # Creating the model
        model = model_generator(n)

        # Training the model and plotting the performance metrics of the model.
        model_trainer(model, f'model {n}', img_data_train,
                      img_labels_train, img_data_test, img_labels_test,
                      256, 5, True, True,
                      False)

    # Showing all the plots
    plt.show()

    # Option to let the best hyperparameter be calculated by the script or let
    # it be inputted by the user
    ask = True

    # Initialising an empty list for keeping track of the performance metrics.
    list_to_plot = []

    # Looping over the next couple models with different hyperparameters.
    for i in range(4, 9):

        # Creating the model.
        model = model_generator(i)

        # Training the model and printing the time taken to train the model.
        metrics = model_trainer(model, f'model {i}', img_data_train,
                                img_labels_train, img_data_test,
                                img_labels_test, 256, 5,
                                False, False, False,
                                True)

        # Keeping track of the performance metrics.
        list_to_plot.append(metrics)

    # Plotting the performance metrics with the function defined above.
    # Also determining the optimal value for the learning rate for the next
    # step.
    best_lr = compare_parameters(list_to_plot,
                                 [1.0, 0.1, 0.01, 0.001, 0.0001],
                                 'Learning Rate', 'lr',
                                 ask=ask)

    # Initialising an empty list for keeping track of the performance metrics.
    list_to_plot = []

    # Looping over the next couple models with different hyperparameters.
    for i in range(9, 11):

        # Creating the model with the best learning rate, determined above.
        model = model_generator(i, best_lr=best_lr)

        # Training the model and printing the time taken to train the model.
        metrics = model_trainer(model, f'model {i}', img_data_train,
                                img_labels_train, img_data_test,
                                img_labels_test, 256, 5,
                                False, False, False,
                                True)

        # Keeping track of the performance metrics.
        list_to_plot.append(metrics)

    # Plotting the performance metrics with the function defined above.
    # Also determining the optimal value for the momentum for the next step.
    best_mom = compare_parameters(list_to_plot, [0.5, 0.99],
                                  'Momentum', 'mom',
                                  ask=ask)

    # Initialising an empty list for keeping track of the performance metrics.
    list_to_plot = []

    # Looping over the next couple models with different hyperparameters.
    for i in range(11, 14):

        # Creating the model with the best learning rate, and momentum
        # determined above.
        model = model_generator(i, best_lr=best_lr, best_mom=best_mom)

        # Training the model and printing the time taken to train the model.
        metrics = model_trainer(model, f'model {i}', img_data_train,
                                img_labels_train, img_data_test,
                                img_labels_test, 256, 5,
                                False, False, False,
                                True)

        # Keeping track of the performance metrics.
        list_to_plot.append(metrics)

    # Plotting the performance metrics with the function defined above.
    # Also determining the optimal value for the drop out rate for the next
    # step.
    best_drop = compare_parameters(list_to_plot, [0.2, 0.5, 0.8],
                                   'Dropout', 'Do',
                                   ask=ask)

    # Determining the model number from the optimal drop out rate.
    drop_tuning = {11: 0.2, 12: 0.5, 13: 0.8}
    n = next(iter({i for i in drop_tuning if drop_tuning[i] == best_drop}))

    # Initialising the number of epoch values
    epoch_list = [5, 10, 20, 30]

    # Initialising an empty list for keeping track of the performance metrics.
    list_to_plot = []

    # Looping over the different hyperparameters for the number of epochs.
    for i in range(0, len(epoch_list)):

        # Creating the model with the best learning rate, and momentum
        # determined above.
        model = model_generator(n, best_lr=best_lr, best_mom=best_mom)

        # Training the model and printing the time taken to train the model.
        metrics = model_trainer(model, f'model {n}', img_data_train,
                                img_labels_train, img_data_test,
                                img_labels_test, 256, epoch_list[i],
                                False, False, False,
                                True)

        # Keeping track of the performance metrics.
        list_to_plot.append(metrics)

    # Plotting the performance metrics with the function defined above.
    # Also determining the optimal value for the number of epochs for the next
    # step.
    best_epoch = compare_parameters(list_to_plot, epoch_list,
                                    'Epoch', 'Ep',
                                    ask=ask)

    # Initialising the batch size values
    batch_list = [32, 64, 1024, 4069]

    # Initialising an empty list for keeping track of the performance metrics.
    list_to_plot = []

    # Looping over the different hyperparameters for the number of epochs.
    for i in range(0, len(batch_list)):

        # Creating the model with the best learning rate, and momentum
        # determined above.
        model = model_generator(n, best_lr=best_lr, best_mom=best_mom)

        # Training the model and printing the time taken to train the model.
        metrics = model_trainer(model, f'model {n}', img_data_train,
                                img_labels_train, img_data_test,
                                img_labels_test, batch_list[i], best_epoch,
                                False, False, False,
                                True)

        # Keeping track of the performance metrics.
        list_to_plot.append(metrics)

    # Plotting the performance metrics with the function defined above.
    # Also determining the optimal value for the batch size for the next step.
    best_batch = compare_parameters(list_to_plot, batch_list,
                                    'Batch size',
                                    'Bs', best_epoch,
                                    ask=ask)

    # Printing the optimal values for the hyperparameters looking solely at the
    # achieved training loss of the function (when ask = False). This often
    # gives poor results, so a manual tuning was performed (aks = True).
    print(f'The optimal learning rate is: {best_lr}')
    print(f'The optimal momentum is: {best_mom}')
    print(f'The optimal dropout rate is: {best_drop}')
    print(f'The optimal epoch amount is: {best_epoch}')
    print(f'The optimal batch size is: {best_batch}')

    # Showing all the plots of the performance metrics.
    plt.show()
