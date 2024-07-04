# Bjorn De Busschere, Bjorn.DeBusschere@student.uantwerpen.be, 19/05/2024
# =============================================================================
# Imports
# =============================================================================

import tensorflow as tf
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Main import show_number, confusion_matrix, ROC, display_label
from sklearn.cluster import KMeans
from statistics import mean
from torchvision.transforms import v2
import torch
import keras
import pandas as pd


# =============================================================================
# Functions
# =============================================================================


def extract_images(folder_path):
    """
    Function for extracting labelled images from a folder structure.

    :param folder_path: directory path to the folder containing the images of
                        the numbers in a string or path format.

    :return: returns a list containing the images as arrays and list containing
             the labels as integers.
    """

    # Initial type check.
    if not isinstance(folder_path, (str, Path)):
        raise TypeError("folder_path must be a string or Path object")

    # Creating empty lists for adding images and labels.
    image_list = []
    label_list = []

    # Creating the path object.
    folder_path = Path(folder_path)

    # Testing if given path is valid.
    if not folder_path.is_dir():
        raise ValueError("folder_path must be a directory")

    # Loop over the elements in the given folder.
    for item in folder_path.iterdir():
        # If item is a directory, recursively call this function on the
        # sub-folders to extract all the images in the folder structure.
        if item.is_dir():
            images, labels = extract_images(item)
            # Adding the images and labels to the main list.
            image_list.extend(images)
            label_list.extend(labels)

        else:
            # Check if item is in the correct image format.
            if any(str(item).lower().endswith(ext)
                   for ext in {'.jpg', '.jpeg', '.png', '.tif', '.bmp'}):
                # Open the image and convert it to the correct format.
                image = Image.open(item)
                image = image.convert('L')
                # Convert the image to an array.
                image_array = np.array(image)
                # Inverting the image so the numbers have high pixel values and
                # the background low pixel values like in the MNIST dataset.
                image_array = 255 - image_array
                try:
                    # Extracting the label from the images name, we expect
                    # the numbers to be in the format label_something.
                    index = item.stem.find('_')
                    label = int(item.stem[index - 1])
                except TypeError:
                    # Raise error if label could not be extracted.
                    raise TypeError("invalid image name, label could not be "
                                    "extracted. make sure the name of the "
                                    "images is of the form: label_something.")

                # Appending the images and labels to the list.
                image_list.append(image_array)
                label_list.append(label)
            else:
                # Print error if the item is not an image.
                print(f"{item} is not an image file.")

    # Return the image and label list.
    return image_list, label_list


def normalize_shape(image, size=None, hist=False):
    """
    Function for resizing greyscale images of numbers on an even background
    into a square format.

    :param image: Image of the number in an array format.
    :param size: Integer specifying the final width and height of the square
                 image. Default is None, in which case size will be made the
                 biggest value between the width and the height.

    :param hist: boolean, when true will plot the histogram clustering as a
                 result of the k-mean clustering.

    :return: The image shaped in a square format.
    """

    # Type checks.
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")

    if size is not None:
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Size should be a positive integer or None")

    if not isinstance(hist, bool):
        raise ValueError("hist should be a boolean")

    # Flattening the image array for clustering.
    pixels = image.flatten()

    # Performing the KMeans clustering, we assume 2 clusters: background
    # and number.
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(pixels.reshape(-1, 1))

    # Getting the labels from the clustering.
    labels = kmeans.labels_

    # Getting the centers of the clusters.
    cluster_centers = kmeans.cluster_centers_

    # We assume there are more background pixels than pixels in the numbers,
    # and we use this to get the label of the background.
    if sum(1 for x in labels if x == 0) >= sum(1 for x in labels if x == 1):
        label_background = 0
        label_foreground = 1
    else:
        label_background = 1
        label_foreground = 0

    # Getting the background pixel values.
    background_pixels = pixels[labels == label_background]

    # Calculating the average pixel value in the background.
    background_value = int(mean(background_pixels))

    # Plotting the histogram of the clustering if specified.
    if hist:
        # Creating the figure
        plt.figure(figsize=(6, 6))

        # Plotting the background histogram.
        plt.hist(background_pixels, bins=15, color='b', label='Background')

        # Plotting the number histogram.
        foreground_pixels = pixels[labels == label_foreground]
        plt.hist(foreground_pixels, bins=15, color='r', label='Number')

        # Plotting the cluster centers on the image.
        plt.axvline(x=cluster_centers[0], color='g', linestyle='--',
                    label='Cluster Center (Background)')
        plt.axvline(x=cluster_centers[1], color='g', linestyle='--',
                    label='Cluster Center (Number)')

        # Formatting the plot.
        plt.xlabel('Pixel Intensities')
        plt.ylabel('Frequency')
        plt.title('Histogram of Pixel Intensities, classified with '
                  'K-means clustering.')
        plt.legend()

        # Showing the plot
        plt.show()

    # Getting the height and width of the image.
    height, width = image.shape

    # Transforming the image to a tensor.
    image = torch.tensor(image)
    image = image.unsqueeze(0)

    # If size is None, set size to the highest value between height and width.
    if size is None:
        size = max(height, width)

    # Determine whether to pad the height of the image or not.
    if height < size:
        padding_amount_height = size - height
    elif height < width:
        padding_amount_height = width - height
    else:
        padding_amount_height = 0

    # Padding the height of the image.
    for i in range(padding_amount_height):
        # For the first padding, check if the number is at the edge of the
        # array, if so pad with the average background intensity.
        if i == 0:
            if torch.any(image[:, -1] >
                         torch.max(torch.tensor(background_pixels))):
                image = v2.Pad(padding=(0, 0, 0, 1),
                               fill=background_value)(image)
            else:
                # If not, pad with the color of the pixels at the edge.
                image = v2.Pad(padding=(0, 0, 0, 1),
                               padding_mode='edge')(image)
        else:
            # For all other sequential paddings, pad with the pixel values of
            # the edge.
            image = v2.Pad(padding=(0, 0, 0, 1), padding_mode='edge')(image)

    # Determine whether to pad the width of the image or not.
    if width < size:
        padding_amount_width = size - width
    elif width < height:
        padding_amount_width = height - width
    else:
        padding_amount_width = 0

    # Padding the width of the image.
    for i in range(padding_amount_width):
        # For the first padding, check if the number is at the edge of the
        # array, if so pad with the average background intensity.
        if i == 0:
            if torch.any(image[:, :, -1] >
                         torch.max(torch.tensor(background_pixels))):
                print('yes')
                image = v2.Pad(padding=(0, 0, 1, 0),
                               fill=background_value)(image)
            else:
                # If not, pad with the color of the pixels at the edge.
                image = v2.Pad(padding=(0, 0, 1, 0),
                               padding_mode='edge')(image)
        else:
            # For all other sequential paddings, pad with the pixel values of
            # the edge.
            image = v2.Pad(padding=(0, 0, 1, 0), padding_mode='edge')(image)

    # If the image is bigger than the size, scale it down.
    if image.size(1) > size:
        image = v2.Resize(size=size)(image)

    # Return the image as an array.
    return image.squeeze().numpy()


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Importing the dataset from folder
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Initiating the folder path for the numbers.
    folder_path = r"C:\Bjorn\fysica Bach 3\programmeren voor fysici\Examenopdracht\my_numbers"

    # Extracting the numbers from the folder.
    image_list, labels = extract_images(folder_path)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Options for running the program.
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Options for running the main function.
    transformations_show = False
    test_DNN = True
    test_CNN = False
    exclude_numbers = False
    number_example = False

    # Main can only test one model at the time.
    if test_DNN and test_CNN:
        print('One at the time please.')
        test_DNN = True
        Test_CNN = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Showing reshaping and formatting of the image.
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Making a plot of the result of the normalize_shape function for some
    # images.
    if transformations_show:
        # Making the figure with subplots.
        fig, axes = plt.subplots(2, 5)

        # Adding the plot title.
        fig.suptitle('Comparison of images: before and after reshaping',
                     fontsize=16)

        # Plotting the original images.
        for i, ax in enumerate(axes[0]):
            # Showing the images.
            show_number(image_list[10 * i], labels[10 * i], lines=False,
                        ax=ax, show=False)

        # Plotting the results of normalize_shape.
        for i, ax in enumerate(axes[1]):
            # transforming the images.
            image = normalize_shape(image_list[10 * i], 32, hist=False)

            # Plotting the transformed images.
            show_number(image, labels[10 * i], lines=False, ax=ax, show=False)

        # Showing the images.
        plt.show()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Preprocessing the data.
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Exclude the numbers 1, 7 and 9 from the dataset, when specified.
    if exclude_numbers:

        # Loop over the labels starting from the last index.
        for i in range(len(labels)-1, -1, -1):

            # If the label is 1, 7 or 9 remove that instance in both the label
            # list and the image_list.
            if labels[i] == 1 or labels[i] == 7 or labels[i] == 9:
                del labels[i]
                del image_list[i]

    # Process the dataset of images.
    if test_DNN or test_CNN or number_example:
        # Setting the correct size for the images according to the model used.
        if test_DNN:
            size = 28
        if test_CNN or number_example:
            size = 32

        # initiating an empty list for the images
        image_array = []

        # Looping over all the images.
        for image in image_list:
            # Transforming all the images and putting them into the list.
            reshaped_image = normalize_shape(image, size, hist=False)
            image_array.append(reshaped_image)

        # Stack the list to make it an array.
        image_array = np.stack(image_array)
        print(image_array.shape)

        if number_example:
            display_label(image_array, labels, 5, 9, True)

        # Normalising the pixel values of the image array.
        image_array = np.expand_dims(image_array, -1).astype("float32") / 255.0

        # When testing the CNN model also transform the images to a tensor
        # format.
        if test_CNN:
            image_tensor = tf.convert_to_tensor(image_array)
            # making the greyscale image have 3 color channels.
            image_tensor = tf.image.grayscale_to_rgb(image_tensor)

        # Stacking the label list into an array.
        label_array = tf.stack(labels).numpy()

        # One hot encoding the labels.
        num_classes = 10
        image_label = keras.utils.to_categorical(label_array, num_classes)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # DNN model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if test_DNN:
        # Load the model from a stored Keras file.
        optimal_model = keras.models.load_model(
            r"C:\Bjorn\fysica Bach 3\programmeren voor fysici\Examenopdracht\optimal_model_trans3.keras")

        # Evaluating the model on the dataset.
        score = optimal_model.evaluate(image_array, image_label, verbose=0)

        # Printing a summary of the model.
        optimal_model.summary()

        # Printing the performance of the model.
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        # Creating the confusion matrix plot and getting the performance
        # metrics.
        precision, recall, f1 = confusion_matrix(optimal_model,
                                                 image_array, image_label,
                                                 show=False,
                                                 give_metrics=True)

        # Printing the performance metrics.
        df = pd.DataFrame({"Precision": precision, "Recall": recall,
                           "F1 Score": f1})
        print(df)

        # Creating the ROC curve plot for the model.
        ROC(optimal_model, image_array, image_label, False)

        # Showing all the plots.
        plt.show()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # CNN model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if test_CNN:
        # Load the model from a stored Keras file.
        model_Net = keras.models.load_model(
            r"C:\Bjorn\fysica Bach 3\programmeren voor fysici\Examenopdracht\model_Net_trans_final2.keras")

        # Evaluating the model on the dataset.
        score = model_Net.evaluate(image_tensor, image_label, verbose=0)

        # Printing a summary of the model.
        model_Net.summary()

        # Printing the performance of the model.
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        # Creating the confusion matrix plot and getting the performance
        # metrics.
        precision, recall, f1 = confusion_matrix(model_Net,
                                                 image_tensor, image_label,
                                                 show=False,
                                                 give_metrics=True)

        # Printing the performance metrics.
        df = pd.DataFrame({"Precision": precision, "Recall": recall,
                           "F1 Score": f1})
        print(df)

        # Creating the ROC curve plot for the model.
        ROC(model_Net, image_tensor, image_label, False)

        # Showing all the plots.
        plt.show()
