# Bjorn De Busschere, Bjorn.DeBusschere@student.uantwerpen.be, 19/05/2024
# =============================================================================
# Imports
# =============================================================================

import numpy as np
import torch
from torchvision.transforms import v2
import torchvision
import torchvision.transforms.functional
import cv2
import math

# =============================================================================
# Class
# =============================================================================


class ImageTransformer:
    """
    Transformer class that contains methods for transforming grayscale
    images of handwritten numbers in the MNIST dataset. The possible
    transformations are:
        - translating
        - scaling
        - rotating
        - changing perspective
        - adding Gaussian noise
        - adding gradient
        - blurring and colorjittering
        - motion blurring

    This class also contains a method for applying all transformations to a
    batch of images with a certain probability.
    """
    def __init__(self):
        """ innit function for the ImageTransformer class"""
        pass

    def translate(self, image, pixels=0, direction='vertical', random=False,
                  suppress_error=False):
        """
        Method for translating the number in the image.

        :param image: array containing the image
        :param pixels: integer containing the amount of pixels to move over.
        :param direction: either 'vertical' or 'horizontal' specifying the
                          direction to move in.
        :param random: Boolean if True, the method will perform the translation
                       with random parameters
        :param suppress_error: Boolean if True, will suppress the error
                               messages from moving the image over the edge.

        :return: array containing the moved image
        """

        # Initial type checks.
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")

        if not isinstance(pixels, int):
            raise TypeError("Pixels must be an integer")

        if direction not in ['vertical', 'horizontal']:
            raise ValueError("Direction must be 'vertical' or 'horizontal'")

        if not isinstance(random, bool):
            raise TypeError("Random must be a boolean")

        if not isinstance(suppress_error, bool):
            raise TypeError("Suppress_error must be a boolean")

        # if random is True, define random parameters to run the method with.
        if random:
            pixels = np.random.randint(-5, 6)
            direction = np.random.choice(['vertical', 'horizontal'])

        # Translation in the vertical direction.
        if direction == 'vertical':
            # Loop over the amount of pixels given.
            for _ in range(abs(pixels)):
                # Sign of pixels gives the direction to move in.
                if pixels > 0:
                    # Checking if border pixels values are empty.
                    if not image[0].any():
                        # Moving the images over buy 1 row.
                        first_row = image[0].copy()
                        image[:-1] = image[1:]
                        image[-1] = first_row
                    else:
                        # If number has reached the edge, break the loop.
                        if not suppress_error:
                            # print error if not suppressed.
                            print("Can't move image more pixels, "
                                  "edge reached.")
                        break
                else:
                    # Checking if border pixels values are empty.
                    if not image[-1].any():
                        # Moving the images over buy 1 row.
                        last_row = image[-1].copy()
                        image[1:] = image[:-1]
                        image[0] = last_row
                    else:
                        # If number has reached the edge, break the loop.
                        if not suppress_error:
                            # print error if not suppressed.
                            print("Can't move image more pixels, "
                                  "edge reached.")
                        break

        # Translation in the horizontal direction.
        if direction == 'horizontal':
            # Loop over the amount of pixels given.
            for _ in range(abs(pixels)):
                # Sign of pixels gives the direction to move in.
                if pixels < 0:
                    # Checking if border pixels values are empty.
                    if not image[:, 0].any():
                        # Moving the images over buy 1 column.
                        first_column = image[:, 0].copy()
                        image[:, :-1] = image[:, 1:]
                        image[:, -1] = first_column
                    else:
                        # If number has reached the edge, break the loop.
                        if not suppress_error:
                            # print error if not suppressed.
                            print("Can't move image more pixels, "
                                  "edge reached.")
                        break
                else:
                    # Checking if border pixels values are empty.
                    if not image[:, -1].any():
                        # Moving the images over buy 1 column.
                        last_column = image[:, -1].copy()
                        image[:, 1:] = image[:, :-1]
                        image[:, 0] = last_column
                    else:
                        # If number has reached the edge, break the loop.
                        if not suppress_error:
                            # print error if not suppressed.
                            print("Can't move image more pixels, "
                                  "edge reached.")
                        break

        # Return the translated image.
        return image

    def translate_torch(self, image, pixels=0, direction='vertical',
                        random=False, suppress_error=False):
        """
        Method for translating the number in the image.

        :param image: array containing the image
        :param pixels: integer containing the amount of pixels to move over.
        :param direction: either 'vertical' or 'horizontal' specifying the
                          direction to move in.
        :param random: Boolean if True, the method will perform the translation
                       with random parameters
        :param suppress_error: Boolean if True, will suppress the error
                               messages from moving the image over the edge.

        :return: array containing the moved image
        """

        # Initial type checks.
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")

        if not isinstance(pixels, int):
            raise TypeError("Pixels must be an integer")

        if direction not in ['vertical', 'horizontal']:
            raise ValueError("Direction must be 'vertical' or 'horizontal'")

        if not isinstance(random, bool):
            raise TypeError("Random must be a boolean")

        if not isinstance(suppress_error, bool):
            raise TypeError("Suppress_error must be a boolean")

        height, width = image.shape

        # Transforming the array to a torch tensor.
        image_tensor = torch.tensor(image)
        image_tensor = image_tensor.unsqueeze(0)

        # if random is True, define random parameters to run the method with.
        if random:
            pixels = np.random.randint(-5, 6)
            direction = np.random.choice(['vertical', 'horizontal'])
        # Translation in the vertical direction.
        if direction == 'vertical':
            # Loop over the amount of pixels given.
            for _ in range(abs(pixels)):
                # Sign of pixels gives the direction to move in.
                if pixels < 0:
                    # Checking if border pixels values are empty.
                    if torch.all(image_tensor[:, -1] == 0):
                        # Add padding on the opposite side.
                        image_tensor = v2.Pad(padding=
                                              (0, 1, 0, 0))(image_tensor)

                        # Crop the image so the number is moved in the frame.
                        image_tensor = (torchvision.transforms.v2.functional
                                        .crop(image_tensor,
                                              0,
                                              0,
                                              height,
                                              width))
                    else:
                        # If number has reached the edge, break the loop.
                        if not suppress_error:
                            # print error if not suppressed.
                            print("Can't move image more pixels, "
                                  "edge reached.")
                        break
                else:
                    # Checking if border pixels values are empty.
                    if torch.all(image_tensor[:, 0] == 0):
                        # Add padding on the opposite side.
                        image_tensor = v2.Pad(padding=
                                              (0, 0, 0, 1))(image_tensor)
                        # Crop the image so the number is moved in the frame.
                        image_tensor = (torchvision.transforms.v2.functional
                                        .crop(image_tensor,
                                              1,
                                              0,
                                              height,
                                              width))
                    else:
                        # If number has reached the edge, break the loop.
                        if not suppress_error:
                            # print error if not suppressed.
                            print("Can't move image more pixels,"
                                  " edge reached.")
                        break

        # Translation in the horizontal direction.
        if direction == 'horizontal':
            # Loop over the amount of pixels given.
            for _ in range(abs(pixels)):
                # Sign of pixels gives the direction to move in.
                if pixels < 0:
                    # Checking if border pixels values are empty.
                    if torch.all(image_tensor[:, :, 0] == 0):
                        # Add padding on the opposite side.
                        image_tensor = v2.Pad(padding=
                                              (0, 0, 1, 0))(image_tensor)
                        # Crop the image so the number is moved in the frame.
                        image_tensor = (torchvision.transforms.v2.functional
                                        .crop(image_tensor,
                                              0,
                                              1,
                                              height,
                                              width))
                    else:
                        # If number has reached the edge, break the loop.
                        if not suppress_error:
                            # print error if not suppressed.
                            print("Can't move image more pixels, "
                                  "edge reached.")
                        break
                else:
                    # Checking if border pixels values are empty.
                    if torch.all(image_tensor[:, :, -1] == 0):
                        # Add padding on the opposite side.
                        image_tensor = v2.Pad(padding=
                                              (1, 0, 0, 0))(image_tensor)
                        # Crop the image so the number is moved in the frame.
                        image_tensor = (torchvision.transforms.v2.functional
                                        .crop(image_tensor,
                                              0,
                                              0,
                                              height,
                                              width))
                    else:
                        # If number has reached the edge, break the loop.
                        if not suppress_error:
                            # print error if not suppressed.
                            print("Can't move image more pixels, "
                                  "edge reached.")
                        break

        # Return the translated image as a numpy array.
        return image_tensor.squeeze().numpy()

    def scaling(self, image, pixels=0, zoom='in', random=False,
                suppress_error=False):
        """
        Method for scaling the number in the image.

        :param image: array containing the image
        :param pixels: amount of 'pixels' to zoom out or in for, e.g. zooming
                       in for 1 pixel makes a 5x5 image 7x7 and crops it
                       back to 5x5.
        :param zoom: either 'in' or 'out' specifying how to scale.
        :param random: Boolean if True, the method will perform the scaling
                       with random parameters
        :param suppress_error: Boolean if True, will suppress the error
                               messages from scaling the image out of frame.

        :return: array containing the scaled image
        """

        # Initial type checks.
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")

        if not isinstance(pixels, int):
            raise TypeError("pixels must be an integer")

        if zoom not in ['in', 'out']:
            raise ValueError("Zoom must be 'in' or 'out'")

        if not isinstance(random, bool):
            raise TypeError("random must be a boolean")

        if not isinstance(suppress_error, bool):
            raise TypeError("suppress_error must be a boolean")

        # Transforming the array to a torch tensor.
        image_tensor = torch.tensor(image)
        image_tensor = image_tensor.unsqueeze(0)

        # if random is True, define random parameters to run the method with.
        if random:
            pixels = np.random.randint(0, 5)
            zoom = np.random.choice(['in', 'out'])

        height, width = image.shape

        # Zooming in.
        if zoom == 'in':
            # Loop over the amount of pixels given.
            for _ in range(pixels):
                # Checking if border pixels are empty.
                if (torch.all(image_tensor[:, 0] == 0)
                        and torch.all(image_tensor[:, -1] == 0)
                        and torch.all(image_tensor[:, :, 0] == 0)
                        and torch.all(image_tensor[:, :, -1] == 0)):
                    # Scaling the image to be height+2 x width+2.
                    image_tensor = v2.Resize(size=(height+2, width+2))(image_tensor)
                    # Cropping the image back to 28x28 effectively zooming in.
                    image_tensor = (torchvision.transforms.v2.functional.
                                    crop(image_tensor,
                                         1,
                                         1,
                                         height,
                                         width))
                else:
                    # If number has reached the edge, break the loop.
                    if not suppress_error:
                        # print error if not suppressed.
                        print("Can't zoom in more on image, edge reached.")
                    break

        # Zooming out.
        if zoom == 'out':
            # Loop over the amount of pixels given.
            for _ in range(pixels):
                # Scaling the image to be height-2 x width-2.
                image_tensor = v2.Resize(size=(height-2, width-2))(image_tensor)
                # Padding the image with 1 empty layer, effectively zooming out
                image_tensor = v2.Pad(padding=1)(image_tensor)

        # Returning the image as an array.
        return image_tensor.squeeze().numpy()

    def random_perspective(self, image):
        """
        Method that changes the perspective of the number in the image
        randomly.

        :param image: array containing the image

        :return: array containing the transformed image
        """

        # Initial type checks.
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")

        # Transforming the array to a torch tensor.
        image_tensor = torch.tensor(image)
        image_tensor = image_tensor.unsqueeze(0)

        # Changing the perspective of the image.
        image_tensor = v2.RandomPerspective(distortion_scale=0.5,
                                            p=1.0)(image_tensor)

        # Returning the image as an array.
        return image_tensor.squeeze().numpy()

    def rotate(self, image, angle=0, random=False):
        """
        Method for rotating the number in the image.

        :param image: array containing the image

        :param angle: integer specifying the amount f degrees to rotate over.

        :param random: Boolean if True, the method will perform the rotating
                       with random parameters

        :return: array containing the rotated image
        """

        # Initial type checks.
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")

        if not isinstance(angle, int):
            raise TypeError("angle must be an integer")

        if not isinstance(random, bool):
            raise TypeError("random must be a boolean")

        # if random is True, define random parameters to run the method with.
        if random:
            angle = np.random.randint(0, 360)

        # Transforming the array to a torch tensor.
        image_tensor = torch.tensor(image)
        image_tensor = image_tensor.unsqueeze(0)

        # Rotating the image.
        image_tensor = (torchvision.transforms.v2.functional.
                        rotate(image_tensor, angle))

        # Returning the image as an array.
        return image_tensor.squeeze().numpy()

    def random_noise(self, image, mean=35, dev=30):
        """
        Method for adding Gaussian noise to the image.

        :param image: array containing the image

        :param mean: integer or float specifying the mean of the Gaussian
                     distribution to generate the noise.

        :param dev: integer or float specifying the deviation of the Gaussian
                    distribution to generate the noise.

        :return: array containing the noisy image
        """

        # Initial type checks.
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")

        if not isinstance(mean, (int, float)):
            raise TypeError("mean must be an integer or float")

        if not isinstance(dev, (int, float)):
            raise TypeError("dev must be an integer or float")

        height, width = image.shape

        # Creating a random noise array.
        random_array = np.random.normal(mean, dev, (height, width))

        # Adding the array to the image.
        image = np.add(image, random_array)

        # Making sure the image stays between 0 and 255.
        image = np.clip(image, 0, 255).astype(int)

        # Returning the image.
        return image

    def gradient(self, image, min_val=0, max_val=100, angle=0, random=False):
        """
        Method for adding a gradient onto the image.

        :param image: array containing the image

        :param min_val: integer specifying the minimal grayscale value of the
                        gradient
        :param max_val: integer specifying the maximal grayscale value of the
                        gradient
        :param angle: integer specifying the amount of degrees the gradient
                      should be rotated over?

        :param random: Boolean if True, the method will add the gradient
                       with random parameters

        :return: array containing the image with gradient
        """

        # Initial type checks.
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")

        if not isinstance(min_val, int):
            raise TypeError("min_val must be an integer")

        if not isinstance(max_val, int):
            raise TypeError("max_val must be an integer")

        if not isinstance(angle, int):
            raise TypeError("angle must be an integer")

        if not isinstance(random, bool):
            raise TypeError("random must be a boolean")

        # if random is True, define random parameters to run the method with.
        if random:
            min_val = np.random.randint(0, 50)
            max_val = np.random.randint(55, 160)
            angle = np.random.randint(0, 360)

        # Transforming the array to a torch tensor.
        image_tensor = torch.tensor(image)
        image_tensor = image_tensor.unsqueeze(0)

        height, width = image.shape

        size = math.ceil(np.sqrt(height**2 + width**2))

        # Creating the gradient array.
        gradient = np.tile(np.linspace(min_val, max_val, size),
                           (size, 1))

        # Changing the array to a tensor.
        gradient_tensor = torch.tensor(gradient)
        gradient_tensor = gradient_tensor.unsqueeze(0)

        # Rotate the gradient tensor.
        gradient_tensor = (torchvision.transforms.v2.functional
                           .rotate(gradient_tensor, angle))

        # Centercrop the gradient so it has the shape height x width.
        gradient_tensor = (torchvision.transforms
                           .CenterCrop((height, width))(gradient_tensor))

        # Round the gradient tensor so it only contains integers.
        rounded_gradient_tensor = torch.round(gradient_tensor)

        # Add the gradient to the original image, clamping it at 255.
        image_tensor = torch.clamp(image_tensor + rounded_gradient_tensor,
                                   max=255)

        # Return the image as an array.
        return image_tensor.squeeze().numpy().astype(np.uint8)

    def image_noise(self, image, brightness=0.4, contrast=0.4, blur=False):
        """
        Method to perform a colorjitter to the image and/or blur the image.

        :param image: array containing the image

        :param brightness: integer or float specifying the brightness of the
                           colorjitter.

        :param contrast: integer or float specifying the contrast of the
                         colorjitter.

        :param blur: Boolean, if true will blur the image

        :return: array containing the transformed image.
        """

        # Initial type checks.
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array")

        if not isinstance(brightness, (int, float)):
            raise TypeError("brightness must be an integer or float")

        if not isinstance(contrast, (int, float)):
            raise TypeError("contrast must be an integer or float")

        if not isinstance(blur, bool):
            raise TypeError("blur must be a boolean")

        # Transforming the array to a torch tensor.
        image_tensor = torch.tensor(image)
        image_tensor = image_tensor.unsqueeze(0)

        # Performing the random color jitter.
        image_tensor = v2.ColorJitter(brightness=brightness,
                                      contrast=contrast)(image_tensor)

        # If specified, perform the blurring.
        if blur:
            image_tensor = v2.GaussianBlur(kernel_size=(3, 3),
                                           sigma=(0.1, 1))(image_tensor)

        # Return the image as an array.
        return image_tensor.squeeze().numpy()

    def invert(self, image):
        """
        Method for inverting the pixel values of the image

        :param image: array containing the image

        :return: array containing the inverted image.
        """

        # Initial type checks.
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a 28x28 numpy array")

        # invert the pixel values.
        image = 255 - image

        # Return the image.
        return image

    def motion_blur(self, image, kern_size=5, angle=0, random=False):
        """
        Method for adding motion blur to the image.

        :param image: array containing the image

        :param kern_size: integer specifying the kernel size

        :param angle: integer specifying the angle to perform the motion blur
                      in

        :param random: Boolean if True, the method will add the motion blur
                       with random parameters

        :return: array containing the image with motion blur.
        """

        # Initial type checks.
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a 28x28 numpy array")

        if not isinstance(kern_size, int):
            raise TypeError("kern_size must be an integer")

        if not isinstance(angle, int):
            raise TypeError("angle must be an integer")

        if not isinstance(random, bool):
            raise TypeError("random must be a boolean")

        # if random is True, define random parameters to run the method with.
        if random:
            kern_size = np.random.randint(4, 6)
            angle = np.random.randint(0, 360)

        # Creating a kernel for the motion blur.
        kernel = np.zeros((kern_size, kern_size))
        kernel[:, int((kern_size - 1) / 2)] = np.ones(kern_size)/kern_size
        # Rotating the kernel to determine the direction of the blur.
        kernel = self.rotate(kernel, angle=angle, random=False)

        # Convert to float32 for processing.
        image = image.astype(np.float32)
        # Convert grayscale to BGR for processing the images with the kernel.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # performing the convolution with the kernel.
        image_blurred = cv2.filter2D(image, -1, kernel)

        # Convert BGR to grayscale
        image_blurred = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2GRAY)
        # Clip and convert back to uint8
        image_blurred = (np.clip(image_blurred, 0, 255)
                         .astype(np.uint8))

        # Returning the image.
        return image_blurred

    def random_batch_transform(self, batch, prob_translate=0.8,
                               prob_scaling=0.8, prob_perspective=0.4,
                               prob_rotate=0.9, prob_noise=0.9,
                               prob_grad=0.75, prob_image_noise=0.8,
                               prob_inv=0.4, prob_motion_blur=0.3):
        """
        Method for performing all the transformation to the image with a
        certain probability. Can also process batches of images.

        :param batch: np.array containing all the  images

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
                                 probability of the blur/colorjitter
                                 transformation.

        :param prob_inv: float less or equal to 1 specifying the
                         probability of the inversion transformation.

        :param prob_motion_blur: float less or equal to 1 specifying the
                                 probability of the motion blur transformation.

        :return: the transformed batch as a np.array.
        """

        # Initial type checks.
        if not isinstance(batch, np.ndarray):
            raise TypeError("batch must be a numpy array")

        if batch.ndim != 3:
            raise ValueError("batch must be a 3D numpy array")

        if not all(0 <= p <= 1 for p in [prob_translate, prob_scaling,
                                         prob_perspective,
                                         prob_rotate, prob_noise,
                                         prob_grad, prob_image_noise,
                                         prob_inv, prob_motion_blur]):
            raise ValueError("probabilities must be between 0 and 1")

        # Initialising an empty list.
        transformed_images = []

        # Printing batch info.
        print(f'Begin transforming {len(batch)} images')

        # Looping over every image in the batch and performing the
        # transformations with a certain probability.
        for i, image in enumerate(batch, start=1):
            if np.random.uniform(0, 1) < prob_translate:
                image = self.translate_torch(image, random=True, suppress_error=True)
            if np.random.uniform(0, 1) < prob_scaling:
                image = self.scaling(image, random=True, suppress_error=True)
            if np.random.uniform(0, 1) < prob_perspective:
                image = self.random_perspective(image)
            if np.random.uniform(0, 1) < prob_rotate:
                image = self.rotate(image, random=True)
            if np.random.uniform(0, 1) < prob_grad:
                image = self.gradient(image, random=True)
            if np.random.uniform(0, 1) < prob_image_noise:
                image = self.image_noise(image, blur=True)
            if np.random.uniform(0, 1) < prob_motion_blur:
                image = self.motion_blur(image, random=True)
            if np.random.uniform(0, 1) < prob_noise:
                image = self.random_noise(image, mean=20, dev=25)
            if np.random.uniform(0, 1) < prob_inv:
                image = self.invert(image)

            transformed_images.append(image)

            # Printing the progress.
            if i % 1000 == 0:
                print(f'processed {i}/{len(batch)} images')

        # Returning the transformed batch of images.
        return np.array(transformed_images)
