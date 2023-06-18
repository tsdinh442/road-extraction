import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.image import flip_left_right, flip_up_down, adjust_brightness
import matplotlib.pyplot as plt
import os


def sort_by_name(arr, split_at):
    return sorted([f for f in arr if f.split(split_at)[0].isdigit()], key=lambda x: int(x.split(split_at)[0]))


def load_data(dir_path):
    # dir_path = '../satellite-roads/train/'
    directory = os.listdir(dir_path)
    images = []
    masks = []

    for filename in directory:
        if filename.split('.')[1] == 'jpg':
            images.append(filename)
        elif filename.split('.')[1] == 'png':
            masks.append(filename)

    sorted_images = sort_by_name(images, '_')
    sorted_masks = sort_by_name(masks, '_')

    return np.array(sorted_images), np.array(sorted_masks)


def preprocess_data(root_path, sorted_images, sorted_masks, input_size, augmented=False):

    images = []
    masks = []
    for img_file, mask_file in zip(sorted_images, sorted_masks):
        img = load_img(root_path + img_file, target_size=input_size, color_mode='rgb')
        mask = load_img(root_path + mask_file, target_size=input_size, color_mode='grayscale')

        # Convert image and mask to arrays
        img_array = img_to_array(img)
        img_array = img_array / 255.0

        mask_array = img_to_array(mask, dtype=np.bool_)

        # Append images and masks to the lists
        images.append(img_array)
        masks.append(mask_array)

        if augmented:
            images.append(flip_left_right(img_array))
            masks.append(flip_left_right(mask_array))

            images.append(flip_up_down(img_array))
            masks.append(flip_up_down(mask_array))

    # Convert lists to numpy arrays
    images = np.array(images)
    masks = np.array(masks)

    return images, masks


def display_data(dir_path, image_paths, mask_paths):

    fig, axes = plt.subplots(5, 2, figsize=(10, 20))

    # Iterate over the image and mask pairs and display them in subplots
    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Load the image and mask using your preferred method
        image = plt.imread(dir_path + image_path)
        mask = plt.imread(dir_path + mask_path)

        # Plot the image and mask in the corresponding subplot
        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask)
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    return

