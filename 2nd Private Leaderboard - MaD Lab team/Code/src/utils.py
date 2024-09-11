import os
import json
import openslide
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_filename_count(file_name_counts_bb_path, image_name):
    # Read the dictionary from the JSON file
    with open(file_name_counts_bb_path, 'r') as json_file:
        filename_counts = json.load(json_file)

    # Function to get the count for a specific filename
    count = filename_counts.get(image_name, 0)
    print(f"The filename '{image_name}' appears {count} times.")
    return count


def scaling_images(slide, x, y, width, height, level):
    """
    Scales the coordinates and dimensions of a patch to the base level (level 0) of a whole slide image.

    Parameters:
    slide (openslide.OpenSlide): An OpenSlide object representing the whole slide image.
    x (int): The x-coordinate of the top-left corner of the patch at the specified level.
    y (int): The y-coordinate of the top-left corner of the patch at the specified level.
    width (int): The width of the patch at the specified level.
    height (int): The height of the patch at the specified level.
    level (int): The magnification level (0-5, where 0 is the base level and 5 is the most zoomed-out).

    Returns:
    tuple: Scaled coordinates and dimensions (base_x, base_y, base_width, base_height) at the base level.
    """
 
    scale_factor = slide.level_downsamples[level]
    base_x = int(x * scale_factor)
    base_y = int(y * scale_factor)
    base_width = int(width * scale_factor)
    base_height = int(height * scale_factor)
    return base_x, base_y, base_width, base_height


def resize_bboxes(bboxes, old_size, new_size):
    """
    Scale the bounding boxes accordingly.
    
    :param bboxes: List of bounding boxes, each in the format [x_min, x_max, y_min, y_max].
    :param display_size: Tuple (new_width, new_height) indicating the display size for the image.
    :param size_0: Tuple (width, height) indicating the size 0 for the image.
    :return: Resized image and scaled bounding boxes.
    """
    
    # Scale bounding boxes
    x_scale = new_size[0] / old_size[0]
    y_scale = new_size[1] / old_size[1]
    
    scaled_bboxes = []
    for bbox in bboxes:
        x_min, x_max, y_min, y_max = bbox
        new_x_min = int(x_min * x_scale)
        new_y_min = int(y_min * y_scale)
        new_x_max = int(x_max * x_scale)
        new_y_max = int(y_max * y_scale)
        scaled_bboxes.append([new_x_min, new_x_max, new_y_min, new_y_max])
    
    return scaled_bboxes



def check_patch_not_within_bounds(slide, base_width, base_height, base_x, base_y):
    """
    Checks if the requested patch is within the bounds of the base level (level 0) of a whole slide image.

    Parameters:
    slide (openslide.OpenSlide): An OpenSlide object representing the whole slide image.
    base_height (int): The height of the patch at the base level.
    base_width (int): The width of the patch at the base level.
    base_x (int): The x-coordinate of the top-left corner of the patch at the base level.
    base_y (int): The y-coordinate of the top-left corner of the patch at the base level.

    Returns:
    int: Returns 1 if the patch is out of bounds, otherwise returns 0.
    """
    base_dims = slide.level_dimensions[0]
    if base_x + base_width > base_dims[0] or base_y + base_height > base_dims[1]:
        print('x', base_x + base_width, base_dims[0])
        print('y', base_y + base_height, base_dims[1])
        print("Requested patch is out of bounds.")
        return 1
    return 0


def show_image_patch(slide, x, y, width, height, level):
    """
    Displays a specified patch from a pathology image file at a given magnification level.

    Parameters:
    tif_file (str): The path to the TIFF image file.
    x (int): The x-coordinate of the top-left corner of the patch.
    y (int): The y-coordinate of the top-left corner of the patch.
    width (int): The width of the patch.
    height (int): The height of the patch.
    level (int): The magnification level (0-5, where 5 is the most zoomed-out).

    Returns:
    None
    """

    # Scale the coordinates and dimensions to the base level (level 0)
    base_x, base_y, base_width, base_height = scaling_images(slide, x, y, width, height, level)

    #Check if patch is within bounds
    if check_patch_not_within_bounds(slide,base_width, base_height, base_x, base_y):
        return
    
    # Read a region of the slide (patch) at the base level
    patch = slide.read_region((base_x, base_y), level, (width, height))

   # display_patch(slide, patch, width, height, level, x, y)
    return patch

def is_patch_white(patch, threshold_white=0.95, brightness_threshold=200):
    # Convert patch to grayscale
    gray_patch = patch.convert('L')
    # Convert to numpy array
    gray_array = np.array(gray_patch)
    # Calculate the number of white pixels
    white_pixels = np.sum(gray_array >= brightness_threshold)
    total_pixels = gray_array.size
    # Calculate the proportion of white pixels
    white_ratio = white_pixels / total_pixels
    # Check if the patch is more than 90% white
    return white_ratio >= threshold_white



def remove_colors(image, lower_color_bounds, upper_color_bounds):
    """
    Remove specified color ranges from the image.

    Parameters:
    image (numpy array): The input image in OpenCV format.
    lower_color_bounds (list of numpy arrays): The lower bounds for colors to be removed.
    upper_color_bounds (list of numpy arrays): The upper bounds for colors to be removed.

    Returns:
    numpy array: The image with specified colors removed.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], dtype="uint8")

    for lower, upper in zip(lower_color_bounds, upper_color_bounds):
        color_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, color_mask)

    result = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    return result, mask

def highlight_artifacts(original_image, processed_image):
    """
    Highlight artifacts by subtracting processed image from the original image.

    Parameters:
    original_image (numpy array): The original image.
    processed_image (numpy array): The processed image with colors removed.

    Returns:
    numpy array: The image highlighting the artifacts.
    """
    difference = cv2.absdiff(original_image, processed_image)
    _, highlighted = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
    return highlighted

def remove_artifacts(patch):
    """
    Process a pathology image patch to remove artifacts by:
    1. Detecting regions of interest (ROIs) based on color.
    2. Highlighting artifacts.
    3. Keeping only the ROIs in the original image while making the rest white.

    Example Usage:
        image_id = '2qj5MlLLBT_a.tif'
        image_path = os.path.join('data/images', image_id)
        slide = openslide.OpenSlide(image_path)
        patch = show_image_patch(slide, 2000, 63, 5000, 2000, 3)
        patch = patch.convert('RGB')
        patch, highlighted_image, result_image_pil = remove_artifacts(patch)
        plot_artifact_removal(patch, highlighted_image, result_image_pil)


    Parameters:
    patch (PIL Image): The input pathology image patch.

    Returns:
    tuple: The original patch, highlighted artifacts image, and the final image with ROIs kept and non-ROIs made white.
    The patch: 'result_image_pil' is the patch without the artifacts.

    Note:
    The kernel size used in the dilation process is a possible hyperparameter that can be adjusted
    to change the margin around the regions of interest. A larger kernel will result in a larger margin.
    We introduced this kernel in the first place to make sure that colors that are not specifically pink/purple/blue, 
    but are also present in the ROI, are also represented.

    """
    # Convert PIL image to OpenCV format
    original_image = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)

    # Define the range for pink, blue, and purple colors in HSV
    lower_bounds = [np.array([120, 20, 70]), np.array([240, 20, 70]), np.array([160, 20, 70])]
    upper_bounds = [np.array([180, 255, 255]), np.array([270, 255, 255]), np.array([300, 255, 255])]

    # Remove the specified colors
    processed_image, mask = remove_colors(original_image, lower_bounds, upper_bounds)

    # Highlight the artifacts
    highlighted_image = highlight_artifacts(original_image, processed_image)

    # Create a mask from the highlighted artifacts
    highlighted_image_gray = cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(highlighted_image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to keep regions of interest
    roi_mask = np.zeros_like(highlighted_image_gray)
    cv2.drawContours(roi_mask, contours, -1, (255), thickness=cv2.FILLED)

    # Dilate the mask to include some margin around the regions of interest
    """
    1.	Kernel Size Hyperparameter:
        •	The kernel size used in the dilation process is a critical hyperparameter that affects how much the regions of interest (ROIs) are expanded.
        •	A kernel is a small, square matrix (in our case, 15x15 pixels) that is used to “slide” over the image. Wherever the kernel encounters a white    pixel (part of the ROI), it turns the entire area covered by the kernel white.
        •	The current kernel size is 15x15 pixels, and we apply it twice (iterations=2). This means that each white pixel in the ROI will expand by 15 pixels in all directions, effectively growing the ROI.
    2.	Purpose of the Kernel:
        •	The kernel is used to ensure that colors which are not specifically pink, purple, or blue, but are present in the ROIs, are also represented. This is important because the ROIs might contain colors that are close to the specified ones but not exact matches.
        •	By using a larger kernel, you can increase the margin around the ROIs, ensuring that even slightly different colors in the surrounding area are included in the ROI.
    3.	Adjusting the Kernel Size:
        •	You can adjust the kernel size to change the margin around the ROIs. A larger kernel will result in a larger margin, meaning the ROIs will expand more.
        •	For example, if you change the kernel size from 15x15 to 20x20, the dilation process will expand the ROIs by 20 pixels in all directions instead of 15 pixels.
    """
    kernel = np.ones((15, 15), np.uint8)
    roi_mask_dilated = cv2.dilate(roi_mask, kernel, iterations=2)

    # Apply the mask to the original image
    result_image = original_image.copy()
    result_image[roi_mask_dilated == 0] = [255, 255, 255]  # Make non-ROI regions white

    # Convert back to PIL format for display
    result_image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

    return patch, highlighted_image, result_image_pil

