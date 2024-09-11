import sys
import os
from tqdm import tqdm  # Import tqdm for progress bar
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import openslide
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import cv2
import math
from PIL import Image
import torch
from transformers import ViTImageProcessor
import torchvision.transforms.functional as F
import stainlib
from tqdm import tqdm  # Import tqdm for progress bar
import torch.nn as nn
from .utils import *
from .binary_classifier import BinaryClassifier
import math



class ExtractLevel3PatchesModule():
    def __init__(self, level, patch_size, custom_patch_sizes, custom_step_sizes, model_level_3_path="", 
                target_patch="", data_path=""):
        super().__init__()
        
        self.data_path = data_path
        self.final_level = level
        self.custom_patch_sizes = custom_patch_sizes
        self.custom_step_sizes = custom_step_sizes
        self.patch_size = patch_size
        ## Load models
        
        if len(model_level_3_path) > 0:
            self.model_level_3 = BinaryClassifier.load_from_checkpoint(model_level_3_path)
            self.model_level_3.eval()
        else:
            self.model_level_3 = None
            
        ## Init transforms
        self.normalizer, self.image_processor = self.load_and_preprocess_target_image(target_patch)
        
        self.sigmoid = nn.Sigmoid()
        
        
        
    def update_patches_info(self, patches_info, new_patches_info):
        for k in patches_info.keys():
            patches_info[k].extend(new_patches_info[k])
        return patches_info
        
    def load_and_preprocess_target_image(self, path_of_target_patch):
        """
        Loads normalizer, preprocessor.
        """
        normalizer = stainlib.normalization.normalizer.ReinhardStainNormalizer()
        target = np.array(Image.open(path_of_target_patch))

        stainlib.utils.stain_utils.LuminosityStandardizer.standardize(target)
        normalizer.fit(target)

        image_processor = ViTImageProcessor.from_pretrained(
            "google/vit-large-patch16-224", local_files_only=True
        )
        return normalizer, image_processor
    
    def classify_relevant_patches_image(self, image_file_name):
        image_path = os.path.join(self.data_path, image_file_name)
        slide = openslide.OpenSlide(image_path)
        width, height = slide.level_dimensions[0]
        print(width, height)
        level_5_width, level_5_height = slide.level_dimensions[5]

        patches_info = self.do_pyramid_level(
            slide,
            image_file_name,
            width,
            height,
            5,
            0,
            0,
            level_5_width,
            level_5_height
        )

        return patches_info
    
    def calculate_total_patches(self, slide, start_level, custom_step_sizes, custom_patch_sizes):
        total_patches = 0
        for lev in range(start_level, 5 + 1):
            width, height = slide.level_dimensions[lev]

            # Adjust step size and patch size based on the current level
            current_step_size = custom_step_sizes[lev]
            current_patch_size = custom_patch_sizes[lev]

            patches_x = max(
                1, (width - current_patch_size + current_step_size) // current_step_size
            )
            patches_y = max(
                1, (height - current_patch_size + current_step_size) // current_step_size
            )
            total_patches += patches_x * patches_y * 2
        return total_patches
    
    def do_pyramid_level(self,
        slide,
        image_file_name,
        width,
        height,
        current_level,
        base_x,
        base_y,
        search_width,
        search_height):
        is_final_level = self.final_level == current_level
        patches_info = {"prediction": [],
            "x1": [],
            "y1": [],
            "x2": [],
            "y2": [],          
            "filename": [],
            "level": [],
            "image_width" : [],
            "image_height" : [],
                        
        }

        # Define step size and patch size adjustments based on current level
        current_step_size = self.custom_step_sizes[current_level]
        current_patch_size = self.custom_patch_sizes[current_level]
        for y in range(
            base_y,
            base_y + search_height - current_patch_size + current_step_size,
            current_step_size,
        ):
            if y > base_y + search_height - current_patch_size:
                patch_y = base_y + search_height - current_patch_size
            else:
                patch_y = y
            for x in range(
                base_x,
                base_x + search_width - current_patch_size + current_step_size,
                current_step_size,
            ):
                if x > base_x + search_width - current_patch_size:
                    patch_x = base_x + search_width - current_patch_size
                else:
                    patch_x = x

                # Ensure the patch is within bounds
                if (
                    patch_x < 0
                    or patch_y < 0
                    or patch_x * (2 ** (current_level))
                    + current_patch_size * (2 ** (current_level))
                    - 1
                    > width
                    or patch_y * (2 ** (current_level))
                    + current_patch_size * (2 ** (current_level))
                    - 1
                    > height
                ):
                    continue
                patch = show_image_patch(
                    slide,
                    patch_x,
                    patch_y,
                    current_patch_size,
                    current_patch_size,
                    current_level,
                )
                if patch is None:
                    # print(f"Failed to extract patch: x={patch_x} y={patch_y} current_patch_size={current_patch_size} level={current_level}")
                    continue  # Skip patches that failed to extract
                else:
                    patch = patch.convert("RGB")
                    is_background, patch = self.decide_which_function_for_background(
                        patch, current_level, is_final_level
                    )

                    if not is_background:
                        if is_final_level:
                            new_patches_info = {
                                "prediction": [None],
                                "x1": [patch_x],
                                "y1": [patch_y],
                                "x2": [patch_x+current_patch_size],
                                "y2": [patch_y+current_patch_size],
                                "filename": [image_file_name],
                                "level": [self.final_level],
                                "image_width":[slide.level_dimensions[self.final_level][0]],
                                "image_height":[slide.level_dimensions[self.final_level][1]],
                            }
                            patches_info = self.update_patches_info(patches_info, new_patches_info)
                        else:
                            new_base_x, new_base_y = patch_x * 2, patch_y * 2
                            new_search_width = self.custom_patch_sizes[current_level - 1] * 2
                            new_search_height = (
                                self.custom_patch_sizes[current_level - 1] * 2
                            )
                            next_level = current_level - 1

                            new_patches_info = self.do_pyramid_level(
                                slide,
                                image_file_name,
                                width,
                                height,
                                next_level,
                                new_base_x,
                                new_base_y,
                                new_search_width,
                                new_search_height)

                            patches_info = self.update_patches_info(patches_info, new_patches_info)

        return patches_info
    
    def count_patches_in_region(self,
        current_patch_size,
        current_level):
        
        # Calculate the scaling factor between current level and target level
        scaling_factor = 2 ** (current_level - self.final_level)

        # Calculate the dimensions of the region at the target level
        target_patch_size = current_patch_size * scaling_factor

        # Calculate the step size at the target level
        target_step_size = self.custom_step_sizes[self.final_level]

        # Calculate the number of patches in the x and y directions
        patches_x = max(
            1,
            (target_patch_size - self.custom_patch_sizes[self.final_level] + target_step_size)
            // target_step_size,
        )
        patches_y = max(
            1,
            (target_patch_size - self.custom_patch_sizes[self.final_level] + target_step_size)
            // target_step_size,
        )

        # Return the total number of patches
        return patches_x * patches_y

    def decide_which_function_for_background(self, patch, level, final_level=True):
        _, _, result_image_pil = remove_artifacts(patch)
        if final_level:
            if is_patch_white(result_image_pil, 0.50, 254):
                return True, result_image_pil
            return detect_blurriness_and_contrast(patch), result_image_pil
        if level==5:
            return is_patch_white(result_image_pil, 0.95, 254), result_image_pil
        return is_patch_white(result_image_pil, 0.90, 254), result_image_pil


    


    def patch_preprocessing(self, patch):
        """
        Pre-processes the patch, the way the training patches were preprocessed.
        Input: Path to target patch, fitted normalizer
        Returns: Tensor
        """
        
        try:
            patch = patch.convert("RGB")
        except ValueError:
            pass  # If already RGB, do nothing

        patch = F.adjust_contrast(patch, contrast_factor=1.5)
        patch = np.array(patch)
        patch = stainlib.utils.stain_utils.LuminosityStandardizer.standardize(patch)
        patch = self.normalizer.transform(patch)
        tensor = self.image_processor(
            patch, return_tensors="pt", do_rescale=True, do_normalize=False
        )
        pixel_values = tensor["pixel_values"]

        return pixel_values  # Patch is returned as a tensor
    
    def make_predictions(self, model, preprocessed_patch):
        # Convert preprocessed_patch to tensor if it's not already
        if not isinstance(preprocessed_patch, torch.Tensor):
            preprocessed_patch = torch.tensor(preprocessed_patch).unsqueeze(0)  # Add batch dimension

            # Get the probability from the model
        with torch.no_grad():  # No need to track gradients
            probability = model(preprocessed_patch)
            probability = probability.view(-1)
            probability = self.sigmoid(probability)
            prediction_binary = (probability > 0.50).int().item()

        return prediction_binary, probability.item()



    def process_specific_patch(self,
        x,
        y,
        slide,
        level,
        model,
        width,
        height,
    ):
        if (
            x < 0
            or y < 0
            or x * (2 ** (level)) + self.patch_size * (2 ** (level)) - 1 > width * (2 ** (level))
            or y * (2 ** (level)) + self.patch_size * (2 ** (level)) - 1 > height * (2 ** (level))
        ):
            pass
        else:
            patch = show_image_patch(slide, x, y, self.patch_size, self.patch_size, level)
            if patch is None:
                return None
            else:
                patch = patch.convert("RGB")
                is_background, patch = self.decide_which_function_for_background(
                    patch, level, True
                )
                if not is_background:
                    preprocessed_patch = self.patch_preprocessing(patch)
                    _, prediction = self.make_predictions(
                        model,
                        preprocessed_patch)
                    return prediction
                else:
                    return 0.0
        return None




def detect_blurriness_and_contrast(image, blur_threshold=60, contrast_threshold=0.35):
    # Read the image using OpenCV
    if isinstance(image, Image.Image):
        image = np.array(image)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute the Laplacian of the image and then the variance
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    # Determine if the image is blurry
    is_blurry = laplacian_var < blur_threshold
    return is_blurry


