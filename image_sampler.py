import os
import random
import math
from PIL import Image

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_NEAREST = Image.NEAREST


def generate_random_crops(
    image_path: str,
    output_dir: str,
    output_name_prefix: str = "crop",
    num_samples: int = None ,
    index_shift: int = 0,
    allowed_x: int = None,
    allowed_y: int = None,
    allowed_width: int = None,
    allowed_height: int = None,
    target_size: tuple = (224, 224),
    gaussian_std_dev_factor: float = 2.0,
    min_center_distance_pixels: float = None,
    randomness: str = "gauss" 
) -> int:
    """
    Generates a batch of N randomly cropped and rotated images.
    Ensures a minimum distance between the centers of chosen crops.
    Args:
        image_path (str): Path to the source image.
        num_samples (int): The number (N) of random crops to generate.
        output_dir (str): The directory where cropped images will be saved.
        output_name_prefix (str, optional): Prefix for output filenames. Defaults to "crop".
        allowed_x (int, optional): Top-left x of allowed region. Defaults to 0.
        allowed_y (int, optional): Top-left y of allowed region. Defaults to 0.
        allowed_width (int, optional): Width of allowed region. Defaults to image width.
        allowed_height (int, optional): Height of allowed region. Defaults to image height.
        target_size (tuple, optional): (width, height) of output crops. Defaults to (224, 224).
        gaussian_std_dev_factor (float, optional): Factor for Gaussian std dev relative to
            half-width/height of allowed region. Defaults to 2.0.
        min_center_distance_pixels (float, optional): Minimum Euclidean distance between the
            centers of any two chosen crops (in original image coordinates).
            Defaults to target_size[0] / 4.0.
        randomness (str, optional): defaults to "gauss", otherwise uses uniform distribution. 
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' does not exist.")
        return
    if gaussian_std_dev_factor <= 0:
        print("Warning: gaussian_std_dev_factor must be positive. Defaulting to 2.0.")
        gaussian_std_dev_factor = 2.0

    target_w, target_h = target_size
    if min_center_distance_pixels is None:
        min_center_distance_pixels = target_w * 0.71 #multiplying by sqrt(2)/2 so that the center doesn't lie in a previously chosen patch
    
    # Ensure min_center_distance_pixels is not negative for squared comparison
    min_center_distance_pixels = max(0.0, min_center_distance_pixels)
    min_dist_sq = min_center_distance_pixels**2 # Use squared distance for efficiency

    try:
        img_orig_loaded = Image.open(image_path)
    except Exception as e:
        print(f"Error: Could not open image '{image_path}'. Exception: {e}")
        return

    img_for_processing = img_orig_loaded.copy()

    # --- Create initial mask ---
    if img_for_processing.mode == 'RGBA' or img_for_processing.mode == 'LA':
        mask_orig = img_for_processing.split()[-1]
    elif img_for_processing.mode == 'PA':
        mask_orig = img_for_processing.convert('RGBA').split()[-1]
    else:
        if img_for_processing.mode == 'P' or img_for_processing.mode == 'L':
            img_for_processing = img_for_processing.convert('RGB')
        elif 'A' not in img_for_processing.mode and img_for_processing.mode != 'RGB':
            img_for_processing = img_for_processing.convert('RGB')
        mask_orig = Image.new('L', img_for_processing.size, 255)

    # --- Normalize image mode for processing ---
    if img_for_processing.mode == 'P' or (img_for_processing.mode == 'L' and 'A' not in img_for_processing.mode):
        img_for_processing = img_for_processing.convert('RGB')
    elif 'A' not in img_for_processing.mode and img_for_processing.mode != 'RGB':
         img_for_processing = img_for_processing.convert('RGB')

    orig_w, orig_h = img_for_processing.size

    # --- Define allowed region boundaries and parameters for Gaussian ---
    ax_min = allowed_x if allowed_x is not None else 0
    ay_min = allowed_y if allowed_y is not None else 0
    aw = allowed_width if allowed_width is not None else orig_w
    ah = allowed_height if allowed_height is not None else orig_h

    ax_min = max(0, ax_min)
    ay_min = max(0, ay_min)
    aw = min(aw, orig_w - ax_min)
    ah = min(ah, orig_h - ay_min)

    if aw <= 0 or ah <= 0:
        print(f"Error: Allowed region has non-positive effective width ({aw}) or height ({ah}).")
        return
    if aw < 1e-6 or ah < 1e-6:
        print(f"Error: Allowed region width ({aw}) or height ({ah}) is too small for Gaussian sampling.")
        return

    ax_max = ax_min + aw
    ay_max = ay_min + ah
    
    mean_x = ax_min + aw / 2.0
    mean_y = ay_min + ah / 2.0
    
    sigma_x = (aw / 2.0) * gaussian_std_dev_factor
    sigma_y = (ah / 2.0) * gaussian_std_dev_factor
    sigma_x = max(sigma_x, 1e-6)
    sigma_y = max(sigma_y, 1e-6)

    os.makedirs(output_dir, exist_ok=True)
    
    chosen_centers_orig_coords = [] # Store (x,y) of chosen original centers
    generated_and_saved_count = 0
    max_attempts_per_sample_slot = 400 # May need more attempts with distance constraint


    done = False    
    sample_slot_idx = 0
    while not done:
        if sample_slot_idx%50==0: print(sample_slot_idx)
        current_attempts_for_slot = 0
        center_found_for_slot = False
        while current_attempts_for_slot < max_attempts_per_sample_slot and not center_found_for_slot:
            current_attempts_for_slot += 1

            # --- Step 1: Pick a center ---
            if randomness == "gauss":
                candidate_cx, candidate_cy = -1, -1 # Placeholder
                gaussian_attempts = 0
                max_gaussian_attempts = 100 # Prevent infinite loop if bounds are impossible for Gaussian
                while gaussian_attempts < max_gaussian_attempts:
                    gaussian_attempts +=1
                    intended_orig_cx_candidate = random.gauss(mean_x, sigma_x)
                    if ax_min <= intended_orig_cx_candidate <= ax_max:
                        candidate_cx = intended_orig_cx_candidate
                        break
                if candidate_cx == -1: # Failed to find x within bounds
                    if current_attempts_for_slot == 1: print("Debug: Failed to find valid Gaussian X for center.")
                    continue

                gaussian_attempts = 0
                while gaussian_attempts < max_gaussian_attempts:
                    gaussian_attempts += 1
                    intended_orig_cy_candidate = random.gauss(mean_y, sigma_y)
                    if ay_min <= intended_orig_cy_candidate <= ay_max:
                        candidate_cy = intended_orig_cy_candidate
                        break
                if candidate_cy == -1:  # Failed to find y within bounds
                    if current_attempts_for_slot == 1: print("Debug: Failed to find valid Gaussian Y for center.")
                    continue
            else:
                candidate_cx = random.uniform(ax_min, ax_max)
                candidate_cy = random.uniform(ay_min, ay_max)
            
            # --- Step 1.5: Check distance to previously chosen centers ---
            is_dist_ok = True
            if chosen_centers_orig_coords:
                for prev_cx, prev_cy in chosen_centers_orig_coords:
                    dist_sq = (candidate_cx - prev_cx)**2 + (candidate_cy - prev_cy)**2
                    if dist_sq < min_dist_sq:
#                        print("too close:", prev_cx,prev_cy, candidate_cx, candidate_cy)
                        is_dist_ok = False
                        break
            
            if not is_dist_ok:
                if current_attempts_for_slot % 100 == 0 : # Log occasionally if struggling
                    print(f"Debug: Slot {sample_slot_idx+1}, Attempt {current_attempts_for_slot}: center too close. Retrying center point.")
                continue # This attempt for the slot fails, try a new random center (+ angle implicitly)

            # If here, candidate_cx, candidate_cy are valid w.r.t Gaussian bounds and distance
            intended_orig_cx = candidate_cx
            intended_orig_cy = candidate_cy

            angle = random.uniform(0, 360)
            
            img_rotated = img_for_processing.rotate(angle, resample=RESAMPLE_BICUBIC, expand=True)
            mask_rotated = mask_orig.rotate(angle, resample=RESAMPLE_NEAREST, expand=True, fillcolor=0) 
            
            rot_w, rot_h = img_rotated.size

            if rot_w < target_w or rot_h < target_h:
                if current_attempts_for_slot % 50 == 0:
                    print(f"Note for slot {sample_slot_idx+1}: Rotated image ({rot_w}x{rot_h}) too small.")
                continue 

            orig_img_center_x, orig_img_center_y = orig_w / 2.0, orig_h / 2.0
            rotated_img_center_x, rotated_img_center_y = rot_w / 2.0, rot_h / 2.0
            
            vec_to_intended_center_x = intended_orig_cx - orig_img_center_x
            vec_to_intended_center_y = intended_orig_cy - orig_img_center_y

            rad_angle_transform = math.radians(angle)
            cos_a = math.cos(rad_angle_transform)
            sin_a = math.sin(rad_angle_transform)

            rot_vec_x = vec_to_intended_center_x * cos_a - vec_to_intended_center_y * sin_a
            rot_vec_y = vec_to_intended_center_x * sin_a + vec_to_intended_center_y * cos_a

            transformed_center_x_in_rotated = rotated_img_center_x + rot_vec_x
            transformed_center_y_in_rotated = rotated_img_center_y + rot_vec_y

            crop_tl_x = round(transformed_center_x_in_rotated - target_w / 2.0)
            crop_tl_y = round(transformed_center_y_in_rotated - target_h / 2.0)
            crop_br_x = crop_tl_x + target_w
            crop_br_y = crop_tl_y + target_h

            if not (0 <= crop_tl_x < rot_w and 0 <= crop_tl_y < rot_h and \
                    crop_br_x <= rot_w and crop_br_y <= rot_h):
                if current_attempts_for_slot % 50 == 0:
                     print(f"Debug: Slot {sample_slot_idx+1}, Attempt {current_attempts_for_slot}: crop out of rotated bounds.")
                continue

            final_crop_img = img_rotated.crop((crop_tl_x, crop_tl_y, crop_br_x, crop_br_y))
            final_crop_mask = mask_rotated.crop((crop_tl_x, crop_tl_y, crop_br_x, crop_br_y))

            mask_extrema = final_crop_mask.getextrema()
            if isinstance(mask_extrema, tuple) and len(mask_extrema) == 2: # Grayscale mask 'L'
                 min_mask_val = mask_extrema[0]
            elif isinstance(mask_extrema, list) and len(mask_extrema) > 0 and isinstance(mask_extrema[0], tuple): # Multi-band, e.g. from RGBA mask
                 min_mask_val = min(val[0] for val in mask_extrema) # Check min of all bands' min
            else: # Unknown mask format, assume not empty for safety (or handle specific cases)
                print(f"Warning: Unexpected mask extrema format: {mask_extrema}. Assuming valid crop.")
                min_mask_val = 255 # Default to valid if unknown

            if min_mask_val == 0: 
                if current_attempts_for_slot % 50 == 0:
                    print(f"Debug: Slot {sample_slot_idx+1}, Attempt {current_attempts_for_slot}: mask crop includes fill.")
                continue
            
            # --- All checks passed ---
            chosen_centers_orig_coords.append((intended_orig_cx, intended_orig_cy))
            generated_and_saved_count += 1
            output_filename = f"{output_name_prefix}_{generated_and_saved_count+index_shift}.png"
            output_filepath = os.path.join(output_dir, output_filename)
            final_crop_img.save(output_filepath)
            center_found_for_slot = True # Mark that this slot is filled
            break # Exit while current_attempts_for_slot loop for this sample_slot_idx
        
        if not center_found_for_slot:
            print(f"Warning: Could not generate a valid image for sample slot {sample_slot_idx + 1} "
                  f"within {max_attempts_per_sample_slot} attempts (constraints too tight?).")
            done = True
            break
        if num_samples is not None and generated_and_saved_count >= num_samples:
            done = True
            break
        sample_slot_idx += 1
    
    if num_samples is not None and generated_and_saved_count < num_samples:
        print(f"\nWarning: Only {generated_and_saved_count} out of {num_samples} desired samples were generated and saved.")
    else: print(f"\nSuccessfully generated and saved {generated_and_saved_count} samples in '{output_dir}'.")
    return generated_and_saved_count
        
def find_image_paths(folder_path):
    """
    Recursively finds all image files within a given folder and its subfolders.

    Args:
        folder_path (str): The absolute or relative path to the folder to search.

    Returns:
        list: A list of full paths to all found image files.
              Returns an empty list if the folder_path is invalid or contains no images.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The provided path '{folder_path}' is not a valid directory.")
        return []

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'}
    image_paths = []

    # os.walk() generates the file names in a directory tree
    # by walking the tree either top-down or bottom-up.
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file extension is in our set of image extensions
            if any(file.lower().endswith(ext) for ext in image_extensions):
                # Construct the full path to the file and add it to our list
                full_path = os.path.join(root, file)
                image_paths.append(full_path)

    return image_paths

if __name__ == '__main__':
    destination_folder = "test_images (separate source image)"
    source_images = find_image_paths("sources/drone_footage_test_only")
    print(f"Found {len(source_images)} source images")
    shift = 0
    for image in source_images:
        print(f"\n--- Generating crops from f{image} ---")
        generated = generate_random_crops(
            image_path=image,
            output_dir= destination_folder,
            output_name_prefix = "test_img",
            index_shift = shift,
            randomness = 'uniform',
            min_center_distance_pixels = 224
        )
        shift += generated
    print(f"\nCheck the '{destination_folder}' directory for the output images.")