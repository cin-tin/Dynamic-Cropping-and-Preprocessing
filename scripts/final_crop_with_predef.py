import cv2
import mediapipe as mp
import os

# Predefined cropping dimensions
PREDEFINED_WIDTH = 1291  # Width in pixels
PREDEFINED_HEIGHT = 1080  # Height in pixels
MARGIN = 500  # Margin around shoulders

def load_image(image_path):
    """Load an image from the given path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    return image

def detect_shoulders(image, min_detection_confidence=0.5):
    """Detect shoulder keypoints using Mediapipe with a confidence threshold."""
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        result = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            left_shoulder = result.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = result.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER]
            
            if left_shoulder.visibility >= min_detection_confidence and right_shoulder.visibility >= min_detection_confidence:
                return left_shoulder, right_shoulder
    return None, None

def calculate_dynamic_width(image, left_shoulder, right_shoulder):
    """Calculate the width between shoulders based on detected keypoints."""
    _, width, _ = image.shape
    return int(abs(left_shoulder.x - right_shoulder.x) * width)

def calculate_cropping_boundary(image, predefined_width, predefined_height, margin, left_shoulder=None, right_shoulder=None):
    """Calculate cropping boundary using predefined dimensions and optional margin."""
    height, width, _ = image.shape

    if left_shoulder and right_shoulder:
        # Calculate shoulder positions in pixels
        left_x = int(left_shoulder.x * width)
        right_x = int(right_shoulder.x * width)
        shoulder_width = right_x - left_x

        # Add margin to shoulder width
        overall_width = max(predefined_width, shoulder_width + (2 * margin))

        # Ensure cropping is symmetric around shoulder midpoint
        center_x = (left_x + right_x) // 2
        x_start = max(0, center_x - overall_width // 2)
        x_end = min(width, x_start + overall_width)
        
        # Vertical cropping based on predefined height
        y_top = max(0, (height - predefined_height) // 2)
        y_bottom = min(height, y_top + predefined_height)
        return (x_start, y_top, x_end, y_bottom), left_x - x_start, x_end - right_x

    # No shoulders detected
    return None, None, None

def crop_with_boundary(image, boundary):
    """Crop the image based on calculated boundary."""
    if boundary:
        x_start, y_top, x_end, y_bottom = boundary
        return image[y_top:y_bottom, x_start:x_end]
    return None

def save_image(image, output_path):
    """Save the cropped image to the specified path."""
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if image is not None:
        cv2.imwrite(output_path, image)
        height, width, _ = image.shape
        return width, height
    return None, None

def process_image(input_image_path, output_image_path, predefined_width, predefined_height, margin):
    """Process a single input image using predefined cropping dimensions and margin."""
    input_image = load_image(input_image_path)
    left_shoulder, right_shoulder = detect_shoulders(input_image)

    if left_shoulder and right_shoulder:
        # Calculate cropping boundary and margins
        boundary, left_margin, right_margin = calculate_cropping_boundary(
            input_image, predefined_width, predefined_height, margin, left_shoulder, right_shoulder
        )
        cropped_image = crop_with_boundary(input_image, boundary)
        if cropped_image is not None:
            cropped_width, cropped_height = save_image(cropped_image, output_image_path)
            print(f"Cropped image saved to {output_image_path} | Resolution: {cropped_width}x{cropped_height}")
            # print(f"Margins calculated: Left Margin = {left_margin}px, Right Margin = {right_margin}px")
            return cropped_width, cropped_height
    else:
        # Shoulders not detected
        print(f"Shoulders not detected in {input_image_path}. Skipping cropping.")
        return None, None

if __name__ == "__main__":
    # Input and output image paths
    input_image_path = "/frame_00037.jpg"  # Path to the input image
    output_image_path = "cropped_redef.jpg"  # Path to save cropped image

    # Process the input image
    cropped_width, cropped_height = process_image(
        input_image_path, 
        output_image_path, 
        PREDEFINED_WIDTH, 
        PREDEFINED_HEIGHT, 
        MARGIN
    )
