import cv2
import mediapipe as mp
import os

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
            
            # Check confidence
            if left_shoulder.visibility >= min_detection_confidence and right_shoulder.visibility >= min_detection_confidence:
                return left_shoulder, right_shoulder
    return None, None

def calculate_dynamic_width(image, left_shoulder, right_shoulder):
    """Calculate the width between shoulders based on detected keypoints."""
    _, width, _ = image.shape
    return int(abs(left_shoulder.x - right_shoulder.x) * width)

def calculate_cropping_boundary(image, overall_width, left_shoulder=None, right_shoulder=None):
    """Calculate cropping boundary based on shoulder landmarks."""
    height, width, _ = image.shape

    if left_shoulder and right_shoulder:
        # Calculate shoulder positions in pixels
        left_x = int(left_shoulder.x * width)
        right_x = int(right_shoulder.x * width)
        shoulder_width = right_x - left_x

        # Calculate cropping boundaries
        x_start = max(0, left_x - (overall_width - shoulder_width) // 2)
        x_end = min(width, x_start + overall_width)
        return (x_start, 0, x_end, height)

    # Landmarks not detected
    return None

def crop_with_boundary(image, boundary):
    """Crop the image based on dynamic boundary."""
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

def process_goal_frame(goal_frame_path, output_goal_frame_path, margin):
    """Process and crop the goal frame to determine cropping width."""
    goal_frame = load_image(goal_frame_path)
    left_shoulder, right_shoulder = detect_shoulders(goal_frame)

    if left_shoulder and right_shoulder:
        # Calculate cropping dimensions
        goal_width = calculate_dynamic_width(goal_frame, left_shoulder, right_shoulder)
        overall_width = goal_width + (2 * margin)
        boundary = calculate_cropping_boundary(goal_frame, overall_width, left_shoulder, right_shoulder)

        # Crop and save the goal frame
        cropped_goal_frame = crop_with_boundary(goal_frame, boundary)
        cropped_width, cropped_height = save_image(cropped_goal_frame, output_goal_frame_path)
        print(f"Saved {output_goal_frame_path} | Resolution: {cropped_width}x{cropped_height}")
        return overall_width, cropped_width, cropped_height
    else:
        raise ValueError("No shoulder landmarks detected in the goal frame.")

def process_input_image(input_image_path, output_image_path, overall_width, goal_frame_dimensions):
    """Process a single input image using the width calculated from the goal frame."""
    input_image = load_image(input_image_path)
    input_height, input_width, _ = input_image.shape

    # Check--- if input image matches goal frame dimensions
    if (input_width, input_height) == goal_frame_dimensions:
        print(f"Input image matches the goal frame dimensions. Retaining original image.")
        save_image(input_image, output_image_path)
        return input_width, input_height

    left_shoulder, right_shoulder = detect_shoulders(input_image)

    if left_shoulder and right_shoulder:
        boundary = calculate_cropping_boundary(input_image, overall_width, left_shoulder, right_shoulder)
        cropped_image = crop_with_boundary(input_image, boundary)
        if cropped_image is not None:
            cropped_width, cropped_height = save_image(cropped_image, output_image_path)
            print(f"Cropped image saved to {output_image_path} | Resolution: {cropped_width}x{cropped_height}")
            return cropped_width, cropped_height
    else:
        print(f"Landmarks not detected in {input_image_path}. Skipping cropping.")
        return None, None

if __name__ == "__main__":
    # Define paths and parameters
    goal_frame_path = "frame_00037.jpg"  # Path to the goal frame
    output_goal_frame_path = "output_goal_frame_cropped.jpg"  # Path to save cropped goal frame
    margin = 500  # Margin around shoulders

    # Process the goal frame
    overall_width, goal_frame_width, goal_frame_height = process_goal_frame(goal_frame_path, output_goal_frame_path, margin)
    print(f"Goal Frame Processed | Overall Width: {overall_width}")

    # Process the input image
    input_image_path = "output_goal_frame.jpg"  # Path to the input image
    output_image_path = "/final.jpg"  # Path to save cropped input image
    cropped_width, cropped_height = process_input_image(
        input_image_path, 
        output_image_path, 
        overall_width, 
        (goal_frame_width, goal_frame_height)
    )
