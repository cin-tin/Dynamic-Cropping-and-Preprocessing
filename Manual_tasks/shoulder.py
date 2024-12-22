import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # For visualization
pose = mp_pose.Pose(static_image_mode=True)  # Static mode for images

# Load your image
image_path = "frame_0009.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to detect pose landmarks
results = pose.process(image_rgb)

# Check if landmarks are detected
if results.pose_landmarks:
    print("Pose landmarks detected:")
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        print(f"Landmark {idx}: x={landmark.x}, y={landmark.y}, z={landmark.z}, visibility={landmark.visibility}")

    # Visualize the pose landmarks on the image
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, 
        results.pose_landmarks, 
        mp_pose.POSE_CONNECTIONS
    )
    
    # Display the annotated image
    cv2.imshow("Pose Landmarks", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No pose landmarks detected.")

# Release resources
pose.close()








# # Initialize Mediapipe Pose
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# # Load the image
# image_path = '/home/tenet/Desktop/ISL-NEW/crop/skipppoutput/cropped_frames/missed_frames/missed_frame_0001.jpg'  # Replace with your image file path
# image = cv2.imread(image_path)

# if image is None:
#     print("Error: Unable to load image. Check the path.")
# else:
#     # Set up the Pose model
#     pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

#     # Convert the image to RGB
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Process the image and get pose landmarks
#     results = pose.process(rgb_image)
    
#     # Check if landmarks are detected
#     if results.pose_landmarks:
#         # Draw landmarks on the image
#         annotated_image = image.copy()
#         mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
#         # Extract shoulder landmarks
#         landmarks = results.pose_landmarks.landmark
#         # left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
#         # right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
#         # Draw circles on shoulders
#         # left_shoulder_coords = (int(left_shoulder.x * image.shape[1]), int(left_shoulder.y * image.shape[0]))
#         # right_shoulder_coords = (int(right_shoulder.x * image.shape[1]), int(right_shoulder.y * image.shape[0]))
#         # cv2.circle(annotated_image, left_shoulder_coords, 10, (0, 255, 0), -1)
#         # cv2.circle(annotated_image, right_shoulder_coords, 10, (255, 0, 0), -1)
        
#         # Display coordinates on the image
#         # cv2.putText(annotated_image, f"Left Shoulder: {left_shoulder_coords}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#         # cv2.putText(annotated_image, f"Right Shoulder: {right_shoulder_coords}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
#         # Show the output
#         cv2.imshow('Pose Detection', annotated_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     else:
#         print("No pose landmarks detected.")
    
#     # Release resources
#     pose.close()
