# from ultralytics import YOLO
# import cv2

# # Load your trained model
# model = YOLO('models/pose/best.pt')

# # Run inference
# results = model('2.jpg')

# # Get annotated image
# annotated_frame = results[0].plot()

# cv2.imwrite('2_out.jpg', annotated_frame)




import cv2
from ultralytics import YOLO
import numpy as np
import random

# Load your trained model
model = YOLO('models/pose/best.pt')

# Read input image (BGR format for OpenCV)
img_bgr = cv2.imread('2.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Run inference
results = model(img_rgb)

# Process results
r = results[0]
if r.keypoints is not None:
    for i in range(len(r.boxes)):
        # Get keypoints for current detection
        keypoints = r.keypoints.data[i].cpu().numpy()
        
        # Ensure we have at least 2 keypoints
        if keypoints.shape[0] >= 2:
            # Extract first two keypoints (assumed to be belt endpoints)
            pt1 = (int(keypoints[0][0]), int(keypoints[0][1]))
            pt2 = (int(keypoints[1][0]), int(keypoints[1][1]))
            
            # Calculate Euclidean distance (pixel distance)
            distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            
            # Print coordinates and distance to terminal
            print(f"({pt1[0]}, {pt1[1]}) ({pt2[0]}, {pt2[1]}) {distance:.2f}")
            
            # Generate a random color for this specific pair
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Draw keypoints with the unique color
            cv2.circle(img_bgr, pt1, 5, color, -1)
            cv2.circle(img_bgr, pt2, 5, color, -1)
            
            # Add distance text at midpoint
            text = f"{distance:.2f} px"
            text_pos = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.putText(img_bgr, text, text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save processed image
cv2.imwrite('2_out.jpg', img_bgr)