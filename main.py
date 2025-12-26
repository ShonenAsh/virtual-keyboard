import cv2
import numpy as np
from keyboard_tracker import keyboard_map, point_in_region
from finger_tracker import finger_tracker, hands, tracker, DOWNWARD_TRAVEL_THRESH
# from pynput.keyboard import Controller
#
# keyboard = Controller()
#
def debug_visualization(keyboard_snip, transformed_points, mapped_regions_snip):
    """Create visualization of transformed points and key regions"""
    debug_img = keyboard_snip.copy()
    
    # Draw all region boundaries
    for region, corners in mapped_regions_snip.items():
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
        center = np.mean(corners, axis=0).astype(int)
        cv2.putText(debug_img, region, tuple(center), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Draw transformed points with coordinates
    for i, (x, y) in enumerate(transformed_points):
        cv2.circle(debug_img, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(debug_img, f"{i}:({x:.1f},{y:.1f})", (int(x)+10, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return debug_img

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    key_presses = []
    frame, perspective_matrix, keyboard_snip, mapped_regions_snip = keyboard_map(frame)
    frame, current_presses = finger_tracker(frame, hands, tracker, DOWNWARD_TRAVEL_THRESH)

    if mapped_regions_snip:
        cv2.imshow("Keyboard Detection", keyboard_snip)

    if current_presses and perspective_matrix is not None:
        # Convert to proper input format for perspectiveTransform
        input_points = np.array([current_presses], dtype=np.float32)
        
        try:
            # Perform perspective transformation
            transformed_points = cv2.perspectiveTransform(input_points, perspective_matrix)
            transformed_points = transformed_points[0]  # Remove batch dimension
            
            # Create debug visualization
            debug_img = debug_visualization(keyboard_snip, transformed_points, mapped_regions_snip)
            cv2.imshow("Debug View", debug_img)
            
            # Check each transformed point against regions
            for point in transformed_points:
                pressed_region = point_in_region(mapped_regions_snip, tuple(point))
                if pressed_region:
                    key_presses.append(pressed_region)
                    
        except Exception as e:
            print(f"Transformation error: {str(e)}")

    if key_presses:
        print("Detected presses:", key_presses)
    # for key in key_presses:
    #     keyboard.press(key)
    #     keyboard.release(key)
    #

    cv2.imshow("Main View", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
