import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Configuration
TIP_LANDMARKS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20
}

BUFFER_SIZE = 45  # Track last n frames
CONSECUTIVE_FRAMES = 2  # Required upward frames
SMOOTHING_WINDOW = 5    # Moving average window
DOWNWARD_TRAVEL_THRESH = 10

class FingerTracker:
    def __init__(self):
        self.trajectory = {
            'left': {finger: deque(maxlen=BUFFER_SIZE) for finger in TIP_LANDMARKS},
            'right': {finger: deque(maxlen=BUFFER_SIZE) for finger in TIP_LANDMARKS}
        }
        self.press_states = {
            'left': {finger: False for finger in TIP_LANDMARKS},
            'right': {finger: False for finger in TIP_LANDMARKS}
        }

    def _smooth_positions(self, positions):
        """Smooth both x and y coordinates separately"""
        if len(positions) < SMOOTHING_WINDOW:
            return positions
        x_smooth = np.convolve([p[0] for p in positions], 
                             np.ones(SMOOTHING_WINDOW)/SMOOTHING_WINDOW, 
                             mode='valid')
        y_smooth = np.convolve([p[1] for p in positions],
                             np.ones(SMOOTHING_WINDOW)/SMOOTHING_WINDOW,
                             mode='valid')
        return list(zip(x_smooth.astype(int), y_smooth.astype(int)))

    def update(self, hand_label, finger, position, DOWNWARD_TRAVEL_THRESH):
        """Update finger trajectory and detect press events"""
        buffer = self.trajectory[hand_label][finger]
        buffer.append(position)  # Store (x,y) tuple
        
        if len(buffer) < BUFFER_SIZE:
            return None

        # Get smoothed trajectory points
        smoothed = self._smooth_positions(list(buffer))
        
        # Need at least N+1 points to detect trend
        if len(smoothed) < CONSECUTIVE_FRAMES + 1:
            return None

        # Check upward movement in last N frames
        upward_count = 0
        for i in range(1, CONSECUTIVE_FRAMES + 1):
            current_y = smoothed[-i][1]
            previous_y = smoothed[-(i+1)][1]
            if current_y < previous_y:  # Moving upward (y decreases)
                upward_count += 1

        press_point = None
        if upward_count >= CONSECUTIVE_FRAMES:
            # Find lowest point in buffer (maximum y-value)
            y_values = [pos[1] for pos in buffer]
            lowest_idx = np.argmax(y_values)
            press_point = buffer[lowest_idx]
            
            # Calculate maximum downward travel by checking all points before lowest
            if lowest_idx > 0:
                # Find the minimum y-value (highest position) before lowest_idx
                min_y_before_lowest = min(y_values[:lowest_idx])
                
                # Calculate downward travel
                downward_travel = y_values[lowest_idx] - min_y_before_lowest
                
                # Only register press if downward travel exceeds threshold
                if downward_travel >= DOWNWARD_TRAVEL_THRESH:
                    # Reset tracking for new gesture
                    buffer.clear()
                    self.press_states[hand_label][finger] = True
                else:
                    # Not enough downward travel
                    self.press_states[hand_label][finger] = False
                    press_point = None
            else:
                # Lowest point is first in buffer, can't measure downward travel
                self.press_states[hand_label][finger] = False
                press_point = None
        else:
            self.press_states[hand_label][finger] = False

        return press_point

# Initialize components
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)
tracker = FingerTracker()

def finger_tracker(frame, hands, tracker, downward_travel_thresh=DOWNWARD_TRAVEL_THRESH):
    """Modified tracker with flip compensation"""
    # Flip frame horizontally for MediaPipe
    flipped_frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Process flipped frame
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    current_presses = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                            results.multi_handedness):
            hand_label = handedness.classification[0].label.lower()
            
            for finger, lm_idx in TIP_LANDMARKS.items():
                lm = hand_landmarks.landmark[lm_idx]
                # Convert to original frame coordinates
                x_flipped = int(lm.x * w)
                y = int(lm.y * h)
                # Compensate for horizontal flip
                x = w - x_flipped
                
                # Update tracker with corrected coordinates
                press_point = tracker.update(hand_label, finger, (x, y), downward_travel_thresh)
                finger_state = tracker.press_states[hand_label][finger]
                
                if finger_state and press_point:
                    current_presses.append(press_point)
                
                # Visual feedback on original frame
                color = (0, 0, 255) if finger_state else (0, 255, 0)
                cv2.circle(frame, (x, y), 12, color, cv2.FILLED)
    
    return frame, current_presses
