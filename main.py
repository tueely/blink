import cv2
import numpy as np

# Constants
NUM_SECTIONS = 12  # Number of vertical sections on the wall
SECTION_COLOR = (0, 255, 0)  # Green color for section boundaries
FONT = cv2.FONT_HERSHEY_SIMPLEX
MIN_OBJECT_RADIUS = 10  # Minimum radius to filter out small noise
MAX_OBJECT_RADIUS = 100  # Maximum radius to filter out large objects (e.g., body)
MOTION_HISTORY_THRESHOLD = 15  # Minimum number of frames to track motion history
SMOOTHING_FACTOR = 0.8  # Exponential smoothing factor for predictions
CONFIDENCE_THRESHOLD = 30  # Minimum distance to update the predicted section
GRAVITY = 9.81  # Gravitational acceleration (m/s^2)

# Kalman Filter for trajectory prediction
class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

    def predict(self, coord):
        measurement = np.array([[np.float32(coord[0])], [np.float32(coord[1])]])
        self.kf.correct(measurement)
        prediction = self.kf.predict()
        return int(prediction[0][0]), int(prediction[1][0])  # Extract scalar values

def draw_sections(frame, num_sections):
    """Draw vertical sections on the frame and label them."""
    height, width = frame.shape[:2]
    section_width = width // num_sections  # Width of each section
    
    # Draw vertical lines and label sections
    for i in range(num_sections + 1):  # Include both edges
        x = i * section_width
        cv2.line(frame, (x, 0), (x, height), SECTION_COLOR, 2)
        
        # Label sections (skip the last edge line)
        if i < num_sections:
            label_x = x + section_width // 2 - 10  # Center the label
            cv2.putText(frame, str(i + 1), (label_x, 30), FONT, 1, SECTION_COLOR, 2)
    
    return section_width

def detect_object(mask, min_radius, max_radius):
    """Detect the largest moving object within the size constraints."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_objects = []
    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        if min_radius < radius < max_radius:
            valid_objects.append((int(x), int(y), int(radius)))
    if valid_objects:
        # Return the object with the largest radius
        return max(valid_objects, key=lambda obj: obj[2])
    return None

def predict_section(x, section_width):
    """Predict which section the object lands in."""
    section = int(x // section_width + 1)
    return min(section, NUM_SECTIONS)  # Ensure section number doesn't exceed NUM_SECTIONS

def predict_trajectory(xs, ys, height):
    """
    Predict the landing point using physics-based trajectory modeling.
    Assumes the object follows a parabolic path under gravity.
    """
    try:
        # Fit a quadratic curve to the trajectory
        coeffs = np.polyfit(xs, ys, 2)  # Quadratic fit
        a, b, c = coeffs
        
        # Calculate the discriminant
        discriminant = b**2 - 4 * a * (c - height)
        if discriminant < 0:
            return None  # No real solution
        
        # Solve for x when y = height (bottom of the frame)
        landing_x1 = (-b + np.sqrt(discriminant)) / (2 * a)
        landing_x2 = (-b - np.sqrt(discriminant)) / (2 * a)
        
        # Choose the landing point that makes sense (positive x)
        landing_x = landing_x1 if landing_x1 >= 0 else landing_x2
        return landing_x
    except Exception:
        return None  # Fallback if prediction fails

def smooth_prediction(current, previous, smoothing_factor):
    """Apply exponential smoothing to stabilize predictions."""
    return int(smoothing_factor * current + (1 - smoothing_factor) * previous)

def main():
    # Initialize webcam at maximum resolution
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set max resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get original frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture initial frame.")
        return
    orig_height, orig_width = frame.shape[:2]
    
    # Initialize Background Subtractor
    background_subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
    
    # Initialize Kalman Filter
    kf = KalmanFilter()
    
    # Previous positions for trajectory tracking
    prev_positions = []
    predicted_section = None  # Current predicted section
    smoothed_landing_x = None  # Smoothed landing x-coordinate
    locked_prediction = False  # Whether the prediction is locked
    object_in_frame = False  # Whether an object is currently in the frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        height, width = frame.shape[:2]
        
        # Apply background subtraction to detect moving objects
        fg_mask = background_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.erode(fg_mask, None, iterations=2)
        fg_mask = cv2.dilate(fg_mask, None, iterations=2)
        
        # Draw sections on the wall
        section_width = draw_sections(frame, NUM_SECTIONS)
        
        # Detect the object
        object_info = detect_object(fg_mask, MIN_OBJECT_RADIUS, MAX_OBJECT_RADIUS)
        if object_info:
            object_in_frame = True
            x, y, radius = object_info
            prev_positions.append((x, y))  # Store the detected position
            
            # Predict the next position using Kalman Filter
            predicted_x, predicted_y = kf.predict((x, y))
            
            # Draw detected object and predicted position
            cv2.circle(frame, (x, y), radius, (0, 255, 255), 2)  # Detected object
            cv2.circle(frame, (predicted_x, predicted_y), 5, (255, 0, 0), -1)  # Predicted position
            
            # If enough positions are available, predict the landing section
            if len(prev_positions) > MOTION_HISTORY_THRESHOLD and not locked_prediction:
                xs, ys = zip(*prev_positions[-MOTION_HISTORY_THRESHOLD:])
                
                # Predict landing x-coordinate using physics-based trajectory modeling
                landing_x = predict_trajectory(xs, ys, height)
                
                if landing_x is not None:
                    # Smooth the landing x-coordinate
                    if smoothed_landing_x is None:
                        smoothed_landing_x = landing_x
                    else:
                        smoothed_landing_x = smooth_prediction(landing_x, smoothed_landing_x, SMOOTHING_FACTOR)
                    
                    # Predict the section based on smoothed landing x
                    new_section = predict_section(smoothed_landing_x, section_width)
                    
                    # Update predicted section only if the change is significant
                    if predicted_section is None or abs(new_section - predicted_section) > CONFIDENCE_THRESHOLD:
                        predicted_section = new_section
                    
                    # Lock the prediction once determined
                    locked_prediction = True
                
                # Display the predicted section
                if predicted_section is not None:
                    cv2.putText(frame, f"Predicted Section: {predicted_section}", (10, 30), FONT, 1, (255, 0, 0), 2)
        else:
            object_in_frame = False
        
        # Retain prediction after object leaves frame
        if not object_in_frame and locked_prediction:
            if predicted_section is not None:
                cv2.putText(frame, f"Predicted Section: {predicted_section}", (10, 30), FONT, 1, (255, 0, 0), 2)
        
        # Reset prediction lock if no object is detected for a long time
        if not object_in_frame and len(prev_positions) == 0:
            locked_prediction = False
            predicted_section = None
            smoothed_landing_x = None
        
        # Display the output
        cv2.imshow("Object Tracking and Section Mapping", frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
