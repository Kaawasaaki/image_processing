import cv2
import numpy as np
import os
import sys
from datetime import datetime

# Create directory for filters if it doesn't exist
if not os.path.exists('filters'):
    os.makedirs('filters')
    print("Created 'filters' directory. Please add filter PNG images there.")
    print("Filters should have transparent backgrounds.")
    sys.exit()

# Check if there are any filters in the directory
filters = [f for f in os.listdir('filters') if f.endswith('.png')]
if not filters:
    print("No filters found in 'filters' directory!")
    print("Please add PNG images with transparent backgrounds.")
    sys.exit()

# Load face cascade and facial landmark detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Try to load facial landmarks detector if available
try:
    # Check if we have access to facial landmarks detection
    import dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    USE_DLIB = True
    print("Using dlib for advanced facial landmark detection!")
    print("Make sure you have the shape_predictor_68_face_landmarks.dat file")
except:
    USE_DLIB = False
    print("Dlib not available. Using basic face detection only.")
    print("For better results, install dlib and download the shape_predictor_68_face_landmarks.dat file")

# Function to overlay filter on frame
def overlay_filter(frame, filter_img, x, y, w, h, filter_type):
    # Read the filter image with alpha channel
    filter_img = cv2.imread(os.path.join('filters', filter_img), cv2.IMREAD_UNCHANGED)
    
    # Resize filter based on face size
    if filter_type == "glasses":
        # For glasses, make them appropriately sized and positioned
        filter_width = int(w * 1.1)
        filter_height = int(h * 0.3)
        y_offset = int(y + h * 0.25)  # Position at eye level
    elif filter_type == "hat":
        # For hats, make them wider and place above the face
        filter_width = int(w * 1.4)
        filter_height = int(h * 0.8)
        y_offset = int(y - h * 0.55)  # Position above the face
    else:  # "full" covers the whole face
        filter_width = w
        filter_height = h
        y_offset = y
        
    x_offset = int(x - (filter_width - w) / 2)  # Center horizontally
    
    # Ensure coordinates are within frame boundaries
    if x_offset < 0: x_offset = 0
    if y_offset < 0: y_offset = 0
    
    # Resize filter
    try:
        filter_img = cv2.resize(filter_img, (filter_width, filter_height))
    except:
        return frame  # Return original frame if resize fails
    
    # Get dimensions
    h_filter, w_filter = filter_img.shape[:2]
    
    # Check if filter goes beyond frame boundaries
    if y_offset + h_filter > frame.shape[0] or x_offset + w_filter > frame.shape[1]:
        # Crop filter to fit within frame
        h_part = min(h_filter, frame.shape[0] - y_offset)
        w_part = min(w_filter, frame.shape[1] - x_offset)
        
        if h_part <= 0 or w_part <= 0:
            return frame  # Filter doesn't fit in frame at all
            
        filter_img = filter_img[:h_part, :w_part]
        h_filter, w_filter = filter_img.shape[:2]
    
    # Get region of interest
    roi = frame[y_offset:y_offset+h_filter, x_offset:x_offset+w_filter]
    
    # Check if ROI is valid
    if roi.shape[0] == 0 or roi.shape[1] == 0 or roi.shape[:2] != filter_img.shape[:2]:
        return frame  # Return original if ROI is invalid
    
    # Create mask from alpha channel
    alpha = filter_img[:, :, 3] / 255.0
    alpha = np.stack([alpha, alpha, alpha], axis=2)
    
    # Combine filter with frame
    try:
        frame[y_offset:y_offset+h_filter, x_offset:x_offset+w_filter] = \
            (1 - alpha) * roi + alpha * filter_img[:, :, :3]
    except:
        pass  # If blending fails, just return the original frame
        
    return frame

def add_text_overlay(frame, text, position=(30, 30), font_scale=0.8, color=(255, 255, 255)):
    """Add text overlay to the frame"""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    return frame

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Check for filters directory
    if not os.path.exists('filters'):
        os.makedirs('filters')
        print("Created 'filters' directory. Please run filter_creator.py first to generate filters.")
        cap.release()
        return
    
    # Get filter files
    filters = [f for f in os.listdir('filters') if f.endswith('.png')]
    if not filters:
        print("No filter images found in 'filters' directory.")
        print("Please run filter_creator.py first to generate the required filters.")
        cap.release()
        return
    
    # Initialize filter types dictionary
    filter_types = {"glasses": [], "full": []}
    
    # Our specified filters
    target_filters = {
        "dog": None,
        "demon": None,
        "sunglasses": None,
        "rabbit": None
    }
    
    # Populate filter types
    for f in filters:
        f_lower = f.lower()
        
        # Categorize by type
        if "glasses" in f_lower or "sunglasses" in f_lower:
            filter_types["glasses"].append(f)
        else:
            filter_types["full"].append(f)
            
        # Match our target filters
        if "dog" in f_lower:
            target_filters["dog"] = f
        elif "demon" in f_lower:
            target_filters["demon"] = f
        elif "sunglasses" in f_lower:
            target_filters["sunglasses"] = f
        elif "rabbit" in f_lower:
            target_filters["rabbit"] = f
    
    # Check if we have all our target filters
    missing_filters = [name for name, f in target_filters.items() if f is None]
    if missing_filters:
        print(f"Warning: Some requested filters are missing: {', '.join(missing_filters)}")
        print("Please run filter_creator.py to generate these filters.")
    
    # Create a custom filter order based on the specified filters
    custom_filters = [f for f in target_filters.values() if f is not None]
    
    # If we don't have any of our target filters, use whatever filters we have
    if not custom_filters:
        custom_filters = filters
    
    # Set initial filter
    current_filter_index = 0
    current_filter = custom_filters[current_filter_index]
    
    # Determine filter type of current filter
    current_filter_type = "full"  # Default
    if any(word in current_filter.lower() for word in ["glasses", "sunglasses"]):
        current_filter_type = "glasses"
    
    # Initialize variables
    take_snapshot = False
    show_help = True
    processing_active = True
    
    print("\n=== Snapchat-Style Filter App ===")
    print("Controls:")
    print("  SPACE - Take a snapshot")
    print("  F     - Change filter")
    print("  T     - Change filter type (hat, glasses, full)")
    print("  P     - Pause/resume filter processing")
    print("  H     - Toggle help text")
    print("  Q/ESC - Quit")
    
    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Flip frame horizontally for selfie view
        frame = cv2.flip(frame, 1)
        original = frame.copy()
        
        if processing_active:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Apply filter to each detected face
            for (x, y, w, h) in faces:
                # Optional: Draw rectangle around face (for debugging)
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Apply the current filter
                frame = overlay_filter(frame, current_filter, x, y, w, h, current_filter_type)
        
        # Display help text
        if show_help:
            info_text = [
                f"Filter: {current_filter}",
                f"Type: {current_filter_type}",
                "SPACE: Snapshot | F: Change Filter",
                "T: Change Type | P: Pause/Resume",
                "H: Hide Help | Q/ESC: Quit"
            ]
            for i, text in enumerate(info_text):
                add_text_overlay(frame, text, (20, 30 + i * 30))
        
        # Show the frame
        cv2.imshow('Snapchat Filter App', frame)
        
        # Take snapshot if requested
        if take_snapshot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved as {filename}")
            take_snapshot = False
        
        # Check for keypresses
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27 or key == ord('q'):  # ESC or Q key to quit
            break
        elif key == 32:  # SPACE key to take snapshot
            take_snapshot = True
        elif key == ord('f'):  # F key to change filter
            # Cycle through our custom filters: dog, demon, sunglasses, rabbit
            current_filter_index = (current_filter_index + 1) % len(custom_filters)
            current_filter = custom_filters[current_filter_index]
            
            # Update filter type based on the current filter
            if "glasses" in current_filter.lower() or "sunglasses" in current_filter.lower():
                current_filter_type = "glasses"
            else:
                current_filter_type = "full"
                
            print(f"Changed to filter: {current_filter}")
        elif key == ord('t'):  # T key to change filter type
            # This key is less relevant now with our specific filters, but keeping for compatibility
            filter_type_list = [t for t in filter_types if filter_types[t]]
            if filter_type_list:
                current_index = filter_type_list.index(current_filter_type) if current_filter_type in filter_type_list else -1
                current_index = (current_index + 1) % len(filter_type_list)
                current_filter_type = filter_type_list[current_index]
                
                # Find a filter of this type from our custom filters if possible
                matching_filters = [f for f in custom_filters if 
                                   (current_filter_type == "glasses" and ("glasses" in f.lower() or "sunglasses" in f.lower())) or
                                   (current_filter_type == "full" and not ("glasses" in f.lower() or "sunglasses" in f.lower()))]
                
                if matching_filters:
                    current_filter = matching_filters[0]
                    current_filter_index = custom_filters.index(current_filter)
        elif key == ord('p'):  # P key to pause/resume processing
            processing_active = not processing_active
        elif key == ord('h'):  # H key to toggle help
            show_help = not show_help
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()