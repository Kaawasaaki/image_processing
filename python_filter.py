import cv2
import numpy as np
import os
from PIL import Image

def create_simple_filter(name, filter_type, color=(255, 0, 0, 180)):
    """
    Create a simple filter with transparent background
    
    Parameters:
    name (str): Name of the filter file
    filter_type (str): Type of filter ('glasses', 'hat', 'full', etc.)
    color (tuple): Color in RGBA format (default: semi-transparent red)
    """
    if not os.path.exists('filters'):
        os.makedirs('filters')
    
    if filter_type == 'glasses':
        # Create sunglasses-like filter
        img = np.zeros((150, 400, 4), dtype=np.uint8)
        
        # Left lens
        cv2.circle(img, (100, 75), 60, color, -1)
        
        # Right lens
        cv2.circle(img, (300, 75), 60, color, -1)
        
        # Bridge
        cv2.rectangle(img, (160, 65), (240, 85), color, -1)
        
        # Temples (arms)
        cv2.rectangle(img, (40, 65), (10, 75), color, -1)
        cv2.rectangle(img, (360, 65), (390, 75), color, -1)
        
    elif filter_type == 'hat':
        # Create a simple hat filter
        img = np.zeros((250, 500, 4), dtype=np.uint8)
        
        # Hat body
        cv2.rectangle(img, (100, 50), (400, 150), color, -1)
        
        # Hat top
        pts = np.array([[250, 10], [100, 100], [400, 100]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color)
        
    elif filter_type == 'crown':
        # Create a crown filter
        img = np.zeros((200, 400, 4), dtype=np.uint8)
        
        # Base
        cv2.rectangle(img, (50, 100), (350, 150), color, -1)
        
        # Points
        points = [(50, 100), (100, 50), (150, 100), (200, 20), 
                  (250, 100), (300, 50), (350, 100)]
        
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], color, 15)
            
        # Fill the crown
        cv2.fillPoly(img, [np.array(points, np.int32)], color)
        
    elif filter_type == 'mustache':
        # Create a mustache filter
        img = np.zeros((150, 400, 4), dtype=np.uint8)
        
        # Draw mustache shape
        center_x, center_y = 200, 75
        for i in range(0, 200, 2):
            y_offset = 25 * np.sin(i/30)
            thickness = 20 - abs(i - 100) // 10
            cv2.line(img, 
                     (center_x - 100 + i, center_y + int(y_offset)), 
                     (center_x - 100 + i, center_y + int(y_offset) + thickness), 
                     color, 1)
    
    else:  # 'full' - a simple mask for the whole face
        img = np.zeros((500, 400, 4), dtype=np.uint8)
        
        # Face outline
        cv2.ellipse(img, (200, 250), (180, 250), 0, 0, 360, color, -1)
        
        # Make the center more transparent
        inner = np.zeros_like(img)
        cv2.ellipse(inner, (200, 250), (150, 220), 0, 0, 360, (0, 0, 0, 100), -1)
        
        # Blend
        alpha = inner[:,:,3] / 255.0
        alpha = np.stack([alpha, alpha, alpha, alpha], axis=2)
        img = (1 - alpha) * img + alpha * inner
    
    # Save the filter
    filename = f"{name}_{filter_type}.png"
    filepath = os.path.join('filters', filename)
    
    # Convert to PIL Image and save with transparency
    pil_img = Image.fromarray(img.astype('uint8'))
    pil_img.save(filepath, format='PNG')
    
    print(f"Filter saved as {filepath}")
    return filepath

def create_color_filter(name, filter_type, color_effect):
    """
    Create a color filter effect (this doesn't create a PNG but demonstrates code for a color filter)
    
    Parameters:
    name (str): Name of the filter
    filter_type (str): Type of filter
    color_effect (str): Effect type ('warm', 'cool', 'grayscale', 'sepia', etc.)
    """
    print(f"\nCOLOR FILTER CODE EXAMPLE: {name}_{filter_type}_{color_effect}")
    print("Add this to your main program to apply color filters:")
    
    if color_effect == 'warm':
        print("""
        # Warm filter - increase red channel, decrease blue
        def apply_warm_filter(frame):
            frame_copy = frame.copy()
            frame_copy[:,:,2] = np.clip(frame_copy[:,:,2] * 1.2, 0, 255)  # Increase red
            frame_copy[:,:,0] = np.clip(frame_copy[:,:,0] * 0.8, 0, 255)  # Decrease blue
            return frame_copy
        """)
    
    elif color_effect == 'cool':
        print("""
        # Cool filter - increase blue channel, decrease red
        def apply_cool_filter(frame):
            frame_copy = frame.copy()
            frame_copy[:,:,0] = np.clip(frame_copy[:,:,0] * 1.2, 0, 255)  # Increase blue
            frame_copy[:,:,2] = np.clip(frame_copy[:,:,2] * 0.8, 0, 255)  # Decrease red
            return frame_copy
        """)
    
    elif color_effect == 'grayscale':
        print("""
        # Grayscale filter
        def apply_grayscale_filter(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        """)
    
    elif color_effect == 'sepia':
        print("""
        # Sepia filter
        def apply_sepia_filter(frame):
            frame_copy = frame.copy()
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)
            
            # Apply sepia color transformation
            frame_copy[:,:,0] = np.clip(frame_copy[:,:,0] * 0.272, 0, 255)  # Blue
            frame_copy[:,:,1] = np.clip(frame_copy[:,:,1] * 0.534, 0, 255)  # Green
            frame_copy[:,:,2] = np.clip(frame_copy[:,:,2] * 0.131, 0, 255)  # Red
            
            return frame_copy
        """)
    
    elif color_effect == 'vintage':
        print("""
        # Vintage filter
        def apply_vintage_filter(frame):
            # Apply slight sepia tone
            frame_sepia = frame.copy()
            frame_sepia[:,:,0] = np.clip(frame[:,:,0] * 0.8, 0, 255)  # Blue
            frame_sepia[:,:,1] = np.clip(frame[:,:,1] * 0.9, 0, 255)  # Green
            frame_sepia[:,:,2] = np.clip(frame[:,:,2] * 1.1, 0, 255)  # Red
            
            # Reduce contrast
            frame_vintage = cv2.addWeighted(frame_sepia, 0.7, np.zeros_like(frame_sepia), 0, 60)
            
            # Add vignette effect
            rows, cols = frame_vintage.shape[:2]
            # Generate vignette mask
            kernel_x = cv2.getGaussianKernel(cols, cols/4)
            kernel_y = cv2.getGaussianKernel(rows, rows/4)
            kernel = kernel_y * kernel_x.T
            mask = 255 * kernel / np.linalg.norm(kernel)
            mask = mask.astype(np.uint8)
            
            # Apply the mask
            for i in range(3):
                frame_vintage[:,:,i] = frame_vintage[:,:,i] * mask / 255
                
            return frame_vintage
        """)
    
    print("\nTo use this filter, add the above function to your code and call it on each frame.")
    return

def create_dog_filter():
    """Create a dog filter with ears and nose"""
    if not os.path.exists('filters'):
        os.makedirs('filters')
        
    # Create a transparent canvas
    img = np.zeros((500, 400, 4), dtype=np.uint8)
    
    # Dog ears (triangular shape)
    # Left ear
    pts_left = np.array([[100, 100], [50, 20], [150, 50]], np.int32)
    pts_left = pts_left.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts_left], (139, 69, 19, 220))  # Brown color
    
    # Right ear
    pts_right = np.array([[300, 100], [250, 50], [350, 20]], np.int32)
    pts_right = pts_right.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts_right], (139, 69, 19, 220))  # Brown color
    
    # Dog nose (ellipse)
    cv2.ellipse(img, (200, 250), (40, 30), 0, 0, 360, (0, 0, 0, 230), -1)
    
    # Dog mouth
    cv2.line(img, (200, 280), (200, 320), (0, 0, 0, 230), 3)
    cv2.line(img, (170, 320), (230, 320), (0, 0, 0, 230), 3)
    
    # Save the filter
    filepath = os.path.join('filters', "dog_full.png")
    pil_img = Image.fromarray(img.astype('uint8'))
    pil_img.save(filepath, format='PNG')
    print(f"Dog filter saved as {filepath}")

def create_rabbit_filter():
    """Create a rabbit filter with ears and nose"""
    if not os.path.exists('filters'):
        os.makedirs('filters')
        
    # Create a transparent canvas
    img = np.zeros((600, 400, 4), dtype=np.uint8)
    
    # Rabbit ears (long and pointy)
    # Left ear
    pts_left = np.array([[150, 150], [100, 10], [170, 30], [180, 150]], np.int32)
    pts_left = pts_left.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts_left], (240, 240, 240, 220))  # Light gray
    
    # Inner left ear
    pts_left_inner = np.array([[150, 140], [110, 30], [165, 40], [170, 140]], np.int32)
    pts_left_inner = pts_left_inner.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts_left_inner], (255, 192, 203, 220))  # Pink
    
    # Right ear
    pts_right = np.array([[220, 150], [230, 30], [300, 10], [250, 150]], np.int32)
    pts_right = pts_right.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts_right], (240, 240, 240, 220))  # Light gray
    
    # Inner right ear
    pts_right_inner = np.array([[230, 140], [235, 40], [290, 30], [240, 140]], np.int32)
    pts_right_inner = pts_right_inner.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts_right_inner], (255, 192, 203, 220))  # Pink
    
    # Rabbit nose (small ellipse)
    cv2.ellipse(img, (200, 250), (15, 10), 0, 0, 360, (255, 0, 255, 230), -1)
    
    # Whiskers
    cv2.line(img, (200, 260), (150, 240), (0, 0, 0, 180), 2)
    cv2.line(img, (200, 260), (150, 260), (0, 0, 0, 180), 2)
    cv2.line(img, (200, 260), (150, 280), (0, 0, 0, 180), 2)
    cv2.line(img, (200, 260), (250, 240), (0, 0, 0, 180), 2)
    cv2.line(img, (200, 260), (250, 260), (0, 0, 0, 180), 2)
    cv2.line(img, (200, 260), (250, 280), (0, 0, 0, 180), 2)
    
    # Save the filter
    filepath = os.path.join('filters', "rabbit_full.png")
    pil_img = Image.fromarray(img.astype('uint8'))
    pil_img.save(filepath, format='PNG')
    print(f"Rabbit filter saved as {filepath}")

def create_demon_filter():
    """Create a demon filter with horns and details"""
    if not os.path.exists('filters'):
        os.makedirs('filters')
        
    # Create a transparent canvas
    img = np.zeros((500, 400, 4), dtype=np.uint8)
    
    # Demon horns
    # Left horn
    pts_left = np.array([[140, 150], [80, 20], [150, 50], [160, 150]], np.int32)
    pts_left = pts_left.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts_left], (180, 0, 0, 220))  # Dark red
    
    # Right horn
    pts_right = np.array([[240, 150], [250, 50], [320, 20], [260, 150]], np.int32)
    pts_right = pts_right.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts_right], (180, 0, 0, 220))  # Dark red
    
    # Evil eyebrows
    cv2.line(img, (120, 180), (180, 200), (0, 0, 0, 230), 5)
    cv2.line(img, (280, 180), (220, 200), (0, 0, 0, 230), 5)
    
    # Red eyes effect (circles)
    cv2.circle(img, (150, 220), 20, (255, 0, 0, 180), -1)
    cv2.circle(img, (250, 220), 20, (255, 0, 0, 180), -1)
    
    # Evil smile
    pts_smile = np.array([[150, 300], [200, 330], [250, 300], [220, 310], [200, 315], [180, 310]], np.int32)
    pts_smile = pts_smile.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts_smile], (0, 0, 0, 230))
    
    # Save the filter
    filepath = os.path.join('filters', "demon_full.png")
    pil_img = Image.fromarray(img.astype('uint8'))
    pil_img.save(filepath, format='PNG')
    print(f"Demon filter saved as {filepath}")

def create_sunglasses():
    """Create cool sunglasses filter"""
    if not os.path.exists('filters'):
        os.makedirs('filters')
        
    # Create a transparent canvas for sunglasses
    img = np.zeros((200, 400, 4), dtype=np.uint8)
    
    # Left lens
    cv2.rectangle(img, (50, 50), (150, 100), (0, 0, 0, 250), -1)
    
    # Right lens
    cv2.rectangle(img, (250, 50), (350, 100), (0, 0, 0, 250), -1)
    
    # Bridge
    cv2.rectangle(img, (150, 70), (250, 80), (0, 0, 0, 250), -1)
    
    # Temples (arms)
    cv2.rectangle(img, (50, 75), (30, 65), (0, 0, 0, 250), -1)
    cv2.rectangle(img, (350, 75), (370, 65), (0, 0, 0, 250), -1)
    
    # Add cool reflection effect
    cv2.line(img, (60, 60), (90, 60), (255, 255, 255, 200), 2)
    cv2.line(img, (260, 60), (290, 60), (255, 255, 255, 200), 2)
    
    # Save the filter
    filepath = os.path.join('filters', "sunglasses_glasses.png")
    pil_img = Image.fromarray(img.astype('uint8'))
    pil_img.save(filepath, format='PNG')
    print(f"Sunglasses filter saved as {filepath}")

def main():
    """Create the specified filters: dog, demon, sunglasses, and rabbit"""
    print("Creating the requested filters for the Snapchat-style Filter App")
    
    # Create requested filters
    create_dog_filter()
    create_demon_filter()
    create_sunglasses()
    create_rabbit_filter()
    
    # Create code examples for color filters (keeping these as educational)
    create_color_filter("warm", "color", "warm")
    create_color_filter("cool", "color", "cool")
    
    print("\nAll sample filters created successfully!")
    print("The filters are saved in the 'filters' directory.")
    print("You can now run the main Snapchat Filter App.")

if __name__ == "__main__":
    main()