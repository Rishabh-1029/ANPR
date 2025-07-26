# Libraries

import cv2
import numpy as np

# Result function
def view_result(frame, vehicle_crop, plate_crop, corrected_text, is_valid):
    
    # Resize frame function
    def resize_frame(picture, w, h):
        return cv2.resize(picture, (w, h))
    
    # Border and Padding function
    def frame_border_padding(picture, top, bottom, left, right, color):
        padding = cv2.copyMakeBorder(picture, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        border = cv2.copyMakeBorder(padding, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return border
        
    # Original Frame
    display_frame = resize_frame(frame, 450, 300)
    display_frame = frame_border_padding(display_frame, 0,0,0,0,(255,255,255))
    
    # Detected Vehicle
    display_vehicle = resize_frame(vehicle_crop, 300, 200)
    display_vehicle = frame_border_padding(display_vehicle, 50, 50, 0, 0, (255, 255, 255))
    
    
    # Combining original frame + Detected vehicle (Horizontally Side by side)
    combined1 = np.hstack((display_frame, display_vehicle))
    
    
    # Detected Number Plate
    display_plate = cv2.cvtColor(plate_crop, cv2.COLOR_GRAY2BGR)
    display_plate = resize_frame(display_plate, 260, 90)
    display_plate = frame_border_padding(display_plate, 90, 90, 0, 0, (255, 255, 255))
    
    # Detected Text (Number plate)
    text_canvas = np.ones((190, 480, 3), dtype=np.uint8) * 255
    cv2.putText(text_canvas, f"Number plate : {corrected_text}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    color = (0, 255, 0) if is_valid else (0, 0, 255)
    cv2.putText(text_canvas, f"Status: {'VALID' if is_valid else 'INVALID'}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    text_canvas = frame_border_padding(text_canvas, 40, 40, 5, 5, (255,255,255))
    
    # Combining Number Plate + Detected Text (Horizontally Side by side)
    combined2 = np.hstack((display_plate, text_canvas))
    
    
    # Combining all detected things in a single frame
    combined = np.vstack((combined1, combined2))
    
    
    # Final result
    cv2.imshow("ANPR", combined)
    cv2.waitKey(0)
