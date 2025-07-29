# Libraries
import cv2
from app.Database.database import SessionLocal, PlateEntry
from app.Models.models import vehicle_detection, plate_detection, easy_ocr_reader, tess_ocr_reader
from app.Services.validation import valid_indian_numb_plate_fn, correct_plate_text
from app.Services.preprocessing import preprocess_plate
from app.Services.reports import view_result

# Modles
vehicle_model = vehicle_detection()
plate_model = plate_detection()
easy_plate_reader = easy_ocr_reader()
tess_plate_reader = tess_ocr_reader()


plate_seen = set()
results = []
        
def run_anpr(img_path: str):
    frame = cv2.imread(img_path)
    db = SessionLocal()
    vehicle_results = vehicle_model(frame, verbose=False)
    
    for result in vehicle_results:
        
        for box in result.boxes:
            
            allowed_vehicles = [2,3,5,7]
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if (confidence < 0.3) or (class_id not in allowed_vehicles):
                continue
            
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            padding = 50
            
            height, width, _ = frame.shape
            
            x1_p = max(x1 - padding, 0)
            y1_p = max(y1 - padding, 0)
            x2_p = min(x2 + padding, width)
            y2_p = min(y2 + padding, height)
            
            vehicle_crop = frame[y1_p:y2_p, x1_p:x2_p]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID: {class_id}, Conf: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Detected Vehicles", frame)
            cv2.waitKey(0)

            plate_result = plate_model(vehicle_crop, verbose=False) 
            
            for plate in plate_result:
     
                for pbox in plate.boxes:
                    
                    p_confidence = float(box.conf[0])
                    if (p_confidence < 0.8):
                        continue
                    
                    px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                    
                    plate_crop = vehicle_crop[py1:py2, px1:px2]
                    plate_crop = preprocess_plate(plate_crop)
                    
                    # ocr_result = easy_plate_reader.readtext(plate_crop)
                    ocr_result = tess_plate_reader.readtext(plate_crop)
                    
                    for (_, text, conf) in ocr_result:
                        
                        print(f"\nText : {text}")
                        
                        cleaned_text = text.upper().replace(" ", "").replace("-", "").replace("@", "").replace("?","3").replace(".","").replace(",","")
                        print(f"cleaned_text : {cleaned_text}")
                        
                        if not(7 <= len(cleaned_text) <= 12):
                            continue
                        
                        corrected_text = correct_plate_text(cleaned_text)
                        print(f"corrected_text : {corrected_text}\n")
                        
                        is_valid = valid_indian_numb_plate_fn(corrected_text)
                        
                        entry = PlateEntry(
                            plate_number=corrected_text,
                            status="VALID" if is_valid else "INVALID",
                            confidence=str(round(conf, 2)),
                            image_name=img_path.split("/")[-1]
                        )
                        
                        if corrected_text not in plate_seen:
                            
                            db.add(entry)
                            db.commit()
                            
                            plate_seen.add(corrected_text)
                            
                            view_result(frame, vehicle_crop, plate_crop, corrected_text, is_valid)
                            
                            results.append({'plate':corrected_text,'status':"VALID" if is_valid else "INVALID",})
                            
    db.close()
    return results

cv2.destroyAllWindows()

# img_path = r"C:\Users\Rishabh Surana\Desktop\Projects\Number Plate OCR\app\Models\Dataset\Number plate OCR dataset\State-wise_OLX\RJ\RJ5.jpg"

# img_path = r"C:\Users\Rishabh Surana\Desktop\Projects\ATMS project\test_data\test_image\anpr_test.png"

# # img_path = r"C:\Users\Rishabh Surana\Desktop\Projects\ATMS project\test_data\test_image\Screenshot 2025-07-28 143124.png"

# img_path = r"C:\Users\Rishabh Surana\Desktop\Projects\ATMS project\test_data\test_image\Screenshot 2025-07-29 113325.png"

# run_anpr(img_path)