from ultralytics import YOLO
import torch
import cv2
import easyocr
import pytesseract
from PIL import Image


# Vehicle Detection
def vehicle_detection():
    vehicle_model_path = app/Models/yolov8n.pt
    vehicle_model = YOLO(str(vehicle_model_path))
    vehicle_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return vehicle_model


# Plate detection
def plate_detection():
    plate_model_path = app/Models/best_number_plate_model.pt
    plate_model = YOLO(str(plate_model_path))
    plate_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return plate_model


# OCR
def easy_ocr_reader():
    plate_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    return plate_reader

def tess_ocr_reader():
    
    class TesseractReader:
        
        def __init__(self):
            self.pytesseract = pytesseract
            self.cv2 = cv2
            self.Image = Image
            # self.pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract_OCR\tesseract.exe"
        
        def readtext(self, image):
           
           try:
               h,w = image.shape[:2]
               aspect_ratio = w / h
               
               config = "--oem 1 --psm 7"
               result = []
               
               
               rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
               
               if aspect_ratio < 2.0:
                   
                   top_half = rgb[0:h//2, :]
                   
                   bottom_half = rgb[h//2:, :]
                   
                   top_text = pytesseract.image_to_string(Image.fromarray(top_half), config = config).strip()  
                   bottom_text = pytesseract.image_to_string(Image.fromarray(bottom_half), config = config).strip()
                   
                   print(f"Top text : {top_text}")
                   print(f"Bottom text : {bottom_text}")
                   
                   full_text = top_text + bottom_text
                   print(f"Full text : {full_text}")
                   
               else:
                   
                   full_text = pytesseract.image_to_string(Image.fromarray(rgb),config=config).strip()
                
               if full_text:
                   
                   result.append((None, full_text, 0.0))
                   
               return result
           
           except Exception as e:
               print(f"Tesseract OCR failed : {e}")
               return []
           
    return TesseractReader()
            
