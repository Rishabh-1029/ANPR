from ultralytics import YOLO
import torch
import easyocr


# Vehicle Detection
def vehicle_detection():
    vehicle_model = YOLO(r"C:\Users\Rishabh Surana\Desktop\Projects\Number Plate OCR\app\Models\yolov8n.pt")
    vehicle_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return vehicle_model


# Plate detection
def plate_detection():
    plate_model = YOLO(r"C:\Users\Rishabh Surana\Desktop\Projects\Number Plate OCR\app\Models\best_number_plate_model.pt")
    plate_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return plate_model


# OCR
def ocr_reader():
    plate_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    return plate_reader