# Automatic Number Plate Recognition (ANPR)

A real-time ANPR system using YOLO for vehicle and plate detection, enhanced by OCR with custom preprocessing and postprocessing. This project is deployed as a FastAPI web service on [https://anpr-4u9c.onrender.com](https://anpr-4u9c.onrender.com).

## Overview

This project detects vehicles and their number plates from image input using a custom-trained YOLO model. After accurate license plate localization, OCR is performed using EasyOCR with tailored preprocessing and postprocessing for optimal character recognition.

---

## Features

-  Vehicle detection using YOLOv8
-  License plate detection with **custom-trained YOLO** model  
  - **Precision:** 1.000  
  - **Recall:** 0.999  
  - **mAP@0.5:** 0.995  
  - **mAP@0.5:0.95:** 0.867  
-  OCR using **EasyOCR** with preprocessing and postprocessing pipelines
-  Fast, modular, and scalable pipeline
-  RESTful API powered by **FastAPI**
-  Deployed on **Render**

---

## Tech Stack

- **YOLOv8** for detection (vehicles + plates)
- **EasyOCR** for OCR
- **OpenCV**, **NumPy** for image handling
- **FastAPI** + **Uvicorn** for API deployment
- **PyTorch** for deep learning integration
