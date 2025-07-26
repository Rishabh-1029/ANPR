from fastapi import FastAPI, UploadFile, File
from app.anpr_core import run_anpr
from app.Database.database import SessionLocal, PlateEntry
from fastapi.responses import JSONResponse
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "test_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    plates = run_anpr(file_path)

    return {
        "status": plates['status'],
        "file": file.filename,
        "detected_plates": plates['plate']
    }

@app.get("/plates/")
def get_plates():
    db = SessionLocal()
    records = db.query(PlateEntry).all()
    db.close()
    return records