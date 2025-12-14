import os
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.database import Base, engine, SessionLocal
from backend.errors import validation_exception_handler, general_exception_handler
from backend.ml.detector import count_people
from backend.models import Detection, Room

app = FastAPI()
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# CORS для работы фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажи конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "panel.html"))


@app.post("/analyze")
def analyze_video(room: str = Form(...), file: UploadFile = File(...)):
    room = room.strip()
    if not room:
        raise HTTPException(status_code=400, detail="Room is required")
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    filename = f"{datetime.utcnow().timestamp()}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(filepath, "wb") as f:
            f.write(file.file.read())
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save file")

    db = SessionLocal()
    try:
        room_obj = db.query(Room).filter(Room.name == room).first()
        if not room_obj:
            room_obj = Room(name=room)
            db.add(room_obj)
            db.commit()
            db.refresh(room_obj)

        try:
            people = count_people(filepath)
        except Exception:
            raise HTTPException(status_code=500, detail="YOLO processing error")

        rec = Detection(filename=filename, people_count=people, room_id=room_obj.id)
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return {"filename": filename, "people_count": people, "room": room_obj.name}
    finally:
        db.close()


@app.get("/history")
def get_history():
    db = SessionLocal()
    items = (
        db.query(Detection, Room)
        .join(Room, Detection.room_id == Room.id)
        .order_by(Detection.timestamp.desc())
        .all()
    )
    result = [
        {
            "id": det.id,
            "filename": det.filename,
            "people_count": det.people_count,
            "timestamp": det.timestamp,
            "room": room.name,
        }
        for det, room in items
    ]
    db.close()
    return result


@app.get("/rooms")
def list_rooms():
    db = SessionLocal()
    rooms = db.query(Room).order_by(Room.name).all()
    db.close()
    return [{"id": r.id, "name": r.name} for r in rooms]