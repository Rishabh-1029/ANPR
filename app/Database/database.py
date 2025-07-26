from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./anpr.db"

engine = create_engine(DATABASE_URL, connect_args ={"check_same_thread":False})
SessionLocal = sessionmaker(bind = engine, autoflush = False, autocommit = False)
Base = declarative_base()

class PlateEntry(Base):
    __tablename__ = "plates"
    id = Column(Integer, primary_key = True, index = True)
    plate_number = Column(String, index = True)
    status = Column(String)
    confidence =  Column(String)
    timestamp = Column(DateTime, default = datetime.utcnow)
    image_name = Column(String)
    
Base.metadata.create_all(bind=engine)