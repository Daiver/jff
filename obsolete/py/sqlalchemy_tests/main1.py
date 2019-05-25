from sqlalchemy import *
from sqlalchemy.orm import *
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TrackingTable(Base):

    __tablename__ = 'tracking_table'

    trackid = Column(BigInteger, primary_key=True, autoincrement=True)
    formdata = Column(String(100), nullable=False)

e = create_engine("sqlite:///test", echo=True)
Base.metadata.create_all(e)

s = Session(e)

updateRecord = TrackingTable(formdata='logged in')

s.add(updateRecord)
s.commit()

print updateRecord.trackid
