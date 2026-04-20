#!/usr/bin/env python3

#from typing import Optional
#from pydantic import BaseModel

import mysql.connector
from mysql.connector import Error
import boto3
import json
import os
from fastapi import FastAPI


DB = "epj7rf"

DBUSER = os.environ['DBUSER']
DBHOST = os.environ['DBHOST']
DBPASS = os.environ['DBPASS']
db = mysql.connector.connect(user=DBUSER, host=DBHOST, password=DBPASS, database=DB)
cur=db.cursor()


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/genres')
def get_songs():
    db = mysql.connector.connect(user=DBUSER, host=DBHOST, password=DBPASS, database=DB)
    cur=db.cursor()
    query = "SELECT * FROM genres ORDER BY genreid;"
    try:    
        cur.execute(query)
        headers=[x[0] for x in cur.description]
        results = cur.fetchall()
        json_data=[]
        for result in results:
            json_data.append(dict(zip(headers,result)))
        cur.close()
        db.close()
        return(json_data)
    except Error as e:
        cur.close()
        db.close()
        return {"Error": "MySQL Error: " + str(e)}

@app.get('/songs')
def get_songs():
    db = mysql.connector.connect(user=DBUSER, host=DBHOST, password=DBPASS, database=DB)
    cur=db.cursor()
    query = "SELECT * FROM songs ORDER BY title;"
    try:    
        cur.execute(query)
        headers=[x[0] for x in cur.description]
        results = cur.fetchall()
        json_data=[]
        for result in results:
            json_data.append(dict(zip(headers,result)))
        cur.close()
        db.close()
        return(json_data)
    except Error as e:
        cur.close()
        db.close()
        return {"Error": "MySQL Error: " + str(e)}







@app.get("/")  # zone ap
def zone_apex():
    return {"What's up": "Duuuude Vegeta!"}

# @app.get("/sum/{a}/{b}")
# def add(a: int, b: int):
#     return {"sum": a + b}


@app.get("/multiply/{c}/{d}")
def multiply(c: int, d: int):
    return {"product": c * d}

@app.get("/square/{a}")
def square(a: int):
    return {"square": a * a}

# @app.get("/multiply/{c}/{d}")
# def multiply(c: int, d: int):
#     return {"product": c * d}

#@app.get("/multiply/{c}/{d}")
# def multiply(c: int, d: int):
#     return {"product": c * d}

# @app.get("/customer/{idx}")
# def customer(idx: int):
#         df = pd.read_csv("../customers.csv")
#         # filter dhe data based on the index
#         customer = df.iloc[idx]
#         return customer.to_dict()


# @app.post("/get_body")
# async def get_body(request: Request):
#      response = await request.json()
#      first_name = response["fname"]
#      last_name = response["lname"]
#      favorite_number = response["favnum"]
#      return {"first_name": first_name, "last_name": last_name, "favorite_number": "favorite_number"}
#     # return await request.json()
>>>>>>> 9b6cf9f (spotify project steps 4/8)
