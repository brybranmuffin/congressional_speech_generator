import io
import json
import math

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/generate_speech")
async def generate_speech(
party: str = Query(...),
EMI_index: float = Query(...),
topics: str = Query(...),
time_period: str = Query(...),
x_api_key: str = Header(...)
):
    return "Not yet implemented"
    

