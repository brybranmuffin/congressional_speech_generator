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


@app.post("/calculate_emi")
async def calculate_emi(
    w2v: bool = Query(...),
    bert: bool = Query(...),
    gpt2: bool = Query(...),
    congr_speech_id: str = Query(...),
    text: str = Query(...),
    x_api_key: str = Header(...)
):
    return "Not yet implemented"
    

