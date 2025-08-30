#!/usr/bin/env python3
import os
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn

# import your importer main()
import app as importer

# Secret token for authorization
RUN_TOKEN = os.getenv("RUN_TOKEN", "")

app = FastAPI(title="Eldorado Import Trigger")

class RunResponse(BaseModel):
    status: str
    message: str | None = None

@app.get("/")
def home():
    return {
        "service": "Eldorado Import Trigger",
        "endpoints": [
            "/health",
            "POST /run (header X-Run-Token: ...)",
            "GET /run?token=..."
        ],
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/run", response_model=RunResponse)
@app.get("/run", response_model=RunResponse)   # allow GET with ?token=
def run(request: Request):
    token = request.headers.get("X-Run-Token") or request.query_params.get("token")
    if not RUN_TOKEN or token != RUN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        importer.main()
        return RunResponse(status="ok", message="Import executed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
