# server.py
import os
import asyncio
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

# we will import and call the main() from your existing app.py
import app as importer

RUN_TOKEN = os.getenv("RUN_TOKEN", "")

app = FastAPI(title="Eldorado Import Trigger")

class RunResponse(BaseModel):
    status: str
    message: Optional[str] = None
    appended: Optional[int] = None
    parsed: Optional[int] = None

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/run", response_model=RunResponse)
async def run(request: Request):
    # Simple auth (token in header)
    token = request.headers.get("X-Run-Token", "")
    if not RUN_TOKEN or token != RUN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Call the importer. It logs internally; we’ll grab key stats if exposed.
    # We’ll wrap main() and infer counts from logs if needed, but simplest is: run and respond OK.
    try:
        # Run importer.main() synchronously
        importer.main()
        return RunResponse(status="ok", message="Import executed.")
    except Exception as e:
        # You’ll also see full details in Render logs
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
