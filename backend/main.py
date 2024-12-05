from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
from dotenv import load_dotenv
from pitch_evaluator import PitchEvaluator
import asyncio
import json
from mangum import Mangum

from fastapi.middleware.cors import CORSMiddleware


# Load environment variables
load_dotenv()

# Create FastAPI app instance
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your CloudFront domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the evaluator with API keys
evaluator = PitchEvaluator(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    assemblyai_api_key=os.getenv("ASSEMBLYAI_API_KEY")
)

# Store analysis results
analysis_results = {}

# Use environment variables
PORT = int(os.getenv("PORT", 8000))

@app.get("/")
async def root():
    return {"message": "Pitch Analyzer API is running"}

@app.post("/analyze-pitch/")
async def analyze_pitch(file: UploadFile):
    try:
        print(f"Received file: {file.filename}")  # Debug log
        
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save uploaded file
        file_path = f"uploads/{file.filename}"
        try:
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            print(f"File saved successfully to {file_path}")
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
        
        try:
            # Run evaluation
            print("Starting evaluation...")
            result = evaluator.evaluate_pitch(file_path)
            print("Evaluation completed successfully")
            
            # Clean up
            os.remove(file_path)
            return result
            
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")
            
    except Exception as e:
        print(f"General error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"General error: {str(e)}")

@app.get("/results/{filename}")
async def get_results(filename: str):
    if filename not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis_results[filename]

@app.get("/all-results")
async def get_all_results():
    return analysis_results

# Add the Lambda handler
handler = Mangum(app)