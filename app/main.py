# Import necessary libraries from FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import additional libraries for file handling
import sys
import os.path

# Add the parent directory to the system path for module imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

# Import the ImagePredictor class from the Butterfly_Classification module
from Butterfly_Classification.predictor import ImagePredictor

# Initialize the FastAPI app
app = FastAPI()

# Mount static files directory to serve static assets
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the root endpoint to serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serve the index.html file as the main page of the application.
    """
    with open("index.html", "r") as f:
        return f.read()

# Load the predictor configuration from the specified YAML file
predictor_config_path = "config.yaml"
predictor = ImagePredictor.init_from_config_path(predictor_config_path)

# Define the endpoint to upload files for scoring
@app.post("/scorefile/")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Handle file uploads and perform predictions using the ImagePredictor.
    
    Args:
        file (UploadFile): The uploaded file to be processed.

    Returns:
        The prediction result from the ImagePredictor.
    
    Raises:
        HTTPException: If there's an error during file processing.
    """
    print("dd0")
    try:
        # Perform prediction using the uploaded file
        print("dd1")
        result = predictor.predict_from_file(file.file)
        return result
    except Exception as e:
        # Raise an HTTP exception if an error occurs
        print("dd2")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
