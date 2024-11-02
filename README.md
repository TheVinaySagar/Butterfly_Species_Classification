# 🦋 Butterfly Species Classifier

An end-to-end machine learning project that predicts butterfly species from images using deep learning. The project includes a complete ML pipeline, Docker containerization, and deployment on Render.

## 🌟 Features

- Image-based butterfly species classification
- RESTful API for predictions
- Containerized application using Docker
- Complete ML pipeline from data preprocessing to model deployment
- Deployed and accessible on Render
- Available as a pre-built container on GitHub Container Registry

## 📊 Dataset

This project uses the [Butterfly Image Classification Dataset](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification) from Kaggle. The dataset contains:
- 75 different species of butterflies
- High-quality images with various backgrounds and angles
- Properly labeled and categorized data

To use the dataset:
1. Download from Kaggle: [Download Dataset](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)
2. Accept the dataset terms and conditions
3. Place the downloaded data in the `data/` directory of the project

## 🛠️ Tech Stack

- **Machine Learning**: TensorFlow
- **Web Framework**: FastAPI
- **Containerization**: Docker
- **Cloud Platform**: Render
- **Programming Language**: Python 3.8+

## 🚀 Quick Start

### Using Pre-built Container

Pull and run the image directly from GitHub Container Registry:

```bash
# Pull the image
docker pull ghcr.io/thevinaysagar/butterfly_species_classification/my-image-name:latest

# Run the container
docker run -p 8000:8000 ghcr.io/thevinaysagar/butterfly_species_classification/my-image-name:latest
```

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/TheVinaySagar/Butterfly_Species_Classification.git
cd Butterfly_Species_Classification
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

### 🐳 Building Docker Image Locally

1. Build the Docker image:
```bash
docker build -t butterfly-classifier .
```

2. Run the container:
```bash
docker run -p 8000:8000 butterfly-classifier
```

## 📚 API Documentation

Once the application is running, access the API documentation at:
- Swagger UI: `http://localhost:8000/scorefile/`
- Postman: `http://localhost:8000/scorefile/`
- ReDoc: `http://localhost:8000/scorefile/`

### Example API Request

```python
import requests

url = "https://butterfly-species-classification.onrender.com/"
files = {"file": open("butterfly_image.jpg", "rb")}
response = requests.post(url, files=files)
prediction = response.json()
```

## 🔄 ML Pipeline

The project implements a comprehensive machine learning pipeline:

1. Data Collection and Preprocessing
2. Model Training and Validation
3. Model Optimization
4. Inference Pipeline
5. API Integration
6. Deployment Pipeline

## 🌐 Deployment

The application is deployed on Render and can be accessed at:
[Live Link](https://butterfly-species-classification.onrender.com/)

### Deployment Steps

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure build and start commands
4. Set environment variables
5. Deploy!

## 📁 Project Structure

```
└── Butterfly_Classification/
    └── .gitattributes
    ├── Butterfly_Classification/
    │   └── Butterfly.ipynb
    │   └── __init__.py
    │   └── predictor.py
    │   └── tempCodeRunnerFile.py
    ├── Dockerfile
    ├── Model_pipeline/
    │   └── __init__.py
    │   ├── Model.py
    │   └── predictor.py
    │   └── preprocessing_utilities.py
    │   └── tempCodeRunnerFile.py
    │   └── training.py
    │   └── training_utilities.py
    └── README.md
    ├── app.yaml
    ├── app/
    │   └── Custom_CNN_Model.h5
    │   └── __init__.py
    │   └── config.yaml
    │   └── index.html
    │   └── main.py
    │   └── tempCodeRunnerFile.py
    ├── cnn.png
    ├── example/
    │   └── data_config.yaml
    │   └── predictor_config.yaml
    │   └── training_config.yaml
    └── requirements.txt
    └── setup.py
    └── tempCodeRunnerFile.py
```

## 📈 Model Performance

- Accuracy: XX%
- F1 Score: XX
- Supported Species: /app/config.yaml #Can see there

## 🐳 Container Registry

The project's Docker image is available on GitHub Container Registry. You can find versioned releases and the latest build:

```bash
# Pull specific version
docker pull ghcr.io/thevinaysagar/butterfly_species_classification/my-image-name:latest

# Pull latest
docker pull ghcr.io/thevinaysagar/butterfly_species_classification/my-image-name:latest
```

### Available Tags
- `latest`: Most recent build from the main branch
- `v1.0.0`: Stable release version
- `dev`: Development branch build

## 🤝 Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Vinay Sagar - [GitHub Profile](https://github.com/TheVinaySagar)

## 🙏 Acknowledgments

- [Butterfly Image Classification Dataset](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification) on Kaggle
- Any libraries or tools that were particularly helpful
- Inspiration for the project

## ✍️ Citation

If you use this project in your research, please cite:

```
@software{butterfly_classifier,
  author = {Your Name},
  title = {Butterfly Species Classifier},
  year = {2024},
  url = {https://github.com/TheVinaySagar/Butterfly_Species_Classification.git}
}
```