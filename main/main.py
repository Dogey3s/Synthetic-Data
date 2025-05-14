from fastapi import FastAPI
from fastapi.responses import FileResponse
from enum import Enum
from model_util import generate_iris_data, generate_adhd_data, generate_image_data
import os

app = FastAPI(title=r"main/main")

class TabularModel(str, Enum):
    iris = "iris"
    adhd = "adhd"

class ImageModel(str, Enum):
    mnist = "mnist_grayscale"
    wgan = "wgan_gp_oxford"

class TabularFormat(str, Enum):
    csv = "csv"
    xlsx = "xlsx"
    json = "json"

class ImageFormat(str, Enum):
    jpeg = "jpeg"
    png = "png"
    gif = "gif"

@app.get("/generate/tabular")
def generate_tabular(model: TabularModel, samples: int = 150, file_format: TabularFormat = "csv"):
    if model == TabularModel.iris:
        return generate_iris_data(samples=samples, file_format=file_format)
    elif model == TabularModel.adhd:
        return generate_adhd_data(samples=samples, file_format=file_format)
    
    return {"message": "Inappropriate request."}


@app.get("/generate/image")
def generate_image(model: ImageModel, samples: int = 10, file_format: ImageFormat = "jpeg"):
    zip_path = generate_image_data(model_name=model, samples=samples, file_format=file_format)
    if isinstance(zip_path, dict) and "error" in zip_path:
        return zip_path

    return FileResponse(
        path=zip_path,
        filename=os.path.basename(zip_path),
        media_type="application/zip"
    )