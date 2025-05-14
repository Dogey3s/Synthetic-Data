import os
import json
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from PIL import Image
import joblib
from tensorflow import keras
from fastapi.responses import FileResponse
import shutil



def generate_iris_data(samples=150, file_format="csv", output_dir="iris_output"):
    decoder = keras.models.load_model(r"P:\synthdata\iris\final_models\vae_decoder_model.keras")
    scaler = joblib.loadr(r"P:\synthdata\iris\scalers\iris_scaler.save")

    latent_dim = decoder.input_shape[1]
    random_latent_vectors = np.random.normal(size=(samples, latent_dim))
    synthetic_data = decoder.predict(random_latent_vectors)
    synthetic_data_rescaled = scaler.inverse_transform(synthetic_data)

    feature_names = ['sepal length (cm)', 'sepal width (cm)',
                     'petal length (cm)', 'petal width (cm)']
    df = pd.DataFrame(synthetic_data_rescaled, columns=feature_names)

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"iris_data.{file_format}")

    if file_format == "csv":
        df.to_csv(file_path, index=False)
        media_type = "text/csv"
    elif file_format == "xlsx":
        df.to_excel(file_path, index=False)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif file_format == "json":
        df.to_json(file_path, orient="records", indent=2)
        media_type = "application/json"
    else:
        return {"error": "Unsupported format"}

    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type=media_type
    )


def generate_adhd_data(samples=150, file_format="csv", output_dir="adhd_output"):
    decoder = keras.models.load_model(r"P:\synthdata\adhd\adhd_models\vae_decoder.keras")  
    scaler = joblib.load(r"P:\synthdata\adhd\scalers\adhd_scaler.save") 

    latent_dim = decoder.input_shape[1]
    random_latent_vectors = np.random.normal(size=(samples, latent_dim))
    synthetic_data = decoder.predict(random_latent_vectors)
    synthetic_data_rescaled = scaler.inverse_transform(synthetic_data)

    feature_names = ['Age', 'ADHD Index', 'Inattentive', 'Hyper/Impulsive', 'Full4 IQ']
    df = pd.DataFrame(synthetic_data_rescaled, columns=feature_names)

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"adhd_data.{file_format}")

    if file_format == "csv":
        df.to_csv(file_path, index=False)
        media_type = "text/csv"
    elif file_format == "xlsx":
        df.to_excel(file_path, index=False)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif file_format == "json":
        df.to_json(file_path, orient="records", indent=2)
        media_type = "application/json"
    else:
        return {"error": "Unsupported format"}

    return FileResponse(
        path=file_path,
        filename=os.path.basename(file_path),
        media_type=media_type
    )


def generate_image_data(model_name, samples=10, file_format="jpeg", output_dir="image_output"):
    if not (1 <= samples <= 15):
        return {"error": "Sample size must be between 1 and 15"}

    os.makedirs(output_dir, exist_ok=True)

    if model_name == "mnist_grayscale":
        model_path = r"P:\synthdata\mnsit_grayscale\mnist_final_models\generator_final.h5"
        image_shape = (28, 28)
        is_rgb = False
    elif model_name == "wgan_gp_oxford":
        model_path = r"P:\synthdata\wgan_gp_oxford\wgan_gp_model\generator_final.h5"
        image_shape = (64, 64, 3)
        is_rgb = True
    else:
        return {"error": "Unsupported image model"}

    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    latent_dim = model.input_shape[1]
    latent_vectors = np.random.normal(0, 1, (samples, latent_dim))
    generated_images = model.predict(latent_vectors)

    # Create subfolder
    image_dir = os.path.join(output_dir, f"{model_name}_{samples}_{file_format}")
    os.makedirs(image_dir, exist_ok=True)

    for i in range(samples):
        img_array = generated_images[i]
        img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        
        if not is_rgb:
            img_array = img_array.reshape(image_shape)
            img = Image.fromarray(img_array, mode="L")
        else:
            img_array = img_array.reshape(image_shape)
            img = Image.fromarray(img_array, mode="RGB")

        img.save(os.path.join(image_dir, f"{model_name}_{i + 1}.{file_format}"), format=file_format.upper())

    zip_path = f"{image_dir}.zip"
    shutil.make_archive(image_dir, 'zip', image_dir)
    return zip_path
