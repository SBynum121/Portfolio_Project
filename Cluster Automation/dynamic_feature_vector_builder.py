# Standard library
import inspect
import os
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any


# Numerical and data processing
import numpy as np
import pandas as pd
from numpy.typing import NDArray


# Image processing
import cv2
from PIL import Image
from skimage import data, exposure
from skimage.feature import hog, local_binary_pattern


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


# Scikit-learn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Clustering and dimensionality reduction
import hdbscan
import umap


# PyTorch and TorchVision
import torch
import torchvision.transforms.v2 as v2
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet50_Weights,
    mobilenet_v3_small,
    resnet50,
)


# Hugging Face Transformers
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
)
from transformers.image_utils import load_image


# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2


# HTTP requests
import requests



#TODO
# Take a step back and read all of the code fully understand how it all works. write down each class on a piece of paper. 
# set up methods to run all of these unsuperviseed learning models 
# replace all hard coded variables with dynamic ones

#need to make this collect images file and turn it into a static method. lotta repeated cide for no good reason. 
#need to run quizes on this code to ensure I fully understand how each part works. 
# write down the steps for each method and ensure you understand the flow of data from image collection, preprocessing, embedding extraction, and clustering.
# This class is responsible for managing the deep embedding extraction process for different models (ResNet, CLIP, MobileNetV3, DINOv3). It handles the collection of image files, preprocessing, and extraction of embeddings using the specified model.

""" min_cluster_size = max(15, round(img_count * 0.05))   # knob: lower = more, smaller clusters
min_samples = 7 """
#add a more pre preocessing stpe. 
class PreProcessingImages:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

    def preprocess_file(self, file_path: Path):
        file_path = Path(file_path)

        if file_path.suffix.lower().strip() not in self.valid_extensions:
            print(f"Skipping non-image file: {file_path.name}")
            return None

        image = cv2.imread(str(file_path))

        if image is None:
            print(f"Could not read: {file_path.name}")
            return None

        # Keep BGR for OpenCV descriptor functions
        resized_bgr = cv2.resize(image, self.image_size)

        return resized_bgr, file_path.name



class DeepEmbedding:
    def __init__(
        self,
        File_Name: Path,
        Files_Training: Path | list[Path] | None = None,
        Embeddings=None,
        model_config: dict | None = None
    ):
        self.File_Name = Path(File_Name)
        self.Files_Training = Files_Training
        self.Embeddings = Embeddings if Embeddings is not None else []
        self.model_config = model_config or {
    "resnet": {
        "batch_size": 32,
        "epochs": 200,
        "data_augmentation": True,
        "num_classes": 10,
        "subtract_pixel_mean": True,
        "n": 3,
        "version": 1,
        "image_size": (224, 224),
        "embedding_size": 2048
    },

    "clip": {
        "model_name": "openai/clip-vit-base-patch32",
        "image_size": (224, 224),
        "embedding_size": 512,
        "batch_size": 32
    },

    "MobileNetV3": {
        "model_name": "mobilenet_v3_small",
        "image_size": (224, 224),
        "embedding_size": 576,
        "batch_size": 32
    },
    "DINOv3" : {
        "model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "image_size": (224, 224),
        "embedding_size": 384,
        "batch_size": 32
    }
    
    }

    
    def res_net(self, Files_Training=None, Embeddings: list | None = None):


        
        resnet_config = self.model_config["resnet"]

        def hyperparameteres():
            batch_size = resnet_config["batch_size"]
            epochs = resnet_config["epochs"]
            data_augmentation = resnet_config["data_augmentation"]
            num_classes = resnet_config["num_classes"]
            subtract_pixel_mean = resnet_config["subtract_pixel_mean"]
            n = resnet_config["n"]
            version = resnet_config["version"]

            if version == 1:
                depth = n * 6 + 2
            elif version == 2:
                depth = n * 9 + 2
            else:
                raise ValueError("version must be 1 or 2")

            model_type = "ResNet %dv%d" % (depth, version)

            return {
                "batch_size": batch_size,
                "epochs": epochs,
                "data_augmentation": data_augmentation,
                "num_classes": num_classes,
                "subtract_pixel_mean": subtract_pixel_mean,
                "n": n,
                "version": version,
                "depth": depth,
                "model_type": model_type
            }

        def collect_image_files(data_folder: Path) -> list[Path]:
            data_folder = Path(data_folder)

            image_extensions = {
                ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
            }

            if not data_folder.exists():
                raise FileNotFoundError(f"Data folder does not exist: {data_folder}")

            if not data_folder.is_dir():
                raise NotADirectoryError(f"Data path is not a folder: {data_folder}")

            image_files = [
                file_path
                for file_path in data_folder.rglob("*")
                if file_path.is_file()
                and file_path.suffix.lower() in image_extensions
            ]

            if len(image_files) == 0:
                raise ValueError("No image files found.")

            print(f"Images found: {len(image_files)}")

            return image_files

        #remove this
        def load_and_preprocess_images(
            image_files: list[Path],
            image_size=(224, 224)
        ) -> list[dict]:

            processed_images = []

            image_size = tuple(image_size[:2])

            preprocessor = PreProcessingImages(image_size=image_size)

            for file_path in image_files:
                file_path = Path(file_path)

                result = preprocessor.preprocess_file(file_path)

                if result is None:
                    continue

                resized_bgr, name = result

                image_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

                processed_images.append({
                    "file_name": name,
                    "file_path": file_path,
                    "image": image_rgb
                })

            print(f"Images processed: {len(processed_images)}")

            return processed_images

        # Step 3: load pretrained ResNet model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()

        resnet_model = resnet50(weights=weights)

        #this removes the classification layer in order to just extract embeddings. 
        embedding_extractor = torch.nn.Sequential(
            *list(resnet_model.children())[:-1]
        )

        embedding_extractor = embedding_extractor.to(device)
        embedding_extractor.eval()

        def extract_embedding(image_rgb, file_name):
            try:
                pil_img = Image.fromarray(image_rgb).convert("RGB")

                img_tensor = preprocess(pil_img)
                img_tensor = img_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    features = embedding_extractor(img_tensor)

                embedding = features.squeeze().cpu().numpy()

                return embedding

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                return None

        # Step 1 and 2
        if Files_Training is None:
            Files_Training = self.File_Name
        #loop through the directory and subdir in order to 
        if isinstance(Files_Training, (str, Path)):
            image_files = collect_image_files(Files_Training)
        else:
            image_files = [Path(file) for file in Files_Training]

        processed_images = load_and_preprocess_images(
            image_files=image_files,
            image_size=resnet_config["image_size"]
        )

        config = hyperparameteres()

        # Step 4: extract embeddings
        embeddings = [] if Embeddings is None else Embeddings
        embedding_file_names = []
        embedding_file_paths = []

        for item in processed_images:
            image_rgb = item["image"]
            file_name = item["file_name"]
            file_path = item["file_path"]

            embedding = extract_embedding(image_rgb, file_name)

            if embedding is None:
                continue

            embeddings.append(embedding)
            embedding_file_names.append(file_name)
            embedding_file_paths.append(file_path)
            print(f"Embedding Processed")

        embeddings_array = np.array(embeddings, dtype=np.float32)

        print(f"Embeddings created: {embeddings_array.shape}")

        self.Embeddings = embeddings_array
        return {
            "model": embedding_extractor,
            "weights": weights,
            "config": config,
            "image_files": image_files,
            "processed_images": processed_images,
            "embeddings": embeddings_array,
            "embedding_file_names": embedding_file_names,
            "embedding_file_paths": embedding_file_paths,
            "device": device
        }
    
    def clip(self, Files_Training=None, Embeddings: list | None = None):
        clip_config = self.model_config["clip"]

        def unwrap(value):
            if isinstance(value, tuple) and len(value) == 1:
                return value[0]
            return value

        def hyperparameters():
            batch_size = unwrap(clip_config.get("batch_size", 32))
            image_size = unwrap(clip_config.get("image_size", (224, 224)))
            embedding_size = unwrap(clip_config.get("embedding_size", 512))
            model_name = unwrap(
                clip_config.get("model_name", "openai/clip-vit-base-patch32")
            )

            clip_name_map = {
                "ViT-B/32": "openai/clip-vit-base-patch32",
                "ViT-B-32": "openai/clip-vit-base-patch32",
                "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
            }

            model_name = clip_name_map.get(model_name, model_name)

            return {
                "batch_size": max(1, int(batch_size)),
                "image_size": image_size,
                "embedding_size": int(embedding_size),
                "model_name": model_name,
            }

        def collect_image_files(data_folder: Path) -> list[Path]:
            data_folder = Path(data_folder)

            image_extensions = {
                ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
            }

            if not data_folder.exists():
                raise FileNotFoundError(f"Data folder does not exist: {data_folder}")

            if not data_folder.is_dir():
                raise NotADirectoryError(f"Data path is not a folder: {data_folder}")

            image_files = sorted([
                file_path
                for file_path in data_folder.rglob("*")
                if file_path.is_file()
                and file_path.suffix.lower() in image_extensions
            ])

            if len(image_files) == 0:
                raise ValueError("No image files found.")

            print(f"Images found: {len(image_files)}")

            return image_files

        def load_pil_image(file_path: Path):
            try:
                with Image.open(file_path) as img:
                    return img.convert("RGB")
            except Exception as e:
                print(f"Could not read image: {file_path.name} | {e}")
                return None

        # -------------------------
        # 1. Config
        # -------------------------
        config = hyperparameters()
        batch_size = config["batch_size"]
        model_name = config["model_name"]

        # -------------------------
        # 2. Device
        # -------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # -------------------------
        # 3. Load CLIP model and processor
        # -------------------------
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        model = model.to(device)
        model.eval()

        # -------------------------
        # 4. Get image files
        # -------------------------
        if Files_Training is None:
            Files_Training = self.File_Name

        if isinstance(Files_Training, (str, Path)):
            image_files = collect_image_files(Files_Training)
        else:
            image_files = [Path(file) for file in Files_Training]

        if len(image_files) == 0:
            raise ValueError("No image files were provided.")

        # -------------------------
        # 5. Extract CLIP embeddings
        # -------------------------
        embedding_batches = []
        embedding_file_names = []
        embedding_file_paths = []

        for start_idx in range(0, len(image_files), batch_size):
            batch_paths = image_files[start_idx:start_idx + batch_size]

            batch_images = []
            valid_paths = []

            for file_path in batch_paths:
                image = load_pil_image(file_path)

                if image is None:
                    continue

                batch_images.append(image)
                valid_paths.append(file_path)

            if len(batch_images) == 0:
                continue

            inputs = processor(
                images=batch_images,
                return_tensors="pt"
            )

            pixel_values = inputs["pixel_values"].to(device)

            with torch.no_grad():
                vision_outputs = model.vision_model(
                    pixel_values=pixel_values
                )

                pooled_output = vision_outputs.pooler_output

                image_features = model.visual_projection(
                    pooled_output
                )

            # Normalize embeddings
            image_features = torch.nn.functional.normalize(
                image_features,
                p=2,
                dim=-1
            )

            batch_embeddings = image_features.cpu().numpy().astype(np.float32)

            embedding_batches.append(batch_embeddings)

            for file_path in valid_paths:
                embedding_file_names.append(file_path.name)
                embedding_file_paths.append(file_path)

            print(
                f"Processed batch {start_idx // batch_size + 1} | "
                f"Total embedded so far: {len(embedding_file_names)}"
            )

        # -------------------------
        # 6. Combine batches
        # -------------------------
        if len(embedding_batches) == 0:
            embeddings_array = np.empty(
                (0, config["embedding_size"]),
                dtype=np.float32
            )
        else:
            embeddings_array = np.vstack(embedding_batches).astype(np.float32)

        print(f"CLIP embeddings created: {embeddings_array.shape}")

        self.Embeddings = embeddings_array

        if embeddings_array.shape[0] > 0:
            config["actual_embedding_size"] = embeddings_array.shape[1]

        # -------------------------
        # 7. Return results
        # -------------------------
        return {
            "model": model,
            "processor": processor,
            "config": config,
            "image_files": image_files,
            "embeddings": embeddings_array,
            "embedding_file_names": embedding_file_names,
            "embedding_file_paths": embedding_file_paths,
            "device": device,
        }
    #this is my custom model using tensotr  flow. 
    def MobileNetV3_USL(
        self,
        Files_Training=None,
        Embeddings: list | None = None
    ):
        model_config = self.model_config["MobileNetV3"]

        def hyperparameters():
            batch_size = model_config["batch_size"]
            image_size = model_config["image_size"]
            embedding_size = model_config["embedding_size"]
            model_name = model_config["model_name"]

            return {
                "batch_size": max(1, int(batch_size)),
                "image_size": tuple(image_size),
                "embedding_size": int(embedding_size),
                "model_name": model_name,
            }

    
        def collect_image_files(data_folder: Path) -> list[Path]:
            data_folder = Path(data_folder)

            image_extensions = {
                ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
            }

            if not data_folder.exists():
                raise FileNotFoundError(f"Data folder does not exist: {data_folder}")

            if not data_folder.is_dir():
                raise NotADirectoryError(f"Data path is not a folder: {data_folder}")

            image_files = sorted([
                file_path
                for file_path in data_folder.rglob("*")
                if file_path.is_file()
                and file_path.suffix.lower() in image_extensions
            ])

            if len(image_files) == 0:
                raise ValueError("No image files found.")

            print(f"Images found: {len(image_files)}")

            return image_files

        def load_pil_image(file_path: Path):
            try:
                with Image.open(file_path) as img:
                    return img.convert("RGB")
            except Exception as e:
                print(f"Could not read image: {file_path.name} | {e}")
                return None 
            
        config = hyperparameters()
        batch_size = config["batch_size"]
        model_name = config["model_name"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")


        weights = MobileNet_V3_Small_Weights.DEFAULT
        preprocess = weights.transforms()

        model = mobilenet_v3_small(weights=weights)
        model = model.to(device)

        model.eval()


        if Files_Training is None:
            Files_Training = self.File_Name

        if isinstance(Files_Training, (str, Path)):
            image_files = collect_image_files(Files_Training)
        else:
            image_files = [Path(file) for file in Files_Training]

        if len(image_files) == 0:
            raise ValueError("No image files were provided.")

        embedding_batches = []
        embedding_file_names = []
        embedding_file_paths = []

        for start_idx in range(0, len(image_files), batch_size):
            batch_paths = image_files[start_idx:start_idx + batch_size]

            batch_images = []
            valid_paths = []

            for file_path in batch_paths:
                image = load_pil_image(file_path)

                if image is None:
                    continue

                batch_images.append(image)
                valid_paths.append(file_path)

            if len(batch_images) == 0:
                continue

            processed_images = [
                preprocess(image)
                for image in batch_images
            ]

            pixel_values = torch.stack(processed_images).to(device)

            with torch.no_grad():
                image_features = model.features(pixel_values)
                image_features = model.avgpool(image_features)
                image_features = torch.flatten(image_features, 1)

            batch_embeddings = image_features.cpu().numpy().astype(np.float32)

            embedding_batches.append(batch_embeddings)

            for file_path in valid_paths:
                embedding_file_names.append(file_path.name)
                embedding_file_paths.append(file_path)

            print(
                f"Processed batch {start_idx // batch_size + 1} | "
                f"Total embedded so far: {len(embedding_file_names)}"
            )


        if len(embedding_batches) == 0:
            embeddings_array = np.empty(
                (0, config["embedding_size"]),
                dtype=np.float32
            )
        else:
            embeddings_array = np.vstack(embedding_batches).astype(np.float32)

        print(f"MObile_net_v3_small embeddings created: {embeddings_array.shape}")

        self.Embeddings = embeddings_array

        if embeddings_array.shape[0] > 0:
            config["actual_embedding_size"] = embeddings_array.shape[1]

        # -------------------------
        # 7. Return results
        # -------------------------
        return {
            "model": model,
            "config": config,
            "image_files": image_files,
            "embeddings": embeddings_array,
            "embedding_file_names": embedding_file_names,
            "embedding_file_paths": embedding_file_paths,
            "device": device,
        }
    
    def DINOv3_USL(self , Files_Training=None , Embeddings: list | None = None):
        model_config = self.model_config["DINOv3"]

        def hyperparameters():
            batch_size = model_config["batch_size"]
            image_size = model_config["image_size"]
            embedding_size = model_config["embedding_size"]
            model_name = model_config["model_name"]

            return {
                "batch_size": max(1, int(batch_size)),
                "image_size": tuple(image_size),
                "embedding_size": int(embedding_size),
                "model_name": model_name,
            }
            
        def collect_image_files(data_folder: Path) -> list[Path]:
            data_folder = Path(data_folder)

            image_extensions = {
                ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
            }

            if not data_folder.exists():
                raise FileNotFoundError(f"Data folder does not exist: {data_folder}")

            if not data_folder.is_dir():
                raise NotADirectoryError(f"Data path is not a folder: {data_folder}")

            image_files = sorted([
                file_path
                for file_path in data_folder.rglob("*")
                if file_path.is_file()
                and file_path.suffix.lower() in image_extensions
            ])

            if len(image_files) == 0:
                raise ValueError("No image files found.")

            print(f"Images found: {len(image_files)}")

            return image_files

        def load_pil_image(file_path: Path):
            try:
                with Image.open(file_path) as img:
                    return img.convert("RGB")
            except Exception as e:
                print(f"Could not read image: {file_path.name} | {e}")
                return None 

        config = hyperparameters()
        batch_size = config["batch_size"]
        model_name = config["model_name"]

        # Load the processor and the model
        model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")


        if Files_Training is None:
            Files_Training = self.File_Name

        if isinstance(Files_Training, (str, Path)):
            image_files = collect_image_files(Files_Training)
        else:
            image_files = [Path(file) for file in Files_Training]

        if len(image_files) == 0:
            raise ValueError("No image files were provided.")

        embedding_batches = []
        embedding_file_names = []
        embedding_file_paths = []

        for start_idx in range(0, len(image_files), batch_size):
            batch_paths = image_files[start_idx:start_idx + batch_size]

            batch_images = []
            valid_paths = []

            for file_path in batch_paths:
                image = load_pil_image(file_path)

                if image is None:
                    continue

                batch_images.append(image)
                valid_paths.append(file_path)

            if not batch_images:
                continue

            # Process the entire batch using DINOv3's processor
            inputs = processor(
                images=batch_images,
                return_tensors="pt"
            ).to(device)

            # Run the images through DINOv3
            with torch.inference_mode():
                outputs = model(**inputs)

                # First token is the whole-image CLS embedding
                cls_token = outputs.last_hidden_state[:, 0, :]

                # Optional but useful for clustering and similarity
                cls_token = torch.nn.functional.normalize(
                    cls_token,
                    p=2,
                    dim=1
                )

            batch_embeddings = (
                cls_token
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            embedding_batches.append(batch_embeddings)

            for file_path in valid_paths:
                embedding_file_names.append(file_path.name)
                embedding_file_paths.append(file_path)

            print(
                f"Processed batch {start_idx // batch_size + 1} | "
                f"Total embedded so far: {len(embedding_file_names)}"
            )

        return {
            "model": model,
            "config": config,
            "image_files": image_files,
            "embeddings": np.vstack(embedding_batches).astype(np.float32) if embedding_batches else np.empty((0, config["embedding_size"]), dtype=np.float32),
            "embedding_file_names": embedding_file_names,
            "embedding_file_paths": embedding_file_paths,
            "device": device,
        }
class Descriptors:
    FeatureDict = dict[str, list[Any]]

    def __init__(
        self,
        Image_Name: str = "",
        img_file: Path | None = None,
        Features_HOG: FeatureDict | None = None,
        HSV_HISTOGRAMS: FeatureDict | None = None,
        LBP: FeatureDict | None = None,
        ORB_KP: FeatureDict | None = None,
        ORB_DESC: FeatureDict | None = None,
        LAB_HISTOGRAMS : FeatureDict | None = None,
        SURF_KP : FeatureDict | None = None,
        SURF_DESC : FeatureDict | None = None
    ):
        self.Image = Image_Name
        self.file_name = img_file

        self.Features_HOG = Features_HOG or {
            "File_Name": [],
            "HOG_Values": []
        }

        self.SURF_KP = SURF_KP or {
            "File_Name": [],
            "SURF_Values": []
        }

        self.SURF_DESC = SURF_DESC or {
            "File_Name": [],
            "SURF_Values": []
        }

        self.HSV_HISTOGRAMS = HSV_HISTOGRAMS or {
            "File_Name": [],
            "HSV_Values": []
        }
        self.LAB_HISTOGRAMS = LAB_HISTOGRAMS or {
            "File_Name": [],
            "LAB_Values": []
        }

        self.LBP = LBP or {
            "File_Name": [],
            "LBP_Values": []
        }

        self.ORB_KP = ORB_KP or {
            "File_Name": [],
            "ORB_KP_Values": []
        }

        self.ORB_DESC = ORB_DESC or {
            "File_Name": [],
            "ORB_DESC_Values": []
        }

    def HOG_Data(self, img_file, Image_Name):
        try:
            gray = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)

            fd = hog(
                gray,
                orientations=9,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                visualize=False,
                feature_vector=True
            )

            fd = fd.astype(np.float32)

            self.Features_HOG["File_Name"].append(Image_Name)
            self.Features_HOG["HOG_Values"].append(fd)

        except Exception as e:
            print(f"Error processing HOG for file: {Image_Name}\nError: {e}")

        return self.Features_HOG
    
    def SURF_data(self , img_file , Image_Name):
        try:
            gray = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

            keypoints, descriptors = surf.detectAndCompute(gray, None)
            
            if descriptors is None or len(descriptors) == 0:
                desc_vector = np.zeros(32, dtype=np.float32)
                kp_count = 0
            else:
                desc_vector = descriptors.mean(axis=0).astype(np.float32)
                kp_count = len(keypoints)

            
            self.SURF_KP["File_Name"].append(Image_Name)
            self.SURF_KP["ORB_KP_Values"].append(np.array([kp_count], dtype=np.float32))


            
            self.SURF_DESC["File_Name"].append(Image_Name)
            self.SURF_DESC["ORB_DESC_Values"].append(desc_vector)

        except Exception as e:
            print(f"Error with SURF{e}")

        return self.ORB_KP, self.ORB_DESC
    
    def SIFT(self , img_file , Image_Name):
        pass
    
    def HSV_HISTOGRAM(self, img_file, Image_Name):
        try:
            hsv_img = cv2.cvtColor(img_file, cv2.COLOR_BGR2HSV)

            hist_h = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv_img], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])

            hist = np.concatenate([
                hist_h.flatten(),
                hist_s.flatten(),
                hist_v.flatten()
            ]).astype(np.float32)

            # Normalize so large images / bright images do not dominate
            hist = hist / (hist.sum() + 1e-8)

            self.LAB_HISTOGRAMS["File_Name"].append(Image_Name)
            self.LAB_HISTOGRAMS["HSV_Values"].append(hist)

        except Exception as e:
            print(f"Error processing HSV for file: {Image_Name}\nError: {e}")

        return self.HSV_HISTOGRAMS
    
    def LAB_HISTOGRAM(self, img_file, Image_Name):
        try:
            hsv_img = cv2.cvtColor(img_file, cv2.COLOR_BGR2LAB)

            hist_l = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])
            hist_a = cv2.calcHist([hsv_img], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])

            hist = np.concatenate([
                hist_l.flatten(),
                hist_a.flatten(),
                hist_b.flatten()
            ]).astype(np.float32)

            # Normalize so large images / bright images do not dominate
            hist = hist / (hist.sum() + 1e-8)

            self.LAB_HISTOGRAMS["File_Name"].append(Image_Name)
            self.LAB_HISTOGRAMS["LAB_Values"].append(hist)

        except Exception as e:
            print(f"Error processing HSV for file: {Image_Name}\nError: {e}")

        return self.LAB_HISTOGRAMS

    def Local_binary_Patters(self, img_file, Image_Name):
        try:
            gray = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)

            lbp = local_binary_pattern(
                gray,
                P=8,
                R=1,
                method="uniform"
            )

            # Uniform LBP with P=8 gives values roughly 0-9
            hist, _ = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, 11),
                range=(0, 10)
            )

            hist = hist.astype(np.float32)
            hist = hist / (hist.sum() + 1e-8)

            self.LBP["File_Name"].append(Image_Name)
            self.LBP["LBP_Values"].append(hist)

        except Exception as e:
            print(f"Error processing LBP for file: {Image_Name}\nError: {e}")

        return self.LBP

    def ORB_Descriptors_And_Keypoints(self, Image_Name, img_file):
        try:
            gray = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)

            orb = cv2.ORB_create(nfeatures=500)
            keypoints, descriptors = orb.detectAndCompute(gray, None)

            if descriptors is None or len(descriptors) == 0:
                desc_vector = np.zeros(32, dtype=np.float32)
                kp_count = 0
            else:
                desc_vector = descriptors.mean(axis=0).astype(np.float32)
                kp_count = len(keypoints)

            self.ORB_KP["File_Name"].append(Image_Name)
            self.ORB_KP["ORB_KP_Values"].append(np.array([kp_count], dtype=np.float32))

            self.ORB_DESC["File_Name"].append(Image_Name)
            self.ORB_DESC["ORB_DESC_Values"].append(desc_vector)

        except Exception as e:
            print(f"Error processing ORB for file: {Image_Name}\nError: {e}")

        return self.ORB_KP, self.ORB_DESC

    def combining_descriptors(self):
            sources = {
                "HOG_Values": (self.Features_HOG,   "HOG_Values"),
                "HSV_Values": (self.HSV_HISTOGRAMS, "HSV_Values"),
                "LBP_Values": (self.LBP,            "LBP_Values"),
                "ORB_DESC":   (self.ORB_DESC,       "ORB_DESC_Values"),
                "ORB_KP":     (self.ORB_KP,         "ORB_KP_Values"),
            }

            # anchor on the first source that actually has data
            file_names, n = [], 0
            for src, _ in sources.values():
                if len(src["File_Name"]) > 0:
                    file_names = src["File_Name"]
                    n = len(file_names)
                    break

            combined = {"File_Name": file_names}
            for out_key, (src, src_key) in sources.items():
                values = src.get(src_key, [])
                if n > 0 and len(values) == n:
                    combined[out_key] = values
            return combined

class DescriptorAnalysis:

    def __init__(self, combined: dict, img_count: int , desc_list:list[NDArray]):
        self.combined = combined
        self.img_count = len(combined["File_Name"])
        self.desc_list = desc_list

    #this needs to change and not be staic I need  a dynmaic list of 
    def create_feature_vectors(self):
        file_names = self.combined["File_Name"]

        feature_keys = [
            k for k in ["HOG_Values", "HSV_Values", "LBP_Values", "ORB_DESC", "ORB_KP"]
            if k in self.combined
        ]

        vectors = []
        for i in range(len(file_names)):
            parts = [np.ravel(self.combined[key][i]) for key in feature_keys]
            vectors.append(np.concatenate(parts))

        X = np.array(vectors)
        return file_names, X

    def scale_features(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def reduce_with_pca(self, X_scaled):
        n_samples = X_scaled.shape[0]
        n_features = X_scaled.shape[1]

        n_components = min(50, n_samples - 1, n_features)

        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        print(f"PCA components used: {n_components}")
        print(f"PCA variance kept: {pca.explained_variance_ratio_.sum():.3f}")

        return X_pca

    def run_umap(self, X_scaled, labels=None):
        X_pca = self.reduce_with_pca(X_scaled)

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.05,
            metric="euclidean",
            random_state=42
        )

        embedding = reducer.fit_transform(X_pca)

        plt.figure(figsize=(8, 5))

        if labels is None:
            plt.scatter(embedding[:, 0], embedding[:, 1], s=5)
        else:
            plt.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels,
                cmap="Spectral",
                s=5
            )

        plt.title("UMAP projection of image features")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.show()

        return embedding, X_pca


    def Clusters(
        self,
        embedding,
        img_count: int | None = None,
        return_labels: bool = False
    ):
        # -------------------------
        # 1. Safety checks
        # -------------------------
        if embedding is None or len(embedding) == 0:
            print("No embedding data provided.")
            return (0, None, None) if return_labels else 0

        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.ndim != 2:
            raise ValueError("Embedding must be a 2D array.")

        if embedding.shape[1] != 2:
            raise ValueError("Embedding must be 2D. Expected shape: (n_samples, 2)")

        if np.isnan(embedding).any() or np.isinf(embedding).any():
            raise ValueError("Embedding contains NaN or infinite values.")

        n_samples = embedding.shape[0]

        if img_count is None:
            img_count = n_samples

        if img_count != n_samples:
            print(f"Warning: img_count={img_count}, but embedding has {n_samples} rows.")
            img_count = n_samples

        if img_count < 10:
            print("Too few images for reliable clustering.")
            return (0, None, None) if return_labels else 0

        # -------------------------
        # 2. Build HDBSCAN search grid
        # -------------------------
        if img_count < 200:
            mcs_grid = [3, 5, 8, 10]
            ms_grid = [1, 2, 3]
        elif img_count < 1000:
            mcs_grid = [5, 10, 15, 25, 40]
            ms_grid = [1, 3, 5, 10]
        else:
            mcs_grid = sorted(set([
                max(10, round(img_count * 0.002)),
                max(15, round(img_count * 0.005)),
                max(25, round(img_count * 0.010)),
                max(40, round(img_count * 0.020)),
                max(75, round(img_count * 0.050)),
            ]))
            ms_grid = [1, 5, 10, 15]

        best = {
            "score": -999,
            "silhouette": None,
            "labels": None,
            "mcs": None,
            "ms": None,
            "cluster_count": 0,
            "noise_count": None,
            "noise_ratio": None
        }

        print("\nSearching HDBSCAN settings...")
        print("-" * 70)

        # -------------------------
        # 3. Grid search
        # -------------------------
        for mcs in mcs_grid:
            for ms in ms_grid:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    metric="euclidean",
                    cluster_selection_method="eom"
                )

                labels = clusterer.fit_predict(embedding)

                non_noise_mask = labels != -1
                clustered_points = non_noise_mask.sum()
                noise_count = img_count - clustered_points
                noise_ratio = noise_count / img_count

                unique_non_noise = set(labels[non_noise_mask])
                cluster_count = len(unique_non_noise)

                if cluster_count < 2:
                    continue

                if clustered_points < 10:
                    continue

                # Avoid scoring thousands of points if dataset is large
                if clustered_points > 3000:
                    sample_size = 3000
                else:
                    sample_size = clustered_points

                try:
                    sil = silhouette_score(
                        embedding[non_noise_mask],
                        labels[non_noise_mask],
                        sample_size=sample_size,
                        random_state=42
                    )
                except Exception as e:
                    print(f"Skipping mcs={mcs}, ms={ms}. Silhouette error: {e}")
                    continue

                # -------------------------
                # 4. Penalize bad cluster behavior
                # -------------------------
                cluster_sizes = [
                    np.sum(labels == label)
                    for label in unique_non_noise
                ]

                smallest_cluster = min(cluster_sizes)
                largest_cluster = max(cluster_sizes)

                tiny_cluster_penalty = 0
                if smallest_cluster < max(5, round(img_count * 0.002)):
                    tiny_cluster_penalty = 0.05

                too_many_clusters_penalty = 0
                if cluster_count > 50:
                    too_many_clusters_penalty = 0.10
                elif cluster_count > 25:
                    too_many_clusters_penalty = 0.05

                noise_penalty = noise_ratio * 0.25

                final_score = sil - noise_penalty - tiny_cluster_penalty - too_many_clusters_penalty

                print(
                    f"mcs={mcs:>4} ms={ms:>2} | "
                    f"clusters={cluster_count:>3} | "
                    f"noise={noise_count:>4} ({noise_ratio:.1%}) | "
                    f"smallest={smallest_cluster:>4} | "
                    f"largest={largest_cluster:>4} | "
                    f"sil={sil:.3f} | "
                    f"score={final_score:.3f}"
                )

                if final_score > best["score"]:
                    best.update(
                        score=final_score,
                        silhouette=sil,
                        labels=labels,
                        mcs=mcs,
                        ms=ms,
                        cluster_count=cluster_count,
                        noise_count=noise_count,
                        noise_ratio=noise_ratio
                    )

        # -------------------------
        # 5. Handle failure
        # -------------------------
        if best["labels"] is None:
            print("\nNo HDBSCAN configuration produced usable clusters.")
            return (0, None, None) if return_labels else 0

        cluster_labels = best["labels"]
        cluster_count = best["cluster_count"]

        print("\nBest HDBSCAN configuration")
        print("-" * 70)
        print(f"min_cluster_size: {best['mcs']}")
        print(f"min_samples:      {best['ms']}")
        print(f"clusters:         {best['cluster_count']}")
        print(f"noise count:      {best['noise_count']}")
        print(f"noise ratio:      {best['noise_ratio']:.1%}")
        print(f"silhouette:       {best['silhouette']:.3f}")
        print(f"final score:      {best['score']:.3f}")

        # -------------------------
        # 6. Print final label counts
        # -------------------------
        labels, counts = np.unique(cluster_labels, return_counts=True)

        print("\nFinal cluster label counts")
        print("-" * 70)

        for label, count in zip(labels, counts):
            name = "Noise" if label == -1 else f"Cluster {label}"
            print(f"{name}: {count}")

        print(f"\nTotal points accounted for: {counts.sum()}")

        # -------------------------
        # 7. Plot
        # -------------------------
        plt.figure(figsize=(12, 7))

        unique_labels = sorted(set(cluster_labels))

        for label in unique_labels:
            mask = cluster_labels == label

            label_name = "Noise" if label == -1 else f"Cluster {label}"

            if label == -1:
                plt.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    s=8,
                    alpha=0.35,
                    label=label_name
                )
            else:
                plt.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    s=14,
                    alpha=0.85,
                    label=label_name
                )

        plt.title(
            f"HDBSCAN Clustering Results | "
            f"clusters={cluster_count}, "
            f"mcs={best['mcs']}, "
            f"ms={best['ms']}, "
            f"noise={best['noise_ratio']:.1%}"
        )
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")

        if cluster_count <= 15:
            plt.legend()
        else:
            plt.legend([], [], frameon=False)

        plt.tight_layout()
        plt.show()

        if return_labels:
            return cluster_count, cluster_labels, best

        return cluster_count

class K_Means_Comparision_sequester:
    """Cluster images and copy them into folders by cluster label."""

    def __init__(
        self,
        K_clusters_LB: int,
        K_clusters_UB: int,
        IMG_FEATURES,
        IMG_NAmes,
        IMG_PATHS,
        Parent_folder_name: str,
        Output_folder: Path
    ):
        self.K_clusters_LB = K_clusters_LB
        self.K_clusters_UB = K_clusters_UB
        self.IMG_FEATURES = IMG_FEATURES
        self.IMG_NAmes = IMG_NAmes
        self.IMG_PATHS = IMG_PATHS
        self.Parent_folder_name = Parent_folder_name
        self.Output_folder = Path(Output_folder)

    def clustering(
        self,
        K_clusters: int = 2,
        RS: int = 32,
        NI: int = 8
    ):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.IMG_FEATURES)

        kmeans = KMeans(
            n_clusters=K_clusters,
            random_state=RS,
            n_init=NI
        )

        labels = kmeans.fit_predict(X_scaled)

        # Example:
        # USPL / CLIP / 0
        # USPL / CLIP / 1
        # USPL / RESNET / 0
        parent = self.Output_folder / self.Parent_folder_name
        parent.mkdir(parents=True, exist_ok=True)

        for label in set(labels):
            cluster_folder = parent / str(label)
            cluster_folder.mkdir(parents=True, exist_ok=True)

        for fp, label in zip(self.IMG_PATHS, labels):
            fp = Path(fp)

            if not fp.exists():
                print(f"File does not exist, skipping: {fp}")
                continue

            destination = parent / str(label) / fp.name

            # Copy instead of move
            shutil.copy2(str(fp), str(destination))

        return labels
    

class excution():
    def __init__(self , data_folder , output_folder):
        
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
    
    def validate_folders(self):
        if not self.data_folder.exists():
            raise FileNotFoundError(
                f"Data folder does not exist: {self.data_folder}"
            )

        if not self.data_folder.is_dir():
            raise NotADirectoryError(
                f"Data path is not a folder: {self.data_folder}"
            )

        if self.output_folder.exists() and not self.output_folder.is_dir():
            raise NotADirectoryError(
                f"Output path is not a folder: {self.output_folder}"
            )

        self.output_folder.mkdir(
            parents=True,
            exist_ok=True
        )
    def run_embedding_cluster_pipeline(self , method_name: str,embedding_results: dict, output_folder: Path):
            X = embedding_results["embeddings"]
            file_names = embedding_results["embedding_file_names"]
            file_paths = embedding_results["embedding_file_paths"]

            if X is None or len(X) == 0:
                raise ValueError(f"No {method_name} embeddings were created.")

            print(f"\n===== {method_name} =====")
            print(f"Images embedded: {len(file_names)}")
            print(f"{method_name} feature matrix shape: {X.shape}")

            combined = {
                "File_Name": file_names,
                "File_Path": file_paths
            }
            #
            analysis = DescriptorAnalysis(
                combined=combined,
                img_count=len(file_names),
                desc_list=[]
            )
            X_scaled = analysis.scale_features(X)

            umap_result = analysis.run_umap(X_scaled)

            if isinstance(umap_result, tuple):
                embedding = umap_result[0]
            else:
                embedding = umap_result

            cluster_result = analysis.Clusters(
                embedding=embedding,
                img_count=len(file_names)
            )

            if isinstance(cluster_result, tuple):
                cluster_count = cluster_result[0]
            else:
                cluster_count = cluster_result

            print(f"Final {method_name} cluster count: {cluster_count}")

            kmeans_runner = K_Means_Comparision_sequester(
                K_clusters_LB=2,
                K_clusters_UB=cluster_count,
                IMG_FEATURES=embedding,
                IMG_NAmes=file_names,
                IMG_PATHS=file_paths,
                Parent_folder_name=method_name,
                Output_folder=output_folder
            )

            labels = kmeans_runner.clustering(
                K_clusters=cluster_count,
                RS=32,
                NI=8
            )

            print(f"{method_name} KMeans clustering complete.")

            return labels



    #this needs to be more dyanmic  I need to pass in the list 
    def embedding_extraction(self, methods: list[str] | None = None):
        deep = DeepEmbedding(
            File_Name=self.data_folder,
            Embeddings=[]
        )

        available_methods = {
            "ResNet": deep.res_net,
            "CLIP": deep.clip,
            "MobileNetV3": deep.MobileNetV3_USL,
            "DINOv3": deep.DINOv3_USL,
        }

        # Run all four unless specific methods were supplied
        selected_methods = methods or list(available_methods.keys())

        clustering_results = {}
        failed_methods = {}

        for method_name in selected_methods:
            if method_name not in available_methods:
                print(f"Unknown embedding method: {method_name}")
                failed_methods[method_name] = "Method does not exist."
                continue

            print(f"\nRunning {method_name} embedding extraction...")

            try:
                embedding_method = available_methods[method_name]

                method_result = embedding_method(
                    Files_Training=self.data_folder,
                    Embeddings=[]
                )

                labels = self.run_embedding_cluster_pipeline(
                    method_name=method_name,
                    embedding_results=method_result,
                    output_folder=self.output_folder
                )

                clustering_results[method_name] = labels

                print(f"{method_name} completed successfully.")

            except Exception as error:
                failed_methods[method_name] = str(error)
                print(f"{method_name} failed: {error}")

            finally:
                # Helps release GPU memory before loading the next model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return {
            "clustering_results": clustering_results,
            "failed_methods": failed_methods,
        }



print(inspect.getmembers(DeepEmbedding, predicate=inspect.isfunction))  


if __name__ == "__main__":
    test = excution(
        data_folder=Path(""),
        output_folder=Path(r"")
    )

    try:
        test.validate_folders()

        results = test.embedding_extraction(
            methods=[
                "ResNet",
                "CLIP",
                "MobileNetV3",
                "DINOv3"
            ]
        )

        print("\n===== FINAL RESULTS =====")

        for method_name in results["clustering_results"]:
            print(f"{method_name}: completed successfully")

        for method_name, error in results["failed_methods"].items():
            print(f"{method_name}: failed — {error}")

    except Exception as error:
        print(f"Pipeline failed: {error}")