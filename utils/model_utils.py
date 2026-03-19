# utils/model_utils.py
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)


class ObjectDetector:
    def __init__(self, model_path='models/keras_model.h5',
                 labels_path='models/labels.txt'):
        """Initialize the Teachable Machine model using official code"""

        print("=" * 50)
        print("🔍 Initializing ObjectDetector")
        print("=" * 50)

        # Get the absolute path to the project directory
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"📁 Project directory: {self.project_dir}")

        # Build absolute paths
        self.model_path = os.path.join(self.project_dir, model_path)
        self.labels_path = os.path.join(self.project_dir, labels_path)

        # Normalize paths for Windows
        self.model_path = os.path.normpath(self.model_path)
        self.labels_path = os.path.normpath(self.labels_path)

        print(f"📁 Looking for model at: {self.model_path}")
        print(f"📁 Looking for labels at: {self.labels_path}")

        # Check if files exist
        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found at: {self.model_path}")

            # List all files in the models directory to help debug
            models_dir = os.path.join(self.project_dir, 'models')
            if os.path.exists(models_dir):
                print(f"\n📁 Files in models folder:")
                for file in os.listdir(models_dir):
                    file_path = os.path.join(models_dir, file)
                    size = os.path.getsize(file_path) / 1024  # KB
                    print(f"   - {file} ({size:.2f} KB)")

            self.model = None
            return

        print(f"✅ Model file found! Size: {os.path.getsize(self.model_path) / (1024 * 1024):.2f} MB")

        # Check labels file
        if not os.path.exists(self.labels_path):
            print(f"❌ Labels file not found at: {self.labels_path}")
            self.model = None
            return

        print(f"✅ Labels file found!")

        try:
            # Load the model
            print("🔄 Loading model...")
            self.model = load_model(self.model_path, compile=False)
            print(f"✅ Model loaded successfully!")

            # Load the labels
            with open(self.labels_path, "r") as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"✅ Labels loaded: {self.class_names}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.class_names = []

    def predict(self, image):
        """Predict using exact Teachable Machine methodology"""

        if self.model is None:
            return "Model not loaded", 0.0

        try:
            # Create the array of the right shape
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            # Handle different input types
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)

            # Ensure image is RGB
            image = image.convert("RGB")

            # Resize and crop from center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            # Turn into numpy array
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Load into array
            data[0] = normalized_image_array

            # Predict
            prediction = self.model.predict(data, verbose=0)

            # Get the prediction
            index = np.argmax(prediction)
            class_name = self.class_names[index]
            confidence_score = float(prediction[0][index])

            # Clean the class name
            if ' ' in class_name:
                parts = class_name.split(' ', 1)
                if len(parts) > 1 and parts[0].isdigit():
                    class_name = parts[1]

            print(f"📊 Prediction: {class_name} ({confidence_score:.2%})")

            return class_name, confidence_score

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Error", 0.0