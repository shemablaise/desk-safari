# utils/model_tflite.py
import numpy as np
from PIL import Image, ImageOps
import cv2
import os
import tensorflow as tf


class TFLiteObjectDetector:
    def __init__(self, model_path='models/keras_model.h5',
                 labels_path='models/labels.txt'):
        """Initialize the TFLite model"""

        print("=" * 50)
        print("🔍 Initializing TFLite ObjectDetector")
        print("=" * 50)

        # Get absolute paths
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.normpath(os.path.join(self.project_dir, model_path))
        self.labels_path = os.path.normpath(os.path.join(self.project_dir, labels_path))

        print(f"📁 Model path: {self.model_path}")
        print(f"📁 Labels path: {self.labels_path}")

        # Check if files exist
        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found!")
            self.interpreter = None
            return

        if not os.path.exists(self.labels_path):
            print(f"❌ Labels file not found!")
            self.interpreter = None
            return

        print(f"✅ Model file size: {os.path.getsize(self.model_path) / (1024 * 1024):.2f} MB")

        try:
            # Load TFLite model
            print("🔄 Loading TFLite model...")
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()

            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            print(f"✅ Model loaded successfully!")
            print(f"📊 Input shape: {self.input_details[0]['shape']}")
            print(f"📊 Output shape: {self.output_details[0]['shape']}")

            # Get expected input size
            self.input_height = self.input_details[0]['shape'][1]
            self.input_width = self.input_details[0]['shape'][2]

            # Load labels
            with open(self.labels_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            print(f"✅ Labels loaded: {self.class_names}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.interpreter = None

    def predict(self, image):
        """Predict using TFLite"""

        if self.interpreter is None:
            return "Model not loaded", 0.0

        try:
            # Prepare image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)

            # Resize image
            image = image.convert('RGB')
            image = image.resize((self.input_width, self.input_height))
            image_array = np.asarray(image)

            # Normalize to 0-1 (common for TFLite models)
            input_data = image_array.astype(np.float32) / 255.0

            # Add batch dimension
            input_data = np.expand_dims(input_data, axis=0)

            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

            # Run inference
            self.interpreter.invoke()

            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions = output_data[0]

            # Get highest confidence
            index = np.argmax(predictions)
            confidence = float(predictions[index])

            # Get class name
            class_name = self.class_names[index] if index < len(self.class_names) else f"Class {index}"

            # Clean class name (remove number if present)
            if ' ' in class_name:
                parts = class_name.split(' ', 1)
                if len(parts) > 1 and parts[0].isdigit():
                    class_name = parts[1]

            print(f"📊 Prediction: {class_name} ({confidence:.2%})")

            return class_name, confidence

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return "Error", 0.0