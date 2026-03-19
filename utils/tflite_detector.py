# utils/tflite_detector.py
import numpy as np
from PIL import Image
import cv2
import os
import platform

# Try to import TFLite runtime
try:
    # First try the full tensorflow (if installed)
    import tensorflow as tf

    interpreter_func = tf.lite.Interpreter
    print("✅ Using TensorFlow Lite")
except:
    try:
        # Then try tflite-runtime
        import tflite_runtime.interpreter as tflite

        interpreter_func = tflite.Interpreter
        print("✅ Using tflite-runtime")
    except:
        print("❌ No TFLite library found")
        interpreter_func = None


class TFLiteDetector:
    def __init__(self, model_path='models/model_unquant.tflite',
                 labels_path='models/labels.txt'):
        """Initialize the TensorFlow Lite model"""

        print("=" * 50)
        print("🔍 Initializing TFLite Detector")
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
            if interpreter_func:
                self.interpreter = interpreter_func(model_path=str(self.model_path))
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
            else:
                print("❌ No TFLite interpreter available")
                self.interpreter = None
                return

            # Load labels
            with open(self.labels_path, 'r') as f:
                raw_labels = [line.strip() for line in f.readlines()]

            # Clean labels (remove numbers like "0 cup" → "cup")
            self.class_names = []
            for label in raw_labels:
                if ' ' in label:
                    parts = label.split(' ', 1)
                    if parts[0].isdigit():
                        self.class_names.append(parts[1].lower())
                    else:
                        self.class_names.append(label.lower())
                else:
                    self.class_names.append(label.lower())

            print(f"✅ Labels loaded: {self.class_names}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.interpreter = None

    def preprocess_image(self, image):
        """Preprocess image for the model"""
        try:
            # Handle different input types
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)

            # Resize to expected dimensions
            image = image.resize((self.input_width, self.input_height))

            # Convert to numpy array
            image_array = np.asarray(image)

            # Normalize to [0,1] (common for TFLite models)
            image_array = image_array.astype(np.float32) / 255.0

            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)

            return image_array

        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return None

    def predict(self, image):
        """Make prediction on image"""

        if self.interpreter is None:
            return "Model not loaded", 0.0

        try:
            # Preprocess image
            input_data = self.preprocess_image(image)
            if input_data is None:
                return "Error", 0.0

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
            if index < len(self.class_names):
                class_name = self.class_names[index]
            else:
                class_name = f"Class {index}"

            return class_name, confidence

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Error", 0.0

    def predict_top_k(self, image, k=3):
        """Get top k predictions"""
        if self.interpreter is None:
            return []

        try:
            input_data = self.preprocess_image(image)
            if input_data is None:
                return []

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()

            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions = output_data[0]

            # Get top k indices
            top_indices = np.argsort(predictions)[-k:][::-1]

            results = []
            for idx in top_indices:
                if idx < len(self.class_names):
                    label = self.class_names[idx]
                else:
                    label = f"Class {idx}"
                confidence = float(predictions[idx])
                results.append((label, confidence))

            return results

        except Exception as e:
            print(f"❌ Error: {e}")
            return []