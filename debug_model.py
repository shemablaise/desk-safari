# test_tflite.py
from utils.model_tflite import TFLiteObjectDetector
from PIL import Image
import numpy as np

print("=" * 60)
print("🔍 TESTING TFLITE MODEL")
print("=" * 60)

# Initialize detector
detector = TFLiteObjectDetector()

if detector.interpreter is not None:
    print("\n✅ Model loaded successfully!")

    # Create a test image
    test_image = Image.new('RGB', (224, 224), color='red')

    # Make prediction
    label, confidence = detector.predict(test_image)
    print(f"\n🎯 Test prediction: {label} ({confidence:.2%})")

else:
    print("\n❌ Model failed to load")

print("\n" + "=" * 60)