# teachable_machine_game.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import time
import random
import os

# Disable scientific notation
np.set_printoptions(suppress=True)

# Page config
st.set_page_config(page_title="Desk Safari", layout="wide")

st.title("🦓 Desk Safari")


# ===== LOAD TFLITE MODEL =====
@st.cache_resource
def load_model():
    """Load TFLite model"""
    try:
        models_dir = 'models'

        # Your actual model file name
        model_filename = 'model_unquant.tflite'
        model_path = os.path.join(models_dir, model_filename)

        if not os.path.exists(model_path):
            st.error(f"❌ Model not found: {model_path}")
            st.info(f"Files in models folder: {os.listdir(models_dir)}")
            return None, None, None, None

        st.success(f"✅ Found model: {model_filename}")

        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        st.info(f"📊 Input shape: {input_details[0]['shape']}")
        st.info(f"📊 Output shape: {output_details[0]['shape']}")

        # Load labels
        labels_path = os.path.join(models_dir, 'labels.txt')
        if not os.path.exists(labels_path):
            st.error(f"❌ labels.txt not found!")
            return None, None, None, None

        with open(labels_path, 'r') as f:
            class_names = f.readlines()

        # Clean labels (remove numbers)
        clean_names = []
        for name in class_names:
            name = name.strip()
            if ' ' in name:
                # Take the part after the number (e.g., "0 cup" -> "cup")
                parts = name.split(' ', 1)
                if parts[0].isdigit():
                    name = parts[1]
            clean_names.append(name.lower())

        st.success(f"✅ Model loaded! Classes: {clean_names}")
        return interpreter, input_details, output_details, clean_names

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None


# ===== SIMPLE GAME CLASS =====
class Game:
    def __init__(self):
        self.items = ["cup", "book", "phone", "pen", "keys", "remote"]
        self.emoji = {
            "cup": "☕", "book": "📚", "phone": "📱",
            "pen": "✒️", "keys": "🔑", "remote": "📺"
        }
        self.score = 0
        self.time_left = 60
        self.current_item = None
        self.items_found = []
        self.game_active = False
        self.start_time = None

    def start_game(self):
        self.game_active = True
        self.score = 0
        self.time_left = 60
        self.items_found = []
        self.start_time = time.time()
        self._pick_new_item()

    def _pick_new_item(self):
        available = [i for i in self.items if i not in self.items_found]
        if available:
            self.current_item = random.choice(available)
        else:
            self.current_item = None
            self.game_active = False

    def update_timer(self):
        if self.game_active and self.start_time:
            elapsed = time.time() - self.start_time
            self.time_left = max(0, 60 - int(elapsed))
            if self.time_left <= 0:
                self.game_active = False

    def check_detection(self, detected_item, confidence):
        if not self.game_active:
            return None

        if confidence < 0.3:  # Lower threshold for testing
            return None

        if detected_item.lower() == self.current_item.lower():
            self.score += 1
            self.items_found.append(self.current_item)
            self._pick_new_item()

            if len(self.items_found) == len(self.items):
                self.game_active = False
                return "complete"
            return "correct"
        return None

    def get_state(self):
        return {
            'active': self.game_active,
            'score': self.score,
            'time_left': self.time_left,
            'current_item': self.current_item,
            'current_emoji': self.emoji.get(self.current_item, "❓"),
            'items_found': self.items_found,
            'progress': len(self.items_found) / len(self.items) if self.items else 0
        }


# ===== INITIALIZE =====
if 'game' not in st.session_state:
    st.session_state.game = Game()

if 'model_data' not in st.session_state:
    with st.spinner("Loading AI model..."):
        interpreter, input_details, output_details, class_names = load_model()
        st.session_state.model_data = {
            'interpreter': interpreter,
            'input_details': input_details,
            'output_details': output_details,
            'class_names': class_names
        }

# ===== MAIN UI =====
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📸 Camera")

    # Check if model loaded
    if st.session_state.model_data['interpreter'] is None:
        st.error("❌ Model not loaded. Please check:")
        if os.path.exists('models'):
            files = os.listdir('models')
            st.write(f"Files in models folder: {files}")
            st.write("Expected: model_unquant.tflite and labels.txt")
        else:
            st.write("❌ Create a 'models' folder and add your model files")
        st.stop()

    # Camera input
    img_file = st.camera_input("Take a photo")

    if img_file:
        # Load image
        image = Image.open(img_file)
        st.image(image, caption="Captured", width=300)

        # ===== TFLITE PREDICTION =====
        with st.spinner("Analyzing..."):
            interpreter = st.session_state.model_data['interpreter']
            input_details = st.session_state.model_data['input_details']
            output_details = st.session_state.model_data['output_details']
            class_names = st.session_state.model_data['class_names']

            # Prepare image
            image = image.convert('RGB')
            image = image.resize((224, 224))
            image_array = np.asarray(image)

            # TRY BOTH NORMALIZATION METHODS
            st.write("**Trying both normalization methods:**")

            # Method 1: 0-1 normalization
            input_data_1 = image_array.astype(np.float32) / 255.0
            input_data_1 = np.expand_dims(input_data_1, axis=0)

            # Method 2: -1 to 1 normalization (Teachable Machine default)
            input_data_2 = (image_array.astype(np.float32) / 127.5) - 1
            input_data_2 = np.expand_dims(input_data_2, axis=0)

            # Try first method
            interpreter.set_tensor(input_details[0]['index'], input_data_1)
            interpreter.invoke()
            output_1 = interpreter.get_tensor(output_details[0]['index'])[0]

            # Try second method
            interpreter.set_tensor(input_details[0]['index'], input_data_2)
            interpreter.invoke()
            output_2 = interpreter.get_tensor(output_details[0]['index'])[0]

            # Use the one with higher max confidence
            if np.max(output_1) > np.max(output_2):
                predictions = output_1
                method_used = "Method 1 (0-1 normalization)"
            else:
                predictions = output_2
                method_used = "Method 2 (-1 to 1 normalization)"

            # Get best prediction
            index = np.argmax(predictions)
            confidence = float(predictions[index])

            if index < len(class_names):
                predicted_label = class_names[index]
            else:
                predicted_label = f"class_{index}"

        # Show prediction
        st.markdown("---")
        st.subheader("📊 Detection Result")

        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Detected", predicted_label)
        with col_res2:
            # Color code confidence
            if confidence > 0.7:
                st.metric("Confidence", f"{confidence:.1%}", delta="High")
            elif confidence > 0.3:
                st.metric("Confidence", f"{confidence:.1%}", delta="Medium")
            else:
                st.metric("Confidence", f"{confidence:.1%}", delta="Low")
        with col_res3:
            # Compare with target
            game = st.session_state.game
            if game.game_active and game.current_item:
                if predicted_label == game.current_item:
                    st.success("✅ MATCH!")
                else:
                    st.error(f"❌ Need: {game.current_item}")

        # Show which method worked better
        st.info(f"✅ Using {method_used}")

        # Process game detection
        result = st.session_state.game.check_detection(predicted_label, confidence)

        # Show result message
        if result == "correct":
            st.success("🎉 +1 POINT!")
        elif result == "complete":
            st.balloons()
            st.success("🏆 YOU WIN!")

        # Show raw predictions
        with st.expander("🔬 Raw Predictions"):
            col_raw1, col_raw2 = st.columns(2)
            with col_raw1:
                st.write("**Method 1 (0-1 norm):**")
                for i, (name, prob) in enumerate(zip(class_names, output_1)):
                    st.write(f"{i}: {name} = {prob:.4f} ({prob:.2%})")
            with col_raw2:
                st.write("**Method 2 (-1 to 1 norm):**")
                for i, (name, prob) in enumerate(zip(class_names, output_2)):
                    st.write(f"{i}: {name} = {prob:.4f} ({prob:.2%})")

with col2:
    st.subheader("🎮 Game")

    game = st.session_state.game
    state = game.get_state()

    # Start button
    if not state['active']:
        if st.button("🚀 START GAME", type="primary", use_container_width=True):
            game.start_game()
            st.rerun()

    # Game display
    if state['active']:
        # Update timer
        game.update_timer()
        state = game.get_state()

        # Score and timer
        st.metric("Score", state['score'])
        st.metric("Time Left", f"{state['time_left']}s")

        # Current target
        if state['current_item']:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                color: white;
                margin: 20px 0;
            ">
                <h3>FIND THIS:</h3>
                <h1 style="font-size: 80px;">{state['current_emoji']}</h1>
                <h2>{state['current_item'].upper()}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Progress
        st.progress(state['progress'])

        # Items found
        if state['items_found']:
            st.write("✅ Found:")
            for item in state['items_found']:
                st.write(f"  {game.emoji.get(item, '')} {item}")

        # Time's up
        if state['time_left'] <= 0:
            st.error("⏰ TIME'S UP!")
            game.game_active = False
            st.rerun()
    else:
        st.info("Click START GAME to begin!")
        st.write("**Items to find:**")
        for item in game.items:
            st.write(f"  {game.emoji.get(item, '')} {item}")

# Auto-refresh
if st.session_state.game.game_active:
    time.sleep(1)
    st.rerun()

# Show model info in sidebar
with st.sidebar:
    st.header("ℹ️ Info")
    if st.session_state.model_data['class_names']:
        st.write("**Model can detect:**")
        for name in st.session_state.model_data['class_names']:
            st.write(f"• {name}")

    st.write("**Model file:**")
    st.write(f"📁 model_unquant.tflite")

    st.write("**Game items:**")
    for item in st.session_state.game.items:
        st.write(f"• {item}")