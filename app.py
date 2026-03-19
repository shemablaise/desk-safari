# app.py
import streamlit as st
from PIL import Image
import time
import os
import base64
from utils.tflite_detector import TFLiteDetector
from utils.game_logic import GameManager

# Page configuration
st.set_page_config(
    page_title="Desk Safari",
    page_icon="🦓",
    layout="wide"
)


# ===== SOUND SYSTEM USING ASSETS FOLDER =====
def play_sound(sound_file):
    """Play a sound from the assets folder"""
    try:
        sound_path = os.path.join("assets", sound_file)
        if os.path.exists(sound_path):
            with open(sound_path, "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()
                sound_html = f'''
                    <audio autoplay style="display:none">
                        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                '''
                st.markdown(sound_html, unsafe_allow_html=True)
        else:
            print(f"Sound file not found: {sound_path}")
    except Exception as e:
        print(f"Could not play sound {sound_file}: {e}")


# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 20px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        border: none;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .target-box {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .score-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        border-left: 5px solid #667eea;
    }
    .game-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .floating-emoji {
        position: fixed;
        font-size: 24px;
        opacity: 0.2;
        pointer-events: none;
        z-index: -1;
        animation: float 15s infinite linear;
    }
    @keyframes float {
        0% { transform: translateY(100vh) rotate(0deg); }
        100% { transform: translateY(-100px) rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)


# ===== FLOATING EMOJIS BACKGROUND =====
def add_floating_emojis():
    emojis = ["🦓", "☕", "📚", "📱", "✒️", "🔑", "📺", "🎮", "⭐", "✨"]
    floating_html = ""
    for i in range(15):
        emoji = emojis[i % len(emojis)]
        left = i * 7
        delay = i * 2
        floating_html += f"""
        <div class="floating-emoji" style="left: {left}%; animation-delay: -{delay}s;">{emoji}</div>
        """
    st.markdown(floating_html, unsafe_allow_html=True)


# Add floating emojis
add_floating_emojis()

# ===== HEADER =====
st.markdown("""
<div class="game-header">
    <h1>🦓 Desk Safari</h1>
    <p style="font-size: 18px;">Find the objects before time runs out!</p>
</div>
""", unsafe_allow_html=True)

# ===== SOUND ENABLE BUTTON (for browser autoplay) =====
if 'sound_enabled' not in st.session_state:
    st.session_state.sound_enabled = False

if not st.session_state.sound_enabled:
    col_sound1, col_sound2, col_sound3 = st.columns([1, 2, 1])
    with col_sound2:
        if st.button("🔊 Enable Sounds", use_container_width=True):
            st.session_state.sound_enabled = True
            # Play a test sound when enabled
            play_sound("start.mp3")
            st.rerun()
    st.info("👆 Click 'Enable Sounds' to hear sound effects")
    st.markdown("---")

# ===== INITIALIZE SESSION STATE =====
if 'game' not in st.session_state:
    st.session_state.game = GameManager()

if 'detector' not in st.session_state:
    with st.spinner("🎯 Loading AI model... Please wait..."):
        try:
            st.session_state.detector = TFLiteDetector()
            if st.session_state.detector.interpreter is not None:
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Model failed to load. Please check your model files.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.detector = None

# ===== MAIN LAYOUT =====
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📸 Camera")

    # Check if model loaded
    if st.session_state.detector is None or st.session_state.detector.interpreter is None:
        st.warning("⚠️ Model not loaded. Please check the error message above.")

        # Show debug info
        with st.expander("🔧 Debug Info"):
            st.write(f"Current directory: {os.getcwd()}")
            if os.path.exists('models'):
                st.write(f"Models folder contents: {os.listdir('models')}")
            else:
                st.write("Models folder not found!")
        st.stop()

    # Camera input
    img_file = st.camera_input("Take a photo", key="camera")

    if img_file:
        # Load image
        image = Image.open(img_file)
        st.image(image, caption="Captured", width=300)

        # Predict
        with st.spinner("🔍 Analyzing..."):
            label, confidence = st.session_state.detector.predict(image)

        # Show prediction
        st.markdown("---")
        st.subheader("📊 Detection Result")

        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Detected", label)
        with col_res2:
            # Color code confidence
            if confidence > 0.7:
                st.metric("Confidence", f"{confidence:.1%}", delta="High")
            elif confidence > 0.4:
                st.metric("Confidence", f"{confidence:.1%}", delta="Medium")
            else:
                st.metric("Confidence", f"{confidence:.1%}", delta="Low")
        with col_res3:
            # Compare with target
            game = st.session_state.game
            if game.game_active and game.current_item:
                if label == game.current_item:
                    st.success("✅ MATCH!")
                else:
                    st.error(f"❌ Need: {game.current_item}")

        # Process game detection
        result = st.session_state.game.process_detection(label, confidence)

        # Handle game results with sounds
        if result == "correct":
            if st.session_state.sound_enabled:
                play_sound("correct.mp3")
            st.success("🎉 +1 POINT!")
        elif result == "complete":
            if st.session_state.sound_enabled:
                play_sound("win.mp3")
            st.balloons()
            st.success("🏆 GAME COMPLETE! You found all items!")

        # Show top predictions
        with st.expander("🔬 All Predictions"):
            top_k = st.session_state.detector.predict_top_k(image, k=3)
            for i, (pred_label, pred_conf) in enumerate(top_k, 1):
                st.write(f"{i}. {pred_label}: {pred_conf:.2%}")

with col2:
    st.subheader("🎮 Game")

    game = st.session_state.game
    state = game.get_game_state()

    # Start button
    if not state['active']:
        if st.button("🚀 START GAME", type="primary", use_container_width=True):
            if st.session_state.sound_enabled:
                play_sound("start.mp3")
            game.start_game()
            st.rerun()

    # Game display
    if state['active']:
        # Update timer
        game.update_timer()
        state = game.get_game_state()

        # Score and timer
        st.markdown(f"""
        <div class="score-box">
            <h2 style="margin:0">Score: {state['score']}</h2>
            <h1 style="margin:5px 0; color: {'#FF4B4B' if state['time_left'] < 10 else '#333'};">
                ⏱️ {state['time_left']}s
            </h1>
        </div>
        """, unsafe_allow_html=True)

        # Timer warning sound
        if state['time_left'] < 10 and state['time_left'] > 0 and st.session_state.sound_enabled:
            play_sound("timer.mp3")

        # Current target
        if state['current_item']:
            st.markdown(f"""
            <div class="target-box">
                <h3 style="margin:0">FIND THIS:</h3>
                <h1 style="font-size: 80px;">{state['current_item_emoji']}</h1>
                <h2 style="margin:0">{state['current_item'].upper()}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Progress
        st.subheader("📊 Progress")
        st.progress(state['progress'])

        # Items found and remaining
        col_found, col_remaining = st.columns(2)

        with col_found:
            st.markdown("**✅ Found:**")
            if state['items_found']:
                for item in state['items_found']:
                    emoji = game.items_emoji.get(item, "")
                    st.success(f"{emoji} {item}")
            else:
                st.write("None yet")

        with col_remaining:
            st.markdown("**⏳ Remaining:**")
            if state['items_remaining']:
                for item in state['items_remaining']:
                    emoji = game.items_emoji.get(item, "")
                    st.write(f"{emoji} {item}")
            else:
                st.write("✨ All found!")

        # Game over
        if state['time_left'] <= 0:
            if st.session_state.sound_enabled:
                play_sound("wrong.mp3")
            st.error("⏰ **TIME'S UP!**")
            if st.button("🔄 Play Again", use_container_width=True):
                game.reset_game()
                st.rerun()

    else:
        # Instructions
        st.info("👆 Click START GAME to begin!")

        with st.expander("📖 How to Play"):
            st.write("""
            1. Click **START GAME**
            2. Look at the item displayed
            3. Show that item to your camera
            4. Take a photo
            5. Get a point if detected correctly!
            6. Find all 6 items before time runs out!
            """)

        # Show all items
        st.markdown("**🎯 Items to Find:**")
        for item, emoji in game.items_emoji.items():
            st.write(f"{emoji} {item}")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("ℹ️ Game Info")

    if st.session_state.detector and st.session_state.detector.class_names:
        st.success(f"✅ Model: Loaded")
        st.write(f"📊 Detects: {len(st.session_state.detector.class_names)} objects")

    st.write(f"🎮 Game: {'Active' if st.session_state.game.game_active else 'Inactive'}")

    # Sound status
    sound_status = "🔊 On" if st.session_state.sound_enabled else "🔇 Off"
    st.write(f"🎵 Sound: {sound_status}")

    if st.button("🔄 Reset Game"):
        game.reset_game()
        st.rerun()

    st.markdown("---")
    st.header("🎵 Sound Files")

    # Check which sound files exist
    sound_files = {
        "correct.mp3": "✅ Correct",
        "wrong.mp3": "❌ Wrong",
        "win.mp3": "🏆 Win",
        "start.mp3": "🚀 Start",
        "timer.mp3": "⏰ Timer"
    }

    for filename, description in sound_files.items():
        filepath = os.path.join("assets", filename)
        if os.path.exists(filepath):
            st.write(f"✅ {description}")
        else:
            st.write(f"❌ {description} (missing)")

# ===== AUTO REFRESH FOR TIMER =====
if st.session_state.game.game_active:
    time.sleep(1)
    st.rerun()

# ===== FOOTER =====
st.markdown("---")
st.markdown(
    "<center>Built with Streamlit and TensorFlow Lite | Desk Safari Game</center>",
    unsafe_allow_html=True
)