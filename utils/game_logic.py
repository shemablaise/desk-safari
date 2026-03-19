# utils/game_logic.py
import random
import time


class GameManager:
    def __init__(self):
        # IMPORTANT: This order MUST match your labels.txt EXACTLY!
        # labels.txt order: 0=cup, 1=book, 2=phone, 3=pen, 4=keys, 5=remote
        self.items = ["cup", "book", "phone", "pen", "keys", "remote"]

        # Emoji mapping for visual display
        self.items_emoji = {
            "cup": "☕",
            "book": "📚",
            "phone": "📱",
            "pen": "✒️",
            "keys": "🔑",
            "remote": "📺"
        }

        # Game settings
        self.game_duration = 60  # seconds
        self.confidence_threshold = 0.5  # Minimum confidence to accept a detection

        # State variables
        self.score = 0
        self.time_left = self.game_duration
        self.current_item = None
        self.items_found = []
        self.game_active = False
        self.start_time = None

        print("✅ GameManager initialized with items:", self.items)

    def start_game(self):
        """Start a new game"""
        print("\n🎮 ===== STARTING NEW GAME =====")
        self.game_active = True
        self.score = 0
        self.time_left = self.game_duration
        self.items_found = []
        self.start_time = time.time()

        # Pick first item
        success = self.pick_new_item()
        print(f"📋 First item to find: {self.current_item}")
        return success

    def pick_new_item(self):
        """Pick a new random item that hasn't been found yet"""
        # Get items not yet found
        available = [item for item in self.items if item not in self.items_found]

        if not available:
            print("🎉 No items left! Game complete!")
            self.game_active = False
            self.current_item = None
            return False

        # Pick random available item
        self.current_item = random.choice(available)
        print(f"🎯 New target: {self.current_item}")
        return True

    def update_timer(self):
        """Update remaining time - call this every second"""
        if not self.game_active or not self.start_time:
            return True

        elapsed = time.time() - self.start_time
        self.time_left = max(0, self.game_duration - int(elapsed))

        # Check if time ran out
        if self.time_left <= 0:
            print("⏰ Time's up!")
            self.game_active = False
            return False

        return True

    def process_detection(self, detected_item, confidence):
        """
        Process a detection from the model
        Returns: "correct" if point awarded, "complete" if game finished, None otherwise
        """
        # Check if game is active
        if not self.game_active:
            return None

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return None

        # Check if detected item matches current target (case-insensitive)
        if detected_item.lower() == self.current_item.lower():
            print(f"✅ CORRECT! {detected_item} matches {self.current_item}")

            # AWARD POINT!
            self.score += 1
            self.items_found.append(self.current_item)

            # Pick new item
            success = self.pick_new_item()

            if not success:
                print("🏆 GAME COMPLETE! All items found!")
                self.game_active = False
                return "complete"

            return "correct"

        return None

    def get_game_state(self):
        """Return current game state for display"""
        return {
            'active': self.game_active,
            'score': self.score,
            'time_left': self.time_left,
            'current_item': self.current_item,
            'current_item_emoji': self.items_emoji.get(self.current_item, ""),
            'items_found': self.items_found.copy(),
            'items_remaining': [item for item in self.items if item not in self.items_found],
            'total_items': len(self.items),
            'progress': len(self.items_found) / len(self.items) if self.items else 0
        }

    def reset_game(self):
        """Reset the game completely"""
        print("🔄 Resetting game")
        self.game_active = False
        self.score = 0
        self.time_left = self.game_duration
        self.current_item = None
        self.items_found = []
        self.start_time = None