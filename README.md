AI Push-Up Trainer
==================

> Your personal AI-powered fitness coach that counts your reps and corrects your form in real-time, right from your webcam.

This project uses Python with OpenCV and MediaPipe to create a smart push-up counter. It goes beyond simple counting by implementing an intelligent state machine that waits for the user to be in the correct plank position before starting, and provides real-time feedback on elbow flare to ensure proper form.

_(A live demonstration of the application counting reps and providing real-time form feedback.)_

✨ Key Features
--------------

*   🧠 **Intelligent State Machine:** The trainer waits for you to be in a proper, stable plank position before it starts counting. No more accidental counts from random movements!
    
*   🏋️‍♂️ **Real-Time Repetition Counting:** Accurately counts push-ups by tracking the angle of both elbows.
    
*   💪 **Form Correction:** Actively monitors for flared elbows (a common mistake) and provides instant visual feedback to help you maintain good form.
    
*   📊 **Intuitive UI:** A clean, on-screen display shows your rep count, current stage (GET\_READY, READY, DOWN), and color-coded form feedback.
    
*   📹 **Session Recording:** Automatically records your workout session and saves it as an .mp4 file for you to review later.
    
*   🐛 **Live Debug View:** An optional display shows the exact back and elbow angles the AI is seeing, helping you understand its logic.
    

🧠 How It Works
---------------

The application processes your webcam feed frame-by-frame to analyze your posture and count repetitions.

1.  **Video Capture:** OpenCV is used to capture the live video feed from your webcam.
    
2.  **Pose Estimation:** Google's MediaPipe library is used to detect and track 33 key body landmarks (joints) in real-time.
    
3.  **Angle Calculation:** The program calculates the angles at the elbows (shoulder-elbow-wrist) and hips (shoulder-hip-ankle) to understand your body's position.
    
4.  **State Machine Logic:** The core of the trainer is a state machine that progresses through three states:
    
    *   **GET\_READY**: The initial state. The program waits for you to hold a stable plank position (straight back, extended arms) before proceeding. This prevents false starts.
        
    *   **READY**: You are in the correct starting position. The trainer is now actively waiting for you to lower your body.
        
    *   **DOWN**: You have successfully completed the downward motion. The trainer now waits for you to push back up. A rep is counted upon returning to the READY state.
        
5.  **Form Analysis:** In every frame, the program also calculates the "flare angle" (hip-shoulder-elbow). If this angle is too wide, it triggers the "TUCK ELBOWS" feedback, helping to prevent shoulder injuries.
    

🚀 Getting Started
------------------

Follow these instructions to get the project running on your local machine.

### Prerequisites

*   Python 3.8+
    
*   A webcam connected to your computer.
    

### 🛠️ Installation & Setup

1. **Clone the repository:**

```bash
git clone \[https://github.com/your-username/AI-Pushup-Trainer.git\](https://github.com/your-username/AI-Pushup-Trainer.git)cd AI-Pushup-Trainer
```
    
2.  **Create and activate a virtual environment:**
    
    ```bash
    *   python -m venv venv
        venv\\Scripts\\activate
     ```

    
        
3. **Install the required libraries:**

```bash
 pip install opencv-python mediapipe numpy
   ``` 

▶️ How to Run
-------------

With your virtual environment activated and dependencies installed, run the following command in your terminal:

```bash
  python pushup_trainer_smart.py   
  ```

*   A window will pop up showing your webcam feed.
    
*   Position yourself so your full body is visible.
    
*   Follow the on-screen prompts to get into position.
    
*   Press the **'q'** key to quit the application.
    
*   A video file named pushup\_session\_smart.mp4 will be saved in the project folder.
    

🔧 Technologies Used
--------------------

*   **Python:** The core programming language.
    
*   **OpenCV:** For video capture, image processing, and rendering the UI.
    
*   **MediaPipe:** For high-fidelity body pose tracking.
    
*   **NumPy:** For numerical operations and angle calculations.
    

💡 Future Improvements
----------------------

*   \[ \] **Audio Feedback:** Add voice cues for rep counts and form corrections.
    
*   \[ \] **Back Straightness Check:** Implement feedback for a sagging or arched back.
    
*   \[ \] **Workout Log:** Save workout stats (reps, date, time) to a file.
    
*   \[ \] **Support for More Exercises:** Extend the logic to count other exercises like squats or bicep curls.