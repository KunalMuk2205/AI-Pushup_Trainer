import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def run_trainer(debug=False):
    DEBUG = debug
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = 'pushup_session_smart.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    counter = 0
    state = 'get_ready'
    feedback = ''
    visibility_threshold = 0.5

    READY_FRAMES = 4
    DOWN_FRAMES = 2
    ready_counter = 0
    down_counter = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        last_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            avg_back_angle = 0
            avg_elbow_angle = 0

            try:
                if not results.pose_landmarks:
                    raise ValueError("no landmarks")

                landmarks = results.pose_landmarks.landmark

                l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                l_el = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                l_wr = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                l_hp = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                l_ank = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

                r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                r_el = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                r_wr = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                r_hp = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                r_ank = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                is_body_visible = all(
                    lm.visibility > visibility_threshold
                    for lm in [l_sh, l_el, l_wr, l_hp, r_sh, r_el, r_wr, r_hp]
                )

                l_sh_xy = [l_sh.x, l_sh.y]
                l_el_xy = [l_el.x, l_el.y]
                l_wr_xy = [l_wr.x, l_wr.y]
                l_hp_xy = [l_hp.x, l_hp.y]
                l_ank_xy = [l_ank.x, l_ank.y]

                r_sh_xy = [r_sh.x, r_sh.y]
                r_el_xy = [r_el.x, r_el.y]
                r_wr_xy = [r_wr.x, r_wr.y]
                r_hp_xy = [r_hp.x, r_hp.y]
                r_ank_xy = [r_ank.x, r_ank.y]

                # --- plank detection: require torso roughly horizontal ---
                mid_sh_x = (l_sh.x + r_sh.x) / 2.0
                mid_sh_y = (l_sh.y + r_sh.y) / 2.0
                mid_hp_x = (l_hp.x + r_hp.x) / 2.0
                mid_hp_y = (l_hp.y + r_hp.y) / 2.0

                torso_angle_rad = np.arctan2(abs(mid_sh_y - mid_hp_y), abs(mid_sh_x - mid_hp_x) + 1e-6)
                torso_angle_deg = abs(torso_angle_rad * 180.0 / np.pi)
                torso_y_diff = abs(mid_sh_y - mid_hp_y)

                PLANK_ANGLE_THRESH = 30.0
                PLANK_Y_DIFF_THRESH = 0.15

                is_in_plank = (torso_angle_deg < PLANK_ANGLE_THRESH) and (torso_y_diff < PLANK_Y_DIFF_THRESH)
                # --- end plank detection ---

                left_elbow_angle = calculate_angle(l_sh_xy, l_el_xy, l_wr_xy)
                right_elbow_angle = calculate_angle(r_sh_xy, r_el_xy, r_wr_xy)
                avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2.0

                left_back_angle = calculate_angle(l_sh_xy, l_hp_xy, l_ank_xy)
                right_back_angle = calculate_angle(r_sh_xy, r_hp_xy, r_ank_xy)
                avg_back_angle = (left_back_angle + right_back_angle) / 2.0

                form_feedback = "GOOD FORM"
                left_sh_el_hip = calculate_angle(l_hp_xy, l_sh_xy, l_el_xy)
                right_sh_el_hip = calculate_angle(r_hp_xy, r_sh_xy, r_el_xy)
                if left_sh_el_hip > 65 or right_sh_el_hip > 65:
                    form_feedback = "TUCK ELBOWS"

                if not is_body_visible:
                    state = 'get_ready'
                    feedback = "NO BODY DETECTED"
                    ready_counter = 0
                    down_counter = 0

                elif state == 'get_ready':
                    # require a stable plank for a few frames
                    if is_in_plank and avg_back_angle > 145 and avg_elbow_angle > 155:
                        ready_counter += 1
                        if ready_counter >= READY_FRAMES:
                            state = 'ready'
                            feedback = form_feedback
                            ready_counter = 0
                    else:
                        ready_counter = 0
                        feedback = "GET INTO PLANK POSITION" if is_body_visible else "NO BODY DETECTED"

                elif state == 'ready':
                    feedback = form_feedback
                    # detect start lowering (only if still in plank)
                    if is_in_plank and avg_elbow_angle < 95:
                        down_counter += 1
                        if down_counter >= DOWN_FRAMES:
                            state = 'down'
                            down_counter = 0
                            feedback = form_feedback
                    else:
                        down_counter = 0

                elif state == 'down':
                    feedback = form_feedback
                    # count rep only if still in plank orientation
                    if is_in_plank and avg_elbow_angle > 155:
                        counter += 1
                        state = 'ready'
                        feedback = "REP COUNTED!"
                        ready_counter = 0

                if DEBUG:
                    print(f"DEBUG: state={state} back={avg_back_angle:.1f} elbow={avg_elbow_angle:.1f} visible={is_body_visible} rep={counter}")
                    print(f"DEBUG: torso_angle={torso_angle_deg:.1f} torso_y_diff={torso_y_diff:.3f} in_plank={is_in_plank}")

            except Exception:
                state = 'get_ready'
                feedback = "NO BODY DETECTED"

            if "TUCK ELBOWS" in feedback:
                feedback_box_color = (0, 0, 255)
            elif "GOOD" in feedback:
                feedback_box_color = (0, 150, 0)
            elif "COUNTED" in feedback:
                feedback_box_color = (200, 100, 0)
            elif "DETECTED" in feedback or "NO BODY" in feedback:
                feedback_box_color = (0, 165, 255)
            else:
                feedback_box_color = (128, 0, 0)

            cv2.rectangle(image, (0, 0), (300, 90), (50, 50, 50), -1)
            cv2.putText(image, 'REPS', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STATUS', (160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, state.upper(), (160, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(image, (300, 0), (frame_width, 90), feedback_box_color, -1)
            (text_width, _), _ = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = 300 + max(10, (frame_width - 300 - text_width) // 2)
            cv2.putText(image, feedback, (text_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, f"BACK: {int(avg_back_angle)}", (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"ELBOW: {int(avg_elbow_angle)}", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                )

            out.write(image)
            cv2.imshow('Smart Push-Up Trainer', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def _print_push_steps(username: str, repo_name: str):
    remote_url = f"https://github.com/{username}/{repo_name}.git"
    print("\n--- Manual push steps ---")
    print("1) On GitHub: create a new repository named:", repo_name)
    print("   URL: https://github.com/new")
    print("2) In your project folder run these commands:")
    print(f"   git remote remove origin            # if an existing origin blocks you (optional)")
    print(f"   git remote add origin {remote_url}")
    print("   git branch -M main")
    print("   git push -u origin main")
    print("If authentication fails, create a Personal Access Token (PAT) and use it when prompted, or set up SSH keys.")
    print("---------------------------\n")

def _attempt_auto_push(username: str, repo_name: str):
    import shutil, subprocess, sys
    remote_url = f"https://github.com/{username}/{repo_name}.git"

    if shutil.which("gh"):
        print("Found GitHub CLI (gh). Attempting to create & push using gh...")
        subprocess.run(["gh", "repo", "create", f"{username}/{repo_name}", "--public", "--source", ".", "--remote", "origin", "--push"])
        return

    if shutil.which("git") is None:
        print("ERROR: git not found in PATH. Install git and retry.")
        return

    proc = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
    if proc.returncode == 0:
        print("Existing origin detected:", proc.stdout.strip())
        print("Removing existing origin...")
        subprocess.run(["git", "remote", "remove", "origin"])

    print("Adding origin:", remote_url)
    subprocess.run(["git", "remote", "add", "origin", remote_url])
    subprocess.run(["git", "branch", "-M", "main"])
    print("Pushing to origin main (may prompt for credentials)...")
    push_proc = subprocess.run(["git", "push", "-u", "origin", "main"])
    if push_proc.returncode == 0:
        print("Push completed successfully.")
    else:
        print("Push finished with non-zero exit. Check output above. If authentication failed, use a Personal Access Token or configure SSH keys.")
    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="pushup_trainer", description="Run trainer or help push repo to GitHub")
    parser.add_argument('--push', action='store_true', help='Show push instructions for GitHub (or try to push)')
    parser.add_argument('--repo', default='AI-Pushup-Trainer', help='Repository name on GitHub')
    parser.add_argument('--username', default='KunMuk2205', help='GitHub username')
    parser.add_argument('--auto', action='store_true', help='Attempt to run git/gh commands automatically')
    parser.add_argument('--debug', action='store_true', help='Run trainer in debug mode')
    args = parser.parse_args()

    if args.push:
        _print_push_steps(args.username, args.repo)
        if args.auto:
            _attempt_auto_push(args.username, args.repo)
    else:
        run_trainer(debug=args.debug)
