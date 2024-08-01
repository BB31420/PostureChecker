from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Function to find the distance between two points
def find_distance(a_x, a_y, b_x, b_y):
    return np.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)

# Function to check posture
def check_posture(landmarks, w, h):
    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x * w, landmarks[mp_pose.PoseLandmark.NOSE.value].y * h]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
    left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w,
                landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h]

    # Calculate shoulder angle
    shoulder_angle = calculate_angle(left_shoulder, right_shoulder, [right_shoulder[0], right_shoulder[1] - 0.1 * h])  # Using a slight downward offset

    # Calculate head angle
    head_angle = calculate_angle(left_shoulder, nose, right_shoulder)

    # Calculate neck inclination
    neck_inclination = calculate_angle(left_shoulder, left_ear, [left_shoulder[0], left_shoulder[1] - 0.1 * h])

    posture = "Good posture"

    if shoulder_angle < 85 or shoulder_angle > 88:
        posture = "Incorrect shoulder posture"
    elif head_angle < 84 or head_angle > 90:
        posture = "Incorrect head posture"

    return posture, neck_inclination

# Video streaming generator function
def gen_frames():
    cap = cv2.VideoCapture(0)
    good_frames = 0
    bad_frames = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (0, 255, 0)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    light_green = (144, 238, 144)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Null.Frames")
            break

        # Get fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Get height and width of the frame
        h, w = frame.shape[:2]

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and detect the pose
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Check posture
            posture, neck_inclination = check_posture(landmarks, w, h)

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display posture message
            cv2.putText(image, posture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Overlay for feedback
            if "Incorrect" in posture:
                cv2.rectangle(image, (10, 10), (630, 470), (0, 0, 255), 2)  # Red border for incorrect posture
            else:
                cv2.rectangle(image, (10, 10), (630, 470), (0, 255, 0), 2)  # Green border for correct posture

            # Draw inclination angles
            left_shoulder = [int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)]
            left_ear = [int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * w),
                        int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * h)]

            # Ensure coordinates are integers
            left_shoulder = tuple(map(int, left_shoulder))
            left_ear = tuple(map(int, left_ear))

            cv2.circle(image, left_shoulder, 7, yellow, -1)
            cv2.circle(image, left_ear, 7, yellow, -1)
            cv2.line(image, left_shoulder, left_ear, green if "Incorrect" not in posture else red, 4)
            cv2.putText(image, str(int(neck_inclination)), (left_shoulder[0] + 10, left_shoulder[1]), font, 0.9, green if "Incorrect" not in posture else red, 2)

            # Calculate the time of remaining in a particular posture
            if "Incorrect" in posture:
                good_frames = 0
                bad_frames += 1
            else:
                bad_frames = 0
                good_frames += 1

            good_time = (1 / fps) * good_frames
            bad_time = (1 / fps) * bad_frames

            if good_time > 0:
                time_string_good = 'Good Posture Time: ' + str(round(good_time, 1)) + 's'
                cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
            else:
                time_string_bad = 'Bad Posture Time: ' + str(round(bad_time, 1)) + 's'
                cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)
                # Uncomment the following line to send a warning (e.g., an alert)
                # sendWarning()

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
