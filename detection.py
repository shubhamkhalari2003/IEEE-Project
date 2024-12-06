import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
from collections import deque
import os.path
import base64
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


# Email notification function
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def authenticate_gmail_api():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def send_email(subject, message_text, to):
    creds = authenticate_gmail_api()
    service = build('gmail', 'v1', credentials=creds)

    message = MIMEText(message_text)
    message['to'] = to
    message['subject'] = subject
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

    try:
        service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
        print('Message Id: %s' % message['id'])
    except Exception as error:
        print(f'Error sending email: {error}')

# Example usage
send_email("Alert!!", "Take a break, drink coke!!", "selfdependent001@gmail.com")

mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades for face and eye detection
face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haarcascade/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarcascade/haarcascade_righteye_2splits.xml')

# Load the pre-trained model
model = load_model('CNN__model.h5')
path = os.getcwd()
# cap = cv2.VideoCapture(0)  # Access the camera
cap = cv2.VideoCapture('http://192.168.38.75:8080/video')  # Replace with your actual IP address

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2

# Use deque to keep a history of predictions
history_length = 5  # Number of frames to average
rpred_history = deque(maxlen=history_length)
lpred_history = deque(maxlen=history_length)

frame_skip = 5  # Process every 5th frame
frame_count = 0
prev_label = "Unknown"  # Variable to store previous label for stability

while True:
    ret, frame = cap.read()

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (640, 480))  # Reduce the resolution to 640x480

    if frame_count % frame_skip == 0:  # Process every nth frame
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

        # Right Eye Detection
        if len(right_eye) > 0:
            (x, y, w, h) = right_eye[0]
            r_eye = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around the right eye
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye, (100, 100))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(100, 100, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            rpred = model.predict(r_eye)[0][0]  # Get the prediction

            # Store prediction in history
            rpred_history.append(rpred)

        # Left Eye Detection
        if len(left_eye) > 0:
            (x, y, w, h) = left_eye[0]
            l_eye = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around the left eye
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (100, 100))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(100, 100, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            lpred = model.predict(l_eye)[0][0]  # Get the prediction

            # Store prediction in history
            lpred_history.append(lpred)

        # Average predictions
        if len(rpred_history) > 0 and len(lpred_history) > 0:
            avg_rpred = np.mean(rpred_history)
            avg_lpred = np.mean(lpred_history)

            # Determine current label
            if (avg_rpred < 0.5) and (avg_lpred < 0.5):  # Assuming threshold 0.5 for closed eyes
                current_label = "Closed"
                score += 1  # Increment score for closed eyes
            else:
                current_label = "Open"
                score -= 1  # Decrement score for open eyes
                score = max(score, 0)
                sound.stop()
            # Clamp score to non-negative


            # Print current label and score to console
            print(f"Current State: {current_label}, Score: {score}")

            # Display current label and score only if it has changed
            if current_label != prev_label:
                cv2.putText(frame, current_label, (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                prev_label = current_label  # Update previous label

            cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            if score > 6:
                cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
                try:
                    sound.play()
                    send_email()
                except:
                    pass
                if thicc < 16:
                    thicc += 2
                else:
                    thicc -= 2
                    if thicc < 2:
                        thicc = 2
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('Driver Drowsiness Detection', frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
