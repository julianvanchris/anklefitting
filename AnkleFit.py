#from streamlit_webrtc import webrtc_streamer
#import av
import mediapipe as mp
from math import sqrt, acos, degrees, sin, cos
import cv2
import streamlit as st

def calculate_angle(A, B, C):
    AB = (B[0] - A[0], B[1] - A[1])
    AC = (C[0] - A[0], C[1] - A[1])
    
    dot_product = AB[0] * AC[0] + AB[1] * AC[1]    
    magnitude_AB = sqrt(AB[0]**2 + AB[1]**2)
    magnitude_AC = sqrt(AC[0]**2 + AC[1]**2)
    
    cos_theta = dot_product / (magnitude_AB * magnitude_AC)
    
    angle_radians = acos(cos_theta)
    angle_degrees = degrees(angle_radians)
    
    return angle_degrees

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

norm_factor = 5100
pixel_size = 0.00155 # in mm
focal_length = 3.99 # in mm

def video_frame_callback(frame):
    frame = frame.to_ndarray(format="bgr24")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmark_index1 = 25 # left knee
        landmark = results.pose_landmarks.landmark[landmark_index1]
        x1, y1 = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
        
        landmark_index2 = 27 # left ankle
        landmark = results.pose_landmarks.landmark[landmark_index2]
        x2, y2 = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
        
        landmark_index3 = 31 # left foot index
        landmark = results.pose_landmarks.landmark[landmark_index3]
        x3, y3 = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
        
        A = (x2, y2) # joint, left ankle
        B = (x1, y1) # left knee
        C = (x3, y3) # left foot index
        
        x5 = x2
        y5 = y3
        
        D = (x5, y5)
        
        angle = round(calculate_angle(A, B, C), 2)
        angle2 = round(calculate_angle(A, C, D), 2)
        
        # distance between left ankle to left foot index (x2, y2) and (x3, y3)
        dist = sqrt((x2-x3)**2 + (y2-y3)**2)
        
        metatarsal = dist * (3/5)
        
        x6 = x3 + metatarsal * cos(angle2)
        y6 = y3 + metatarsal * sin(angle2)
        
        # distance between left knee to metatarsal (x1, y1) and (x6, y6)
        dist2 = sqrt((x1-x6)**2 + (y1-y6)**2)
        
        sensor_size = dist2 * pixel_size # in mm
        
        real_size = round((sensor_size * norm_factor)/focal_length, 2) / 10
        real_size2 = 0.7332 * real_size + 15.345
        
        text_x = 10
        left_knee = 30 
        text_y1 = 60
        
        left_ankle = 90
        text_y2 = 120
        
        left_footindex = 150
        text_y3 = 180
        
        # angle
        text_y4 = 210
        
        # distance
        text_y5 = 240
        
        x1 = int(x1)
        y1 = int(y1)
        x6 = int(x6)
        y6 = int(y6)
        
        cv2.putText(frame, f'Left Knee', (text_x, left_knee), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f'x1: {x1}, y1: {y1}', (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.putText(frame, f'Left Ankle', (text_x, left_ankle), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f'x2: {x2}, y2: {y2}', (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.putText(frame, f'Left Foot Index', (text_x, left_footindex), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f'x3: {x3}, y3: {y3}', (text_x, text_y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # cv2.line(frame, (x1, y1), (x6, y6), (255, 0, 0), thickness=2)
        
        cv2.putText(frame, f'Angle: {angle}', (text_x, text_y4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.putText(frame, f'Distance: {real_size2} cm', (text_x, text_y5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return av.VideoFrame.from_ndarray(frame, format="bgr24")

def main():
    st.title('Ankle Fitting')
    # webrtc_streamer(key="ankle-fit", video_frame_callback=video_frame_callback)
    st.caption("Powered by OpenCV, Streamlit")
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    start_button_pressed = st.button("Start")
    stop_button_pressed = st.button("Stop")
    while cap.isOpened() and start_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmark_index1 = 25 # left knee
            landmark = results.pose_landmarks.landmark[landmark_index1]
            x1, y1 = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            
            landmark_index2 = 27 # left ankle
            landmark = results.pose_landmarks.landmark[landmark_index2]
            x2, y2 = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            
            landmark_index3 = 31 # left foot index
            landmark = results.pose_landmarks.landmark[landmark_index3]
            x3, y3 = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            
            A = (x2, y2) # joint, left ankle
            B = (x1, y1) # left knee
            C = (x3, y3) # left foot index
            
            x5 = x2
            y5 = y3
            
            D = (x5, y5)
            
            angle = round(calculate_angle(A, B, C), 2)
            angle2 = round(calculate_angle(A, C, D), 2)
            
            # distance between left ankle to left foot index (x2, y2) and (x3, y3)
            dist = sqrt((x2-x3)**2 + (y2-y3)**2)
            
            metatarsal = dist * (3/5)
            
            x6 = x3 + metatarsal * cos(angle2)
            y6 = y3 + metatarsal * sin(angle2)
            
            # distance between left knee to metatarsal (x1, y1) and (x6, y6)
            dist2 = sqrt((x1-x6)**2 + (y1-y6)**2)
            
            sensor_size = dist2 * pixel_size # in mm
            
            real_size = round((sensor_size * norm_factor)/focal_length, 2) / 10
            real_size2 = 0.7332 * real_size + 15.345
            
            text_x = 10
            left_knee = 30 
            text_y1 = 60
            
            left_ankle = 90
            text_y2 = 120
            
            left_footindex = 150
            text_y3 = 180
            
            # angle
            text_y4 = 210
            
            # distance
            text_y5 = 240
            
            x1 = int(x1)
            y1 = int(y1)
            x6 = int(x6)
            y6 = int(y6)
            
            cv2.putText(frame, f'Left Knee', (text_x, left_knee), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f'x1: {x1}, y1: {y1}', (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.putText(frame, f'Left Ankle', (text_x, left_ankle), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f'x2: {x2}, y2: {y2}', (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.putText(frame, f'Left Foot Index', (text_x, left_footindex), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f'x3: {x3}, y3: {y3}', (text_x, text_y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # cv2.line(frame, (x1, y1), (x6, y6), (255, 0, 0), thickness=2)
            
            cv2.putText(frame, f'Angle: {angle}', (text_x, text_y4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.putText(frame, f'Distance: {real_size2} cm', (text_x, text_y5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        frame_placeholder.image(frame, channels="RGB")
        
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
