# Import libraries
import cv2
import numpy as np
import mediapipe as mp
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import threading
import time

# Global variable for 3D points with lock
datapoint_lock = threading.Lock()
latest_3d_points = np.zeros((21, 3))

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Triangulation from three camera views
def triangulate_points(pt1, pt2, pt3, P1, P2, P3):
    pts_4d_12 = cv2.triangulatePoints(P1, P2, np.array(pt1).T, np.array(pt2).T)
    pts_4d_12 /= pts_4d_12[3]
    pts_temp1 = pts_4d_12[:3]

    pts_4d_13 = cv2.triangulatePoints(P1, P3, np.array(pt1).T, np.array(pt3).T)
    pts_4d_13 /= pts_4d_13[3]
    pts_temp2 = pts_4d_13[:3]

    pts_combined = (pts_temp1 + pts_temp2) / 2
    return pts_combined.T

# Dummy projection matrices (should be replaced with real calibration data)
P1 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]], dtype=np.float32)
P2 = np.array([[1, 0, 0, -30],
               [0, 1, 0, 0],
               [0, 0, 1, 0]], dtype=np.float32)
P3 = np.array([[1, 0, 0, 30],
               [0, 1, 0, 0],
               [0, 0, 1, 0]], dtype=np.float32)

def camera_thread():
    global latest_3d_points
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    cap3 = cv2.VideoCapture(2)

    if not (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
        print("Không mở được đủ 3 camera")
        return

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        if not (ret1 and ret2 and ret3):
            print(" Không đọc được từ 3 camera")
            break

        rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        rgb3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)

        results1 = hands.process(rgb1)
        results2 = hands.process(rgb2)
        results3 = hands.process(rgb3)

        if results1.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame1, results1.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        if results2.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame2, results2.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        if results3.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame3, results3.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        if results1.multi_hand_landmarks and results2.multi_hand_landmarks and results3.multi_hand_landmarks:
            pts1, pts2, pts3 = [], [], []
            for i in range(21):
                h, w, _ = frame1.shape
                lm1 = results1.multi_hand_landmarks[0].landmark[i]
                lm2 = results2.multi_hand_landmarks[0].landmark[i]
                lm3 = results3.multi_hand_landmarks[0].landmark[i]
                pts1.append([[lm1.x * w], [lm1.y * h]])
                pts2.append([[lm2.x * w], [lm2.y * h]])
                pts3.append([[lm3.x * w], [lm3.y * h]])

            pts3d = triangulate_points(pts1, pts2, pts3, P1, P2, P3)
            with datapoint_lock:
                latest_3d_points = pts3d.copy()
            print("Updated 3D points:", pts3d[:3])

        cv2.imshow("Cam 1 - Center", frame1)
        cv2.imshow("Cam 2 - Left", frame2)
        cv2.imshow("Cam 3 - Right", frame3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.03)

    cap1.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()

threading.Thread(target=camera_thread, daemon=True).start()

app = Dash(__name__)

app.layout = html.Div([
    html.H1("3D Hand Tracking with Stereo Vision (3 Cams)"),
    dcc.Graph(id='live-graph'),
    dcc.Interval(id='interval-component', interval=200, n_intervals=0)
])

@app.callback(
    Output('live-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    with datapoint_lock:
        points = latest_3d_points.copy()

    print("Graph frame", n, "| Sample points:", points[:3])

    x_vals = [pt[0] for pt in points]
    y_vals = [pt[1] for pt in points]
    z_vals = [pt[2] for pt in points]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='markers+lines',
            marker=dict(size=5, color='blue'),
            line=dict(color='darkblue', width=2)
        )
    ])
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)