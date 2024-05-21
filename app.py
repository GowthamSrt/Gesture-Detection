import streamlit as st
import cv2
import numpy as np
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.preprocessing.image import img_to_array
import tempfile

model = MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

def extract_features(image, model):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_gesture_input(uploaded_file, model):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4' if uploaded_file.type == 'video/mp4' else '.jpg') as tmpfile:
            tmpfile.write(uploaded_file.read())
            input_path = tmpfile.name

        if input_path.endswith('.mp4'):
            cap = cv2.VideoCapture(input_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError("Could not read the video file.")
            return extract_features(frame, model)
        else:
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError("Could not read the image file.")
            return extract_features(image, model)
    else:
        return None

def main():
    st.title("Gesture Detection")

    if 'reset' not in st.session_state:
        st.session_state['reset'] = False

    gesture_file = st.file_uploader("Upload the gesture representation (image or video):", type=['jpg', 'mp4'], key="gesture_uploader")
    test_video_file = st.file_uploader("Upload the test video:", type='mp4', key="video_uploader")

    if st.button("Detect Gesture"):
        if gesture_file and test_video_file:
            gesture_features = process_gesture_input(gesture_file, model)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                tmpfile.write(test_video_file.read())
                test_video_path = tmpfile.name

            cap = cv2.VideoCapture(test_video_path)
            threshold = 0.53
            detected_frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
                frame_features = extract_features(frame, model)
                similarity = cosine_similarity(gesture_features, frame_features)

                if similarity > threshold:
                    cv2.putText(frame, 'DETECTED', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detected_frames.append(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

            for frame in detected_frames:
                st.image(frame)
        else:
            st.error("Please upload both the gesture representation and the test video.")

    if st.button("Reset"):
        st.session_state['reset'] = True
        st.experimental_rerun()

if __name__ == "__main__":
    main()


## MyEnv\Scripts\activate
## pip install --upgrade keras
## pip install --upgrade tensorflow
## To run the file: streamlit run "C:\Users\Abhin\Downloads\Gesture Detection in Video Sequences\app.py"

## If the project shows error then try this out:
## pip uninstall opencv-python-headless
## pip install opencv-python