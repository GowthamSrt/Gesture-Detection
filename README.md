# Gesture Detection by

## Objective

Develop a prototype to detect a specific gesture within a video sequence. The gesture will be defined by an input image or a short video clip. The task is to analyze a test video and determine whether the gesture occurs. If the gesture is detected, overlay the word "DETECTED" in bright green on the top right corner of the output frame(s).

## Inputs

- A desired gesture representation (this could be a single image or a short video clip).
- A test video with random gestures in which you need to detect the presence of the desired gesture.
- Note: The test video will not have the gestures done in the exact same way as in the gesture to be recognized. For example, the test video can have jumps at varying intensity, and the code shall detect those.

## Output

- Annotate the test video frames where the gesture is detected with "DETECTED" in bright green on the top right corner in frames where the gesture is detected.
- The output can be a processed video or a sequence of annotated frames.

## Requirements

To run the code, you will need to install the following libraries:

- streamlit
- opencv-python
- numpy
- keras
- tensorflow

You can install them using the following command:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the application, use the following command:

```
streamlit run app.py
```

Replace `app.py` with the path to your script if necessary.

## Usage

1. Run the application using the command above.
2. Upload the gesture representation (image or video) using the first file uploader.
3. Upload the test video using the second file uploader.
4. Click the "Detect Gesture" button to start the detection process.
5. The application will display the frames where the gesture is detected with "DETECTED" overlayed in bright green on the top right corner.
