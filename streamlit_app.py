# import the rquired libraries.
import numpy as np
import cv2
from keras._tf_keras.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow import keras
import mediapipe as mp
from keras._tf_keras.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import tempfile

# Define the emotions.
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Load model.
classifier =load_model('D:/GUVI/PROJECT/face_expression_recognition/Model_training/model_79.keras')

# load weights into new model
classifier.load_weights("D:/GUVI/PROJECT/face_expression_recognition/Model_training/model_weights_78.weights.h5")

# Load face using OpenCV
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(0, 255, 255), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)
            label_position = (x, y-10)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application 😠🤮😨😀😐😔😮")
    activiteis = ["Home", "Live Face Emotion Detection","upload image","using mediapipe","About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    

    # Homepage.
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#FC4C02;padding:0.5px">
                             <h4 style="color:white;text-align:center;">
                            Start Your Real Time Face Emotion Detection.
                             </h4>
                             </div>
                             </br>"""

        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""    
        Let's find out...
        1. Click the dropdown list in the top left corner and select Live Face Emotion Detection.
        2. This takes you to a page which will tell if it recognizes your emotions.
                 """)

    # Live Face Emotion Detection.
    elif choice == "Live Face Emotion Detection":
        st.header("Webcam Live Feed")
        st.subheader('''
        Welcome to the other side of the SCREEN!!!
        * Get ready with all the emotions you can express. 
        ''')
        st.write("1. Click Start to open your camera and give permission for prediction")
        st.write("2. This will predict your emotion.")
        st.write("3. When you done, click stop to end.")
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)
    elif choice == "upload image":
        img_file_buffer = st.file_uploader('Upload a JPG image', type='jpg')
        if img_file_buffer is not None:
            fig1,fig2=st.columns(2)
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            fig1.image(img, channels="BGR")
            #st.image(img, channels="BGR")
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_classifier = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            face = face_classifier.detectMultiScale(
                gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            for (x, y, w, h) in face:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
            plt.figure(figsize=(20,10))
            #plt.imshow(img_rgb)
            fig2.image(img_rgb)
            plt.axis('off')
    elif choice == "using mediapipe":
        #demo video 
        #DEMO_VIDEO = 'D:/GUVI/PROJECT/human_face_detection_cnn/f1.mp4'
        #mediapipe inbuilt solutions 
        #st,st=st.columns
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        #title 
        st.title('Face Detection App')
        detection_confidence = 1.0
        model_selection=1
        st.markdown(' ## Output')
        stframe = st.empty()
        
        #file uploader
        video_file_buffer = st.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
        
        if video_file_buffer is not None:
        #temporary file name 
            tfflie = tempfile.NamedTemporaryFile(delete=False)
            tfflie.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tfflie.name)

            #values 
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
            codec = cv2.VideoWriter_fourcc('V','P','0','9')
            out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))
            st.text('Input Video')
            st.video(tfflie.name)
            # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            with mp_face_detection.FaceDetection(model_selection=model_selection, min_detection_confidence=detection_confidence) as face_detection:
                while vid.isOpened():
                    ret, image = vid.read()
                    if not ret:
                        break
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(image)

                    if results.detections:
                        for detection in results.detections:
                            mp_drawing.draw_detection(image, detection)
                    stframe.image(image,use_column_width=True)

                vid.release()
                out.release()
                cv2.destroyAllWindows()

        st.success('Video is Processed')
        st.stop()



# About.
    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#36454F;padding:30px">
                                    <h4 style="color:white;">
                                     This app predicts facial emotion using a Convolutional neural network.
                                     Which is built using Keras and Tensorflow libraries.
                                     Face detection is achived through openCV.
                                    </h4>
                                    </div>
                                    </br>
                                    """
        st.markdown(html_temp_about1, unsafe_allow_html=True)


    else:
        pass


if __name__ == "__main__":
    main()