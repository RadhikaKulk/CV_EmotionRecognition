from keras.models import model_from_json
from keras.optimizers import SGD
import numpy as np
from time import sleep
from scipy.ndimage import zoom
import cv2

model = model_from_json(open('model_architecture.json').read())

model.load_weights('model_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

def extract_face_features(gray, detected_face, offset_coefficients):
        #face coordinates
        (x, y, w, h) = detected_face
        
        horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
        vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
	

        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w] 
        
        new_extracted_face = zoom(extracted_face, (48. / extracted_face.shape[0], 48. / extracted_face.shape[1]))

        new_extracted_face = new_extracted_face.astype(np.float32)

        new_extracted_face /= float(new_extracted_face.max())

        return new_extracted_face
        

def detect_face(frame):
        #load classifier
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        #convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(48, 48),
                flags=cv2.cv.CV_HAAR_FEATURE_MAX
            )
        return gray, detected_faces


video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # detect faces
    gray, detected_faces = detect_face(frame)
    
    #face_index = 0
    
    # predict output
    for face in detected_faces:
        (x, y, w, h) = face
        if w > 100:
            # draw rectangle around face 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # extract features
            extracted_face = extract_face_features(gray, face, (0.075, 0.05))

            # predict
            prediction_result = model.predict_classes(extracted_face.reshape(1,48,48,1))

            # give labels
            if prediction_result == 3:
                cv2.putText(frame, "Happy",(x,y), cv2.FONT_ITALIC, 2, 155, 10)
            elif prediction_result == 0:
                cv2.putText(frame, "Angry",(x,y), cv2.FONT_ITALIC, 2, 155, 10)
	    elif prediction_result == 1:
                cv2.putText(frame, "Disgust",(x,y), cv2.FONT_ITALIC, 2, 155, 10)
	    elif prediction_result == 2:
                cv2.putText(frame, "Fear",(x,y), cv2.FONT_ITALIC, 2, 155, 10)
	    elif prediction_result == 4:
                cv2.putText(frame, "Sad",(x,y), cv2.FONT_ITALIC, 2, 155, 10)
	    elif prediction_result == 5:
                cv2.putText(frame, "Surprise",(x,y), cv2.FONT_ITALIC, 2, 155, 10)
 	    else :
                cv2.putText(frame, "Neutral",(x,y), cv2.FONT_ITALIC, 2, 155, 10)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
