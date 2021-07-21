import cv2
import numpy as np

#Capturing video through Camera
cap = cv2.VideoCapture(0)

# Initialising our face classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
# Path to store face data
dataset_path = 'C://Users//kshitij goel//Desktop//Real-Time-Criminal-Face-Detection-master//data//'
file_name = input("Enter the name of the person : ")
while True:
    ## Capturing from camera frame by frame
    ret,frame = cap.read()

    if ret==False:
        continue
    ## Converting the captured frame to Gray Scale
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # Detecting all faces in the captured frame using haarcascades
    # Working of haarcascades:-

    # Parameters:- Frame is input image
    # 1.3 is the scale factor which specifies how much the image size is reduced with each scale.Some faces are near the camera than others. 
    # Such faces would appear more prominent than the ones behind. This factor compensates for that
    # 5 is the min neighbors which specifies how many neighbors each candidate rectangle should have to be called a face.
    # This parameter will affect the quality of the detected faces. Higher value results in fewer detections but with higher quality. 3~6 is a good value for it.
    faces = face_cascade.detectMultiScale(frame,1.3,5)

    # Len faces = number of faces detected.
    # Faces contain x , y , w , h to define the rectangle as (x,y) and (x+w,y+h)
    if len(faces)==0:
        continue

    # Sort according to the area of rectangle i.e. w*h
    faces = sorted(faces,key=lambda f:f[2]*f[3])
    
    # Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
    for face in faces[-1:]:
        x,y,w,h = face
        ## To draw a rectangle in an image
        ## (0,255,255) is the color of rectangle in BGR format
        ## 2 is the thickness of rectangle borderline. -1 value will fill the rectangle with given color
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        #Extract (Crop out the required face) : Region of Interest.
        # Creating a buffer region around the face.
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]

        ## Showing the face section extracted
        face_section = cv2.resize(face_section,(100,100),cv2.INTER_AREA)

        ## Frames are captured at a high rate, so to avoid duplicates we store every Nth image as training data
        skip += 1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))

    ## Showing 2 windows, one whole webcam video with rectangle around the face and the other face section with rectangle.
    cv2.imshow("Frame",frame)
    cv2.imshow("Face Section",face_section)
    key_pressed = cv2.waitKey(1) & 0xFF
    ## Stop detection and break out of the loop when q key is pressed.
    if key_pressed == ord('q'):
        break


# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
## Flattening the data
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()