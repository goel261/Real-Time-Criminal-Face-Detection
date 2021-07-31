# Real-Time-Criminal-Face-Detection
This project uses haarcascade classifier by open-cv to detect faces. This gives almost 100% accuracy on Indian face dataset. This can be used in CCTV cameras to detect criminal faces.

**Detected Results**



**How to run?**

1) Create a folder named data along with these files.
2) Run face_data_collect.py to store face data.
3) Run face_recognition.py to detect faces.

**Explanation:**
 
This project has 2 parts: First program to collect face data and second program to recognize faces.

**Face Data Collection**

 **1. Read and show video stream from webcam, and capture images frame by frame.**   
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    
 2. **Detect Faces and show bounding box using haarcascade classifier.
    **Haar Cascade Classifier**
    Haar Cascade classifiers are an effective way for object detection. This method was proposed by Paul Viola and Michael Jones in their paper Rapid Object Detection using a 
    Boosted Cascade of Simple Features .Haar Cascade is a machine learning-based approach where a lot of positive and negative images are used to train the classifier. 
 

    Positive images – These images contain the images which we want our classifier to identify.
    Negative Images – Images of everything else, which do not contain the object we want to detect.
    
    The haar cascade files can be downloaded from the OpenCV Github repository.
    https://github.com/opencv/opencv/tree/master/data/haarcascades
    
    OpenCV's algorithm is currently using the following Haar-like features which are the input to the basic classifiers:
    Each feature is a single value obtained by subtracting the sum of pixels under the white rectangle from the sum of pixels under the black rectangle.
    ![image](https://user-images.githubusercontent.com/43380724/125499747-4a35934a-2ec9-472a-896e-993d23e592e8.png)
    
    Now all possible sizes and locations of each kernel are used to calculate plenty of features. For each feature calculation, we need to find the sum of the pixels under the
    white and black rectangles. To solve this, they introduced the integral images. It simplifies calculation of the sum of the pixels, how large may be the number of pixels, to
    an operation involving just four pixels.

    But among all these features we calculated, most of them are irrelevant. For example, consider the image below. Top row shows two good features. The first feature selected       seems to focus on the property that the region of the eyes is often darker than the region of the nose and cheeks. The second feature selected relies on the property that       the eyes are darker than the bridge of the nose. But the same windows applying on cheeks or any other place is irrelevant. So how do we select the best features out of           160000+ features? It is achieved by Adaboost.
    ![image](https://user-images.githubusercontent.com/43380724/125500220-708b3cbb-5074-4114-beff-e020af5f4cf8.png)


    For this, we apply each and every feature on all the training images. For each feature, it finds the best threshold which will classify the faces to positive and negative.       But obviously, there will be errors or misclassifications. We select the features with minimum error rate, which means they are the features that best classifies the face       and non-face images. (The process is not as simple as this. Each image is given an equal weight in the beginning. After each classification, weights of misclassified images     are increased. Then again same process is done. New error rates are calculated. Also new weights. The process is continued until required accuracy or error rate is achieved     or required number of features are found).

    Final classifier is a weighted sum of these weak classifiers. It is called weak because it alone can't classify the image, but together with others forms a strong               classifier. The paper says even 200 features provide detection with 95% accuracy. Their final setup had around 6000 features. (Imagine a reduction from 160000+ features to       6000 features. That is a big gain).

    So now you take an image. Take each 24x24 window. Apply 6000 features to it. Check if it is face or not. Wow.. Wow.. Isn't it a little inefficient and time consuming? Yes,       it is. Authors have a good solution for that.
    
    In an image, most of the image region is non-face region. So it is a better idea to have a simple method to check if a window is not a face region. If it is not, discard it     in a single shot. Don't process it again. Instead focus on region where there can be a face. This way, we can find more time to check a possible face region.

    For this they introduced the concept of Cascade of Classifiers. Instead of applying all the 6000 features on a window, group the features into different stages of               classifiers and apply one-by-one. (Normally first few stages will contain very less number of features). If a window fails the first stage, discard it. We don't consider         remaining features on it. If it passes, apply the second stage of features and continue the process. The window which passes all stages is a face region. How is the plan !!!

    Authors' detector had 6000+ features with 38 stages with 1, 10, 25, 25 and 50 features in first five stages. (Two features in the above image is actually obtained as the         best two features from Adaboost). According to authors, on an average, 10 features out of 6000+ are evaluated per sub-window.
    
    ![image](https://user-images.githubusercontent.com/43380724/125500468-4b9e215d-6490-42d5-907b-b034bd5b1e74.png)

    
**3. Flatten the largest face image(gray scale) and save in a numpy array.**
 
**4. Repeat the above for multiple people to generate training data.**


**Face Recognition**

**1. Load the training data (numpy arrays of all the persons)**

    x- values are stored in the numpy arrays

    y-values we need to assign for each person

**2. Read a video stream using opencv**

**3. Extract faces out of it.**
    This is done using Cascade Classifier.

**4. Use knn to find the prediction of face (int)**
    KNN is used to find the label with maximum frequency after sorting the labels according to their frequency.

**5. Map the predicted id to name of the user**

**6. Display the predictions on the screen - bounding box and name**
    Face section is drawn around the detected face and below which name and crime of the person is given.
