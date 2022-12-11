#/Users/nouraroomi/Documents/UW/Machine Learning/Project/.venv/bin/python
import cv2
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os

def load_model(model_path):
    """
    returns the loaded the model
    """
    return keras.models.load_model(model_path)

#set test images folder path
test_images_folder = "581A_ProjectHAPN/data/predictions/test_images"


#  load the CascadeClassifier xml file
classifier = cv2.CascadeClassifier('581A_ProjectHAPN/data/opencv-files/haarcascade_frontalface_default.xml')

model_config = []
#set mask & nomask model config model path classes and test images
mask_nomask_model_config = {

"model_path" : "581A_ProjectHAPN/model/trained_mask_nomask",
"classes" : ["mask","nomask"],
"test_images" : ["people_masks2.jpeg","People.jpg"]
}
model_config.append(mask_nomask_model_config)
#set glasses & noglasses model config model path classes and test images
glasses_noglasses_model_config = {

"model_path" : "581A_ProjectHAPN/model/trained_glasses_noglasses",
"classes" : ["glasses","noglasses"],
"test_images" : ["glasses.jpeg","glasses_1.jpeg","People.jpg"]
}
model_config.append(glasses_noglasses_model_config)

# set model type
model = "glasses" # "glasses"  or "masks"
test_image_i = 0

# fetch model config
if model == "glasses":
    config = model_config[1]
else:
    config = model_config[0]


# check if test_image exist  in test_images
if len(config["test_images"])< test_image_i:
    raise Exception("image does not exist in test_images")

#set test image path
test_image_path = os.path.join(test_images_folder,config["test_images"][test_image_i])

# check if test_image exist
if not(os.path.exists(test_image_path)):
    raise Exception("image does not exist")


#set image segment dimentions
DIM = (128, 120)
#read image
img = cv2.imread(test_image_path)
#grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayscaled_img = tf.keras.utils.img_to_array(tf.image.rgb_to_grayscale(img.copy()))
# detected faces 
faces = classifier.detectMultiScale(gray, 1.3, 5)
#load model
model = load_model(config["model_path"])
for (x,y,w,h) in faces:
    x1,  y1, x2 , y2  = x, y, x+w, y+h
    gray_segment = grayscaled_img[y1:y2, x1:x2]
    # resize segment
    resized_gray_segment = cv2.resize(gray_segment, DIM)
    # reshape segment
    resheped_gray_segmen = resized_gray_segment.reshape(1,DIM[1],DIM[0],1)
    # predict segment
    prediction = model.predict(resheped_gray_segmen)
    # get segment predicted class
    predicted_class = np.argmax(prediction)
    # get segment predicted classes scores
    score = tf.nn.softmax(prediction)
    # get segment predicted class confidence level
    confid = score[0][predicted_class]*100
    # fetch model classes form config
    classes = config["classes"]
    # print(prediction[0],predicted_class,confid)
    # draw rectangled and add classes labels in image
    if predicted_class== 0: 
        cv2.putText(img, classes[predicted_class],(x,y+12), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0), 1)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    else:
        cv2.putText(img, classes[predicted_class],(x,y+12), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0), 1)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
#show image
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
