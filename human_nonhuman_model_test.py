#/Users/nouraroomi/Documents/UW/Machine Learning/Project/.venv/bin/python
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tensorflow import keras
import tensorflow as tf
from PIL import Image


DIM = (128, 120)

def non_max_suppression(boxes, overlapThresh= .5):
    '''This function performs non maxima suppression.  The function was taken from PyImageSearch.com.  ''' 
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return pick


def get_img_segments(img):
    """
    get image segments on image segmanets form Selective Search
    :param img: original image with RGB
    """
    cv2.setUseOptimized(True)
    cv2.setNumThreads(100) #c
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    # ss.switchToSelectiveSearchQuality()
    segments = ss.process()
    ss.clear()
    return segments

def proccess_segments(grayscaled_img, selected_segments):
    """
    proccess segments image segmanets
    :param img: grayscaled image 
    :param selected_segments: selected segments form image search
    """

    segments = []
    segments_coord = []

    for x, y, w, h in selected_segments: 
        x1,  y1, x2 , y2  = x, y, x+w, y+h
        
        segment = grayscaled_img[y1:y2, x1:x2]

        segment = cv2.resize(segment, DIM)
    
        segments.append(segment)
        segments_coord.append((x1, y1, x2, y2))
    return segments, segments_coord


def load_model(model_path):
    """
    returns the loaded the model
    """
    return keras.models.load_model(model_path)


def get_perdiction(model,segments , segments_coord,confid_level):
    """
    get perdiction on image segmanets
    :param model: traind model
    :param segments: image segmanets
    :param segments_coord: image segmanets coordinates
    :param confid_level: prediction class confidence level
    """
    predictions = model.predict(segments)
    h_prediction = []
    for pred_i, predicted in enumerate(predictions) :
        
        prediction = np.argmax(predictions[pred_i])
        score = tf.nn.softmax(predictions[pred_i])
        confid = score[prediction]*100
        
        if prediction == 0 and confid > confid_level:
            # print(prediction," {:.2f} ".format(score[prediction]*100),confid)
            h_prediction.append(pred_i)
    
    p_coord = np.array([segments_coord[i] for i in h_prediction] )
    nms = non_max_suppression(np.array([segments_coord[i] for i in h_prediction] ))
    return p_coord , nms

def draw_predictions_rectangle(img,p_coord,nms):
    """
    draw predictions rectangles 
    :param img: original image with RGB
    :param p_coord: list perdiction coordintes 
    :param nms: list unique perdiction indecies
    """

    for segment in nms:
        x1, y1, x2, y2 = p_coord[segment]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
    cv2.putText(img, "count:"+str(len(nms)),(0,15), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0), 1)


def write_video(file_path, frames, fps,  height,width):
    """
    writes frames to an mp4 video file 
    :param file_path: path to outfile .mp4 video 
    :param frames: list of frames as (pil.image objects)
    :param fps: frame rate
    :param height: height of frames in outfile 
    :param width: width of frames in outfile 
    """

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()


def perdict_image(image_path,model_path,confid_level):
    """
    perdict humans in an image 
    :param image_path: path to the image 
    :param model_path: path to the trained model
    :param confid_level: prediction class confidence level
    """

    img = cv2.imread(image_path)
    

    grayscaled_img = tf.keras.utils.img_to_array(tf.image.rgb_to_grayscale(img.copy()))

    selected_segments = get_img_segments(img)
    segments, segments_coord = proccess_segments(grayscaled_img, selected_segments)
    segments = np.array(segments)
    segments = segments.reshape(segments.shape[0],DIM[1],DIM[0])
    model  = load_model(model_path)

    p_coord , nms = get_perdiction(model,segments , segments_coord,confid_level)
    draw_predictions_rectangle(img,p_coord,nms)

    cv2.imshow('Image', img)
    cv2.waitKey(0)


def perdict_video(video_path,save_path,model_path,confid_level):
    """
    perdict humans in video frames
    :param video_path: path to the video 
    :param save_path: path to outfile .mp4 video 
    :param model_path: path to the trained model
    :param confid_level: prediction class confidence level
    """
    
    video = cv2.VideoCapture(video_path)
    video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) ) 
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(video_frames,width,height)
    frames = []
    grayscaled_frames = []
    for i in range(video_frames):
        try:
            success, frame = video.read() 
            
            frames.append(frame)
            grayscaled_frames.append(tf.keras.utils.img_to_array(tf.image.rgb_to_grayscale(frame.copy())))

            
        except: 
            break
    model  = load_model(model_path)

    for frame_i in range(275):
        grayscaled_frame = grayscaled_frames[frame_i]
        frame = frames[frame_i] 

        selected_segments = get_img_segments(frame)
        segments, segments_coord = proccess_segments(grayscaled_frame, selected_segments)
        segments = np.array(segments)
        segments = segments.reshape(segments.shape[0],DIM[1],DIM[0])
        p_coord , nms = get_perdiction(model,segments , segments_coord,confid_level)
        draw_predictions_rectangle(frame,p_coord,nms)

    write_video(save_path, frames, fps,height,width)


# set model path
model_path = "581A_ProjectHAPN/model/trained_human_nonhuman"
# set input image/video path
path = "581A_ProjectHAPN/data/predictions/test_images/People.jpg"
# set confidance level 
confid_level = 99
# apply perdiction to the image
perdict_image(path,model_path,confid_level)
# set output video save path
# save_path = "outputfile.mp4"
# perdict_video(path,save_path,model_path,confid_level)

