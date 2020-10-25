import cv2, dlib, sys
import numpy as np
import dlib, cv2, os
from imutils import face_utils

SCALE_FACTOR = 0.3 # img resize

# initialize face detector and shape predictor
detector  = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')
predictor = dlib.shape_predictor('landmarkDetector.dat')

print(' step 1')
# load video
cap = cv2.VideoCapture('samples/22.mp4') # 0 video
# load overlay image
overlay = cv2.imread('samples/ryan_transparent.png', cv2.IMREAD_UNCHANGED)

# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  try:
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
      bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
      img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
    return bg_img
  except Exception : return background_img  

face_roi = []
face_sizes = []

# loop
while True:
    # read frame buffer from video
    ret, img = cap.read()
    if not ret:
        break
    print(' step 2')
    # resize frame
    img_result = img.copy()
    img = cv2.resize(img, dsize=None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    ori = img.copy()
    dets = detector(img, upsample_num_times=1)
    '''
    # find faces
    if len(face_roi) == 0:
      dets = detector(img, upsample_num_times=1)
    else:
      roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
      # cv2.imshow('roi', roi_img)
      dets = detector(roi_img)
    # no faces
    if len(dets) == 0:
      print('no faces!')  
    '''
    # find facial landmarks
    print(' step 3')
    for i, d in enumerate(dets): # rectangle 
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

        x1, y1 = int(d.rect.left() / SCALE_FACTOR), int(d.rect.top() / SCALE_FACTOR)
        x2, y2 = int(d.rect.right() / SCALE_FACTOR), int(d.rect.bottom() / SCALE_FACTOR)

        cv2.rectangle(img_result, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(255,0,0), lineType=cv2.LINE_AA)
    
        # Detect Landmarks
        # forehead 0 right ear 1 right eye 2 nose 3 left ear 4 left eye 5 
        shape = predictor(img, d.rect)
        shape = face_utils.shape_to_np(shape)
        #print(shape)

        # computer face center 
        center = np.mean(shape, axis=0).astype(np.int)
        center_x, center_y = np.mean(shape, axis=0).astype(np.int) 
        
        # computer face boundaries
        top_left =np.min(shape, axis=0)
        bottom_right =np.max(shape, axis=0)
        

        face_size = int(max(bottom_right - top_left) * 1.8)
        #face_size = max(bottom_right - top_left)

        result = overlay_transparent(img, overlay, center_x, center_y, overlay_size=(face_size, face_size)) 

        # visualize
        print(' step 4')
        for i, p in enumerate(shape):
            cv2.circle(img_result, center=tuple((p / SCALE_FACTOR).astype(int)), radius=5, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(img_result, center=tuple((top_left / SCALE_FACTOR).astype(int)),radius=6,color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(img_result, center=tuple((bottom_right / SCALE_FACTOR).astype(int)),radius=6,color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(img_result, center=tuple((center / SCALE_FACTOR).astype(int)),radius=7,color=(255,255,0), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img_result, str(i), tuple((p / SCALE_FACTOR).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)


    # visualize
    cv2.imshow('original', ori)
    cv2.imshow('facial landmarks',img_result)
    cv2.imshow('result',result)
    
    if cv2.waitKey(1) == ord('q'):
      sys.exit(1)