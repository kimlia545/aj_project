'''
Reference
#https://github.com/tureckova/Doggie-smile
#https://github.com/kairess/dog_face_detector
https://github.com/kairess/face_detector
'''

# Some pictures are not recognized
import dlib, cv2, os
from imutils import face_utils
import numpy as np
import matplotlib.pyplot as plt
import glob

print(' step 1')
# Load Models
detector  = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')
predictor = dlib.shape_predictor('landmarkDetector.dat')

print(' step 2')
# Load Dog Image
# Choose your img
# You have the following options (head, eye, nose)
img_path      = './img/08.jpg'
deco_path     = './deco/rabbit.png'
filename, ext = os.path.splitext(os.path.basename(img_path))
img           = cv2.imread(img_path)
img           = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(' step 3')
# Detect Faces
dets = detector(img, upsample_num_times=1) # Recognition of multiple faces
print(dets)

print(' step 4')
img_result = img.copy()

print(' step 5')
for i, d in enumerate(dets): # rectangle 
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

    x1, y1 = d.rect.left(), d.rect.top()
    x2, y2 = d.rect.right(), d.rect.bottom()

    cv2.rectangle(img_result, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(255,0,0), lineType=cv2.LINE_AA)

# Detect Landmarks
shapes = [] 

print(' step 6')
for i, d in enumerate(dets): # size dot (6)
    shape = predictor(img, d.rect)
    shape = face_utils.shape_to_np(shape) # dlib shape -> numpy array
    
    for i, p in enumerate(shape):
        shapes.append(shape)
        cv2.circle(img_result, center=tuple(p), radius=3, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(img_result, str(i), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

print(' step 7')
img_out = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
cv2.imwrite('img/%s_out%s' % (filename, ext), img_out)
# forehead 0 right ear 1 right eye 2 nose 3 left ear 4 left eye 5 

# Overlay Reindeer Head and Nose and Eyes
from math import atan2, degrees

print(' step 8')
# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    img_to_overlay_t = cv2.cvtColor(img_to_overlay_t, cv2.COLOR_BGRA2RGBA)
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGB2RGBA)

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
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_RGBA2RGB)

    return bg_img

print(' step 9')
def angle_between(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff))

print(' step 10-1')
# head_decoration
def head_decoration(img_path,deco_path):
    img_result2 = img.copy()
    horns = cv2.imread(deco_path,  cv2.IMREAD_UNCHANGED)
    horns_h, horns_w = horns.shape[:2]
    for shape in shapes:
        horns_center = np.mean([shape[4], shape[1]], axis=0) // [1, 1.3]
        horns_size = np.linalg.norm(shape[4] - shape[1]) * 3
        angle = -angle_between(shape[4], shape[1])
        M = cv2.getRotationMatrix2D((horns_w, horns_h), angle, 1)
        rotated_horns = cv2.warpAffine(horns, M, (horns_w, horns_h))
        try:
            img_result2 = overlay_transparent(img_result2, rotated_horns, horns_center[0], horns_center[1], overlay_size=(int(horns_size), int(horns_h * horns_size / horns_w)))
        except:
            print('failed overlay image')
    img_out2 = cv2.cvtColor(img_result2, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./img/%s_out2%s' % (filename, ext), img_out2) 

print(' step 10-2')
# nose_decoration
def nose_decoration(img_path,deco_path):
    img_result2 = img.copy()
    nose = cv2.imread(deco_path,  cv2.IMREAD_UNCHANGED)
    for shape in shapes:
        nose_center = shape[3]
        nose_size = np.linalg.norm(shape[4] - shape[1]) * 3 // 4
        img_result2 = overlay_transparent(img_result2, nose, nose_center[0], nose_center[1], overlay_size=(int(nose_size), int(nose_size)))
    img_out2 = cv2.cvtColor(img_result2, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./img/%s_out3%s' % (filename, ext), img_out2)

print(' step 10-3')
# mask_decoration
def mask_decoration(img_path,deco_path):
    print('work')
    img_result2 = img.copy()
    mask = cv2.imread(deco_path,  cv2.IMREAD_UNCHANGED)
    for shape in shapes:
        mask_center = shape[3]
        mask_size = np.linalg.norm(shape[4] - shape[1]) * 3 //2
        img_result2 = overlay_transparent(img_result2, mask, mask_center[0], mask_center[1], overlay_size=(int(mask_size), int(mask_size)))
    img_out2 = cv2.cvtColor(img_result2, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./img/%s_out4%s' % (filename, ext), img_out2)

print(' step 10-4')
# eyes_decoration
def eyes_decoration(img_path,deco_path):
    img_result2 = img.copy()
    eyes = cv2.imread(deco_path,  cv2.IMREAD_UNCHANGED)
    eyes_h, eyes_w = eyes.shape[:2]
    for shape in shapes:
        eyes_center = np.mean([shape[5], shape[2]], axis=0)
        eyes_size = np.linalg.norm(shape[5] - shape[2]) * 3
        angle = -angle_between(shape[5], shape[2])
        M = cv2.getRotationMatrix2D((eyes_w, eyes_h), angle, 1)
        rotated_eyes = cv2.warpAffine(eyes, M, (eyes_w, eyes_h))
        try:
            img_result2 = overlay_transparent(img_result2, rotated_eyes, eyes_center[0], eyes_center[1], overlay_size=(int(eyes_size), int(eyes_h * eyes_size / eyes_w)))
        except:
            print('failed overlay image')
    img_out2 = cv2.cvtColor(img_result2, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./img/%s_out5%s' % (filename, ext), img_out2)

if __name__ == '__main__':
    head_decoration(img_path,deco_path)
    #nose_decoration(img_path,deco_path)
    #mask_decoration(img_path,deco_path)
    #eyes_decoration(img_path,deco_path)