from absl import flags,app,logging
from absl.flags import FLAGS

import time
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image


flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_list('line_coordinates','0,0,0,0','[x1,y1,x2,y2] format')



def main(_argv):

    class_names = [c.strip() for c in open('coco.names').readlines()]
    # class_names=['car', 'truck','bus', 'bicycle','motorbike']
    yolo = YoloV3(classes=len(class_names))
    yolo.load_weights('./weights/yolov3.tf')

    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.8

    model_filename = 'mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
        vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(FLAGS.output, codec, vid_fps, (vid_width, vid_height))

    from _collections import deque
    pts = [deque(maxlen=30) for _ in range(1000)]

    counter = []

    while True:
        _, img = vid.read()
        if img is None:
            print('Completed')
            break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, 416)

        t1 = time.time()

        boxes, scores, classes, nums = yolo.predict(img_in)

        classes = classes[0]
        names = []
        for i in range(len(classes)):

            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)

        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(converted_boxes, scores[0], names, features)]

        boxs,scores,classes=[],[],[]
        f=['car', 'truck','bus', 'bicycle','motorbike']
        for d in detections:
            if d.class_name in f:
                boxs.append(d.tlwh)
                scores.append(d.confidence)
                classes.append(d.class_name)



        boxs = np.array(boxs)
        scores = np.array(scores)
        classes = np.array(classes)
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        tracker.predict()
        tracker.update(detections)

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]



        for track in tracker.tracks:
            if track.class_name in f:
                # print("new track")

                if not track.is_confirmed() or track.time_since_update >1:
                    continue
                bbox = track.to_tlbr()
                class_name= track.get_class()
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]

                cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)


                center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
                pts[track.track_id].append(center)

                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                        continue
                    thickness = int(np.sqrt(64/float(j+1))*2)
                    cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

                height, width, _ = img.shape
                # print("p",height,width)
                # print(int(3*height/6+height/20))

                oo=[int(x) for x in FLAGS.line_coordinates]
                print(oo)
                cv2.line(img, (oo[0], oo[1]), (oo[2], oo[3]),(0, 255, 0), thickness=2)


                center_y = int(((bbox[1])+(bbox[3]))/2)

                if center_y <= int(3*height/6+height/20) and center_y >= int(3*height/6-height/20):

                    counter.append(int(track.track_id))

                    print(int(track.track_id))


        total_count = len(set(counter))
        h,w= img.shape[0:2]
        img[0:70,0:500]=[0,0,0]

        cv2.putText(img, "Total Vehicle Count: " + str(total_count), (7,56), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,255,255), 2)



        cv2.resizeWindow('output', 1024, 768)
        cv2.imshow('output', img)
        out.write(img)



        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass