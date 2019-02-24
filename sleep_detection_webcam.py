

import numpy as np
import os
import tensorflow as tf

from matplotlib import pyplot as plt
from playsound import playsound

from utils import label_map_util

from utils import visualization_utils as vis_util

# couter for the number of frames 
framecounter = 0

# number of frames where sleeping has been detected, sets off the allarm when it reaches this point
numframes = 15

#True if sleeping was detected in the previous frame False if not 
true_prev_frame = False

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT =  'models/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'  

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data/label_map.pbtxt') 

NUM_CLASSES = 2 


if not os.path.exists('models/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'):
	print ('No model found')
else:
	print ('Starting \n press q to exit')

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#intializing the web camera device

import cv2
cap = cv2.VideoCapture(0)

# Running the tensorflow session
with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		ret = True
		while (ret):
			ret,image_np = cap.read()
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			# Actual detection.
			(boxes, scores, classes, num_detections) = sess.run(
			    [boxes, scores, classes, num_detections],
			    feed_dict={image_tensor: image_np_expanded})
			# Visualization of the results of a detection.
			vis_util.visualize_boxes_and_labels_on_image_array(
			    image_np,
			    np.squeeze(boxes),
			    np.squeeze(classes).astype(np.int32),
			    np.squeeze(scores),
			    category_index,
			    use_normalized_coordinates=True,
			    line_thickness=8)
		  #      plt.figure(figsize=IMAGE_SIZE)
		  #      plt.imshow(image_np)
			cv2.imshow('image',cv2.resize(image_np,(640,480)))
			if int(classes[0][0]) == 2 and scores[0][0] > 0.5:
				framecounter += 1
				true_prev_frame == True
				if framecounter > numframes:
					playsound('object_detection/data/alarm.mp3', 0)
					framecounter = 0
			elif not int(classes[0][0]) == 2:
				framecounter = 0
				true_prev_frame = False
			if cv2.waitKey(25) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				cap.release()
				break
		  