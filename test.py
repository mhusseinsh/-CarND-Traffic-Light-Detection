import tensorflow as tf
import numpy as np
import cv2
import os

LIGHTS = ['Green', 'Red', 'Yellow', 'Unknown']

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].
    
    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width
    
    return box_coords

def draw_boxes(image, boxes, classes, scores):
    """Draw bounding boxes on the image"""
    print(classes)
    print(scores)
    for i in range(len(boxes)):
        top, left, bot, right = boxes[i, ...]
        cv2.rectangle(image, (left, top), (right, bot), (255,0,0), 3)
        text = LIGHTS[int(classes[i])-1] + ': ' + str(int(scores[i]*100)) + '%'
        cv2.putText(image , text, (left, int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,0,0), 1, cv2.LINE_AA)
        #cv2.putText(image , text, (10, int(50 + i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,0,0), 1, cv2.LINE_AA)

def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)
    
    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes

site = True
if site:
    PATH = 'outputs/frozen_model_real/'
    test_images_path = 'test_images_real'
else: 
    PATH = 'outputs/frozen_model_sim/'
    test_images_path = 'test_images_sim'

FROZEN_GRAPH = PATH + 'frozen_inference_graph.pb'

graph = tf.Graph()
with graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('detection_scores:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')
sess = tf.Session(graph=graph)

test_images = os.listdir(test_images_path)

for testImage in test_images:
    image = cv2.imread(os.path.join(test_images_path, testImage))
    image = np.dstack((image[:, :, 2], image[:, :, 1], image[:, :, 0]))
    image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

    with tf.Session(graph=graph) as sess:                
        # Actual detection.
        (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes], 
                                            feed_dict={image_tensor: image_np})
        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.5
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

    # Write image to disk
    write = True
    if write:
        image = np.dstack((image[:, :, 2], image[:, :, 1], image[:, :, 0]))
        #cv2.imwrite('/home/jose/GitHub/Self-Driving-Car-Nanodegree-Capstone/images/img_raw.jpg', image)
        width, height = image.shape[1], image.shape[0]
        box_coords = to_image_coords(boxes, height, width) 
        draw_boxes(image, box_coords, classes, scores)
        output_path = test_images_path.split("/")[1]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_image = os.path.join("test_images_output", output_path, testImage.split(".")[0] + "_output." + testImage.split(".")[1])
        cv2.imwrite(output_image, image)