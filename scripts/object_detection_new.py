# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""objection_detection for tflite"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import time
from heapq import heappush, nlargest

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper
import cv2


NUM_CLASSES = 91

X_SCALE = 10.0
Y_SCALE = 10.0
H_SCALE = 5.0
W_SCALE = 5.0

def load_labels(filename):
  my_labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    my_labels.append(l.strip())
  return my_labels

def iou(box_a, box_b):
  x_a = max(box_a[0], box_b[0])
  y_a = max(box_a[1], box_b[1])
  x_b = min(box_a[2], box_b[2])
  y_b = min(box_a[3], box_b[3])

  intersection_area = (x_b - x_a + 1) * (y_b - y_a + 1)

  box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
  box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

  iou = intersection_area / float(box_a_area + box_b_area - intersection_area)
  return iou

def nms(p, iou_threshold, max_boxes):
  sorted_p = sorted(p, reverse=True)
  selected_predictions = []
  for a in sorted_p:
    if len(selected_predictions) > max_boxes:
      break
    should_select = True
    for b in selected_predictions:
      if iou(a[3], b[3]) > iou_threshold:
        should_select = False
        break
    if should_select:
      selected_predictions.append(a)

  return selected_predictions

if __name__ == "__main__":
  file_name = "/home/irina/Desktop/car2.jpg"
  model_file = "/home/irina/Desktop/ssd_models_test/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_ssdv1_quan2.tflite"
  label_file = "/home/irina/Desktop/ssd_models_test/coco_labels_list.txt"
  box_prior_file = "/home/irina/Desktop/ssd_models_test/box_priors.txt"
  input_mean = 128
  input_std = 128
  min_score = 0.5
  max_boxes = 20
  floating_model = False
  show_image = True
  alt_output_order = False

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be classified")
  parser.add_argument("--graph", help=".tflite model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_mean", help="input_mean")
  parser.add_argument("--input_std", help="input standard deviation")
  parser.add_argument("--min_score", help="show only > min_score")
  parser.add_argument("--max_boxes", help="max boxes to show")
  parser.add_argument("--show_image", help="show image")
  parser.add_argument("--alt_output_order", help="alternative output index")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_mean:
    input_mean = float(args.input_mean)
  if args.input_std:
    input_std = float(args.input_std)
  if args.min_score:
    min_score = float(args.min_score)
  if args.max_boxes:
    max_boxes = int(args.max_boxes)
  if args.show_image:
    show_image = args.show_image
  if args.alt_output_order:
    alt_output_order = args.alt_output_order

  interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  #print(input_details)
  #print(output_details)

  # check the type of the input tensor
  if input_details[0]['dtype'] == type(np.float32(1.0)):
    floating_model = True

  print("Floating model: ", floating_model)

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  img = cv2.imread(file_name)
  img = cv2.resize(img,(width, height))

  print("Input shape: ", width, height)

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - input_mean) / input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  finish_time = time.time()
  print("time spent:", (finish_time - start_time))

  #box_priors = []
  #load_box_priors(box_prior_file)
  labels = load_labels(label_file)

  p_index = 0
  o_index = 1
  s_index = 2
  nd_index = 3
  if alt_output_order:
    nd_index = 0
    s_index = 1
    o_index = 2
    p_index = 3    

  predictions = np.squeeze( \
                  interpreter.get_tensor(output_details[p_index]['index']))
  output_classes = np.squeeze( \
                     interpreter.get_tensor(output_details[o_index]['index']))
  output_scores = np.squeeze( \
                  interpreter.get_tensor(output_details[s_index]['index']))
  numDetections = np.squeeze( \
                     interpreter.get_tensor(output_details[nd_index]['index']))
  
  print("Predictions: ", predictions)
  print("Classes: ", output_classes)
  print("Scores: ", output_scores)
  print("NumDetections: ", numDetections)

  labelOffset = 1
  '''
  if not floating_model:
    p_scale, p_mean = output_details[p_index]['quantization']
    o_scale, o_mean = output_details[o_index]['quantization']
    s_scale, s_mean = output_details[s_index]['quantization']
    nd_scale, nd_mean = output_details[nd_index]['quantization']

    print("Predictions: ", p_scale, p_mean)
    print("Classes: ", o_scale, o_mean)
    print("Scores: ", s_scale, s_mean)
    print("NumDetections: ", numDetections)

    predictions = (predictions - p_mean * 1.0) * p_scale
    output_classes = (output_classes - o_mean * 1.0) * o_scale
    output_scores = (output_scores - s_mean * 1.0) * s_scale
    numDetections = (numDetections - nd_mean * 1.0) * nd_scale
  
    

  print("Predictions: ", predictions)
  print("Classes: ", output_classes)
  print("Scores: ", output_scores)
  print("NumDetections: ", numDetections)
  '''

  pruned_predictions = []
  
  for r in range(int(numDetections)):
      score = output_scores[r]
      if(score <= min_score):
        continue
      rect = (predictions[r][1] * width, predictions[r][0] * width, \
                predictions[r][3] * width, predictions[r][2] * width)
      clas = int(output_classes[r] + labelOffset)
      pruned_predictions.append((score, r, labels[clas], rect))


  final_predictions = sorted(pruned_predictions, reverse=True)
  if(len(final_predictions) > max_boxes):
    final_predictions = final_predictions[:max_boxes]
  for e in final_predictions:
    score = e[0]
    score_string = '{0:.2f}'.format(score)
    print(score_string, e[2], e[3])
    
    left, top, right, bottom = e[3]
    left = int(left)
    top = int(top)
    right = int(right)
    bottom = int(bottom)

    if show_image:
      label = e[2] + ":" + score_string
      fontface = cv2.FONT_HERSHEY_SIMPLEX;
      scale = 0.6
      thickness = 1

      text_size, _ = cv2.getTextSize(label, fontface, scale, thickness)
      print(text_size)

      cv2.rectangle(img, (left, top),(right, bottom),(255,0,0),2)
      #cv2.rectangle(img, (left,top), (text_size[0], -text_size[1]), (255,255,255), cv2.FILLED)
      cv2.putText(img,label,(left,top),fontface,scale,(255,0,255),thickness)

  if show_image:
    cv2.imshow("image", img)
    cv2.waitKey(0)
