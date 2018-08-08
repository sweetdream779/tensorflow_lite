#evaluation script using tensorflow/models/object_detection 's pascal voc metrics
#(!!!only for object detection task!!!)

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import object_detection_evaluation
from object_detection.metrics import coco_evaluation
from object_detection.core import standard_fields as fields

import numpy as np
import cv2

from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper

label_map_path = "/home/irina/Desktop/mscoco_label_map.pbtxt"
model_file = "/home/irina/Desktop/ssd_models_test/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_ssdv1_notquan2.tflite"
input_mean = 128
input_std = 128
floating_model = False
test_data_file_path = "/home/irina/Desktop/ssd_models_test/test_data.txt"
label_file = "/home/irina/Desktop/ssd_models_test/coco_labels_list.txt"

EVAL_METRICS_CLASS_DICT = {
    'pascal_voc_detection_metrics':
        object_detection_evaluation.PascalDetectionEvaluator,
    'weighted_pascal_voc_detection_metrics':
        object_detection_evaluation.WeightedPascalDetectionEvaluator,
    'pascal_voc_instance_segmentation_metrics':
        object_detection_evaluation.PascalInstanceSegmentationEvaluator,
    'weighted_pascal_voc_instance_segmentation_metrics':
        object_detection_evaluation.WeightedPascalInstanceSegmentationEvaluator,
    'open_images_V2_detection_metrics':
        object_detection_evaluation.OpenImagesDetectionEvaluator,
    'coco_detection_metrics':
        coco_evaluation.CocoDetectionEvaluator,
    'coco_mask_metrics':
        coco_evaluation.CocoMaskEvaluator,
    'oid_challenge_object_detection_metrics':
        object_detection_evaluation.OpenImagesDetectionChallengeEvaluator,
}

EVAL_DEFAULT_METRIC = 'pascal_voc_detection_metrics'

def load_labels(filename):
  my_labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    my_labels.append(l.strip())
  return my_labels

def get_evaluators(categories):
  """Returns the evaluator class according to eval_config, valid for categories.

  Args:
    eval_config: evaluation configurations.
    categories: a list of categories to evaluate.
  Returns:
    An list of instances of DetectionEvaluator.

  Raises:
    ValueError: if metric is not in the metric class dictionary.
  """
  eval_metric_fn_keys = [EVAL_DEFAULT_METRIC]
  evaluators_list = []
  for eval_metric_fn_key in eval_metric_fn_keys:
    evaluators_list.append(
        EVAL_METRICS_CLASS_DICT[eval_metric_fn_key](categories=categories))
  return evaluators_list

def toAbsolute(detection_boxes, orig_shape):
  output_boxes = []
  height, width,_ = orig_shape
  for box in detection_boxes:
    y_min, x_min, y_max, x_max = box
    output_boxes.append([x_min * width, y_min * height, x_max * width, y_max * height])
  return np.array(output_boxes, dtype=np.float32)



def result_dict_for_single_example(image, orig_shape,
                                   key,
                                   detections,
                                   groundtruth):
  """Merges all detection and groundtruth information for a single example.

  Note that evaluation tools require classes that are 1-indexed, and so this
  function performs the offset. If `class_agnostic` is True, all output classes
  have label 1.

  Args:
    image: A single 4D uint8 image tensor of shape [1, H, W, C].
    key: A single string tensor identifying the image.
    detections: A dictionary of detections, returned from
      DetectionModel.postprocess().
    groundtruth: (Optional) Dictionary of groundtruth items, with fields:
      'groundtruth_boxes': [num_boxes, 4] float32 tensor of boxes, in
        normalized coordinates.
      'groundtruth_classes': [num_boxes] int64 tensor of 1-indexed classes.
      'groundtruth_area': [num_boxes] float32 tensor of bbox area. (Optional)
      'groundtruth_is_crowd': [num_boxes] int64 tensor. (Optional)
      'groundtruth_difficult': [num_boxes] int64 tensor. (Optional)
      'groundtruth_group_of': [num_boxes] int64 tensor. (Optional)
      'groundtruth_instance_masks': 3D int64 tensor of instance masks
        (Optional).
    class_agnostic: Boolean indicating whether the detections are class-agnostic
      (i.e. binary). Default False.
    scale_to_absolute: Boolean indicating whether boxes and keypoints should be
      scaled to absolute coordinates. Note that for IoU based evaluations, it
      does not matter whether boxes are expressed in absolute or relative
      coordinates. Default False.

  Returns:
    A dictionary with:
    'original_image': A [1, H, W, C] uint8 image tensor.
    'key': A string tensor with image identifier.
    'detection_boxes': [max_detections, 4] float32 tensor of boxes, in
      normalized or absolute coordinates, depending on the value of
      `scale_to_absolute`.
    'detection_scores': [max_detections] float32 tensor of scores.
    'detection_classes': [max_detections] int64 tensor of 1-indexed classes.
    'detection_masks': [max_detections, H, W] float32 tensor of binarized
      masks, reframed to full image masks.
    'groundtruth_boxes': [num_boxes, 4] float32 tensor of boxes, in
      normalized or absolute coordinates, depending on the value of
      `scale_to_absolute`. (Optional)
    'groundtruth_classes': [num_boxes] int64 tensor of 1-indexed classes.
      (Optional)
    'groundtruth_area': [num_boxes] float32 tensor of bbox area. (Optional)
    'groundtruth_is_crowd': [num_boxes] int64 tensor. (Optional)
    'groundtruth_difficult': [num_boxes] int64 tensor. (Optional)
    'groundtruth_group_of': [num_boxes] int64 tensor. (Optional)
    'groundtruth_instance_masks': 3D int64 tensor of instance masks
      (Optional).

  """
  label_id_offset = 1  # Applying label id offset (b/63711816)

  input_data_fields = fields.InputDataFields
  output_dict = {
      input_data_fields.original_image: image,
      input_data_fields.key: key,
  }

  detection_fields = fields.DetectionResultFields

  detection_boxes = detections[detection_fields.detection_boxes]
  detection_scores = detections[detection_fields.detection_scores]

  detection_classes = (
        detections[detection_fields.detection_classes] +
        label_id_offset)

  num_detections = int(detections[detection_fields.num_detections])
  detection_boxes = detection_boxes[:num_detections]
  detection_scores = detection_scores[:num_detections]
  detection_classes = detection_classes[:num_detections]
  detection_boxes = toAbsolute(detection_boxes, orig_shape)

  output_dict[detection_fields.detection_boxes] = detection_boxes
  output_dict[detection_fields.detection_classes] = detection_classes
  output_dict[detection_fields.detection_scores] = detection_scores
  
  if groundtruth:
    output_dict.update(groundtruth)
    
  return output_dict

def get_pred_dict(img, interpreter, input_details, output_details):
  p_index = 0
  o_index = 1
  s_index = 2
  nd_index = 3

  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]

  detection_fields = fields.DetectionResultFields
  detections = {}

  img = cv2.resize(img,(width, height))
  input_data = np.expand_dims(img, axis=0)
  
  if floating_model:
    input_data = (np.float32(input_data) - input_mean) / input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()

  detections[detection_fields.detection_boxes] = np.squeeze(interpreter.get_tensor(output_details[p_index]['index']))
  detections[detection_fields.detection_scores] = np.squeeze(interpreter.get_tensor(output_details[s_index]['index']))
  detections[detection_fields.detection_classes] = np.squeeze(interpreter.get_tensor(output_details[o_index]['index']))
  detections[detection_fields.num_detections] = np.squeeze(interpreter.get_tensor(output_details[nd_index]['index']))

  return detections

def get_gt_dict(img_shape, annot_name):
  '''
  Returns: Dictionary of groundtruth items, with fields:
      'groundtruth_boxes': [num_boxes, 4] float32 tensor of boxes, in
        normalized coordinates.
      'groundtruth_classes': [num_boxes] int64 tensor of 1-indexed classes.
  '''
  orig_height, orig_width, _ = img_shape
  detections = []
  classes = []
  with open(annot_name, "r") as f:
    for line in f:
      line = line.split("\n")[0]
      x_min, y_min, x_max, y_max, clas = line.split(" ")
      x_min = int(x_min)
      x_max = int(x_max)
      y_min = int(y_min)
      y_max = int(y_max)
      clas = int(clas)
      detections.append([x_min, y_min, x_max, y_max])
      classes.append(clas)
    
  #detections = tf.convert_to_tensor(detections, dtype=tf.float32)
  #classes = tf.convert_to_tensor(classes, dtype=tf.int64)

  groundtruth = {
        fields.InputDataFields.groundtruth_boxes:
            np.array(detections, dtype=np.float32),
        fields.InputDataFields.groundtruth_classes:
            np.array(classes, dtype=np.int64)}

  return groundtruth

def get_all_examples(test_data_file_path):
  images = []
  annots = []
  with open(test_data_file_path, "r") as f:
    for line in f:
      line = line.split("\n")[0]
      image_path, annot_path = line.split(" ")

      images.append(image_path)
      annots.append(annot_path)
  return images, annots

if __name__ == "__main__":
    interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    if input_details[0]['dtype'] == type(np.float32(1.0)):
      floating_model = True

    labels = load_labels(label_file)

    label_map = label_map_util.load_labelmap(label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes)

    evaluators = get_evaluators(categories)
    counters = {'skipped': 0, 'success': 0}

    images_paths, annnots_paths = get_all_examples(test_data_file_path)
    num_examples = len(images_paths)
    
    #try:
    for batch in range(num_examples):
      file_name = images_paths[batch]
      annot_name = annnots_paths[batch]
      img = cv2.imread(file_name)
      img_shape = img.shape

      if (batch + 1) % 100 == 0:
        print('Running eval ops batch', batch + 1,"/", num_examples)
      
      prediction_dict = get_pred_dict(img, interpreter, input_details, output_details)
      gt_dict = get_gt_dict(img_shape, annot_name)
      result_dict = result_dict_for_single_example(np.expand_dims(img, axis=0), img_shape, batch, prediction_dict, gt_dict)
      
      counters['success'] += 1

      for evaluator in evaluators:          
        evaluator.add_single_ground_truth_image_info(
                image_id=batch, groundtruth_dict=result_dict)
        evaluator.add_single_detected_image_info(
                image_id=batch, detections_dict=result_dict)
    print('Running eval batches done.')
    #except tf.errors.OutOfRangeError:
    #  print('Done evaluating -- epoch limit reached')
    #finally:
    print('# success: ', counters['success'])
    print('# skipped: ', counters['skipped'])
    all_evaluator_metrics = {}
    for evaluator in evaluators:
      metrics = evaluator.evaluate()
      evaluator.clear()
      all_evaluator_metrics.update(metrics)

    print("Result: ", all_evaluator_metrics)
