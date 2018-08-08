#evaluation script using tensorflow/models/object_detection 's pascal voc metrics


import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import object_detection_evaluation
from object_detection.metrics import coco_evaluation

label_map_path = "/home/irina/Desktop/mscoco_label_map.pbtxt"

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

label_map = label_map_util.load_labelmap(label_map_path)
max_num_classes = max([item.id for item in label_map.item])
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes)

evaluator_list = get_evaluators(categories)


