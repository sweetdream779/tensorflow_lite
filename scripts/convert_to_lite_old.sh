#!/bin/bash


export DIRR=/home/irina/Desktop/ssd_models_test/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
export INPUT_MODEL_NAME=frozen_inference_graph_fpn640.pb
export MODEL_NAME=fpn

python3 /home/irina/tensorflow/tensorflow/python/tools/optimize_for_inference.py \
--input=$DIRR/$INPUT_MODEL_NAME \
--output=$DIRR/frozen_inference_"$MODEL_NAME"_stripped.pb \
--frozen_graph=True \
--input_names=Preprocessor/sub \
--output_names=concat,concat_1 \
--alsologtostderr

echo "================="
echo "================="


cd /home/irina/tensorflow
bazel-bin/tensorflow/contrib/lite/toco/toco \
--input_file=$DIRR/frozen_inference_"$MODEL_NAME"_stripped.pb \
--output_file=$DIRR/frozen_inference_"$MODEL_NAME"_quan1.tflite \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--input_shapes=1,640,640,3 \
--input_arrays=Preprocessor/sub \
--output_arrays=concat,concat_1 \
--inference_type=QUANTIZED_UINT8 \
--default_ranges_min=0 \
--default_ranges_max=255 \
--input_type=QUANTIZED_UINT8 \
--logtostderrs

echo "================="
echo "================="

bazel-bin/tensorflow/contrib/lite/toco/toco \
--input_file=$DIRR/frozen_inference_"$MODEL_NAME"_stripped.pb \
--output_file=$DIRR/frozen_inference_"$MODEL_NAME"_notquan1.tflite \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--input_shapes=1,640,640,3 \
--input_arrays=Preprocessor/sub \
--output_arrays=concat,concat_1 \
--inference_type=FLOAT \
--input_type=FLOAT \
--logtostderr
