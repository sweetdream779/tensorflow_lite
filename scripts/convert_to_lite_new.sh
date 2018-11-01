#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/irina/Desktop/tensorflowModelsProjects/tensorflow_models/tf_models_new/research:/home/irina/Desktop/tensorflowModelsProjects/tensorflow_models/tf_models_new/research/slim/

export DIRR=/home/irina/Desktop/ssd_models_test/ssdlite_mobilenet_v2_coco_2018_05_09
export MODEL_NAME=ssdlite

export CONFIG_FILE=$DIRR/pipeline.config
export CHECKPOINT_PATH=$DIRR/model.ckpt
export OUTPUT_DIR=$DIRR

cd /home/irina/Desktop/tensorflowModelsProjects/tensorflow_models/tf_models_new/research
python3 object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=$CONFIG_FILE \
--trained_checkpoint_prefix=$CHECKPOINT_PATH \
--output_directory=$OUTPUT_DIR \
--add_postprocessing_op=true

cd /home/irina/tensorflow
#bazel build tensorflow/contrib/lite/toco:toco
#echo "BUILDED"

bazel-bin/tensorflow/contrib/lite/toco/toco \
--input_file=$OUTPUT_DIR/tflite_graph.pb \
--output_file=$OUTPUT_DIR/frozen_inference_"$MODEL_NAME"_quan2.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
--inference_type=QUANTIZED_UINT8 \
--input_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--allow_custom_ops \
--change_concat_input_ranges=false

echo "================"
echo "================"
#bazel run -c opt tensorflow/contrib/lite/toco:toco -- \
bazel-bin/tensorflow/contrib/lite/toco/toco \
--input_file=$OUTPUT_DIR/tflite_graph.pb \
--output_file=$OUTPUT_DIR/frozen_inference_"$MODEL_NAME"_notquan2.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
--input_type=FLOAT \
--inference_type=FLOAT \
--mean_values=128 \
--std_values=128 \
--allow_custom_ops
