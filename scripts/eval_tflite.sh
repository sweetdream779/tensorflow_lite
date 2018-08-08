export PYTHONPATH=$PYTHONPATH:/home/irina/Desktop/tensorflowModelsProjects/tensorflow_models/tf_models_new/research/:/home/irina/Desktop/tensorflowModelsProjects/tensorflow_models/tf_models_new/research/slim/
#run on cpu
export CUDA_VISIBLE_DEVICES="0"
python3 eval_tflite_.py

