from __future__ import print_function
import numpy as np
from tensorflow.python.tools.freeze_graph import freeze_graph
import tfcoreml
from skimage import io, transform, util
import skimage

# Provide these to run freeze_graph:
# Graph definition file, stored as protobuf TEXT
graph_def_file = 'Serial/DexCNN.pbtxt'
# Trained model's checkpoint name
checkpoint_file = 'Serial/DexCNN.ckpt'
# Frozen model's output name
frozen_model_file = './DexCNN.pb'
# Output nodes. If there're multiple output ops, use comma separated string, e.g. "out1,out2".
output_node_names = 'Softmax'

freeze_graph(input_graph=graph_def_file,
             input_saver="",
             input_binary=False,
             input_checkpoint=checkpoint_file,
             output_node_names=output_node_names,
             restore_op_name="save/restore_all",
             filename_tensor_name="save/Const:0",
             output_graph=frozen_model_file,
             clear_devices=True,
             initializer_nodes="")

# convert to coreML

# Provide these inputs in addition to inputs in Step 1
# A dictionary of input tensors' name and shape (with batch)
input_tensor_shapes = {"Placeholder:0":[1, 96, 96, 3]} # batch size is 1
# Output CoreML model path
coreml_model_file = './DexCNN.mlmodel'
output_tensor_names = ['Softmax:0']


coreml_model = tfcoreml.convert(
        tf_model_path=frozen_model_file,
        mlmodel_path=coreml_model_file,
        input_name_shape_dict=input_tensor_shapes,
        output_feature_names=output_tensor_names,
        image_input_names='Placeholder:0',
        class_labels = 'Serial/DexCNN_labels.txt',
                     image_scale = 1.0/255.0)

# Test the converted model
# # Provide CoreML model with a dictionary as input. Change ':0' to '__0'
# # as Swift / Objective-C code generation do not allow colons in variable names
# img = io.imread("test/squirtle_x.png")
# # img_resized = transform.resize(img, (96,96,3))
# # input = np.array(img_resized, dtype="float") / 255.0
# # # inp = np.zeros((1,1,96,96,3))
# # # inp[0][0] = input
# # # print(inp.shape)
# coreml_inputs = {'Placeholder__0': img} # (sequence_length=1,batch=1,channels=784)
# coreml_output = coreml_model.predict(coreml_inputs, useCPUOnly=False)
# print(coreml_output)
