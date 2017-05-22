'''
HTTP Rest server for allowing the model to be infered from a json type request
References:
    https://github.com/sugyan/tensorflow-mnist
    https://github.com/benman1/tensorflow_flask
    https://github.com/sofeikov/WebImageCrop/blob/master/main.py
    http://stackoverflow.com/questions/4628529/how-can-i-created-a-pil-image-from-an-in-memory-file
    https://uk.mathworks.com/help/matlab/ref/webread.html
    https://uk.mathworks.com/help/matlab/ref/webwrite.html
'''

# Import Flask stuff
from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf
import argparse
import scipy.misc
import model
import io
from PIL import Image

# Force to see just the first GPU
# https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/
import os

# Parser command arguments
# Reference:
# https://www.youtube.com/watch?v=cdblJqEUDNo
parser = argparse.ArgumentParser(description='HTTP server to infer angles from images')
parser.add_argument('--port', type=int, required=False, default=8090, help='HTTP Port')
parser.add_argument('--model', type=str, required=False, default='save/model-0', help='Trained driver model')
parser.add_argument('--gpu', type=int, required=False, default=0, help='GPU number (-1) for CPU')
parser.add_argument('--top_crop', type=int, required=False, default=126, help='Top crop to avoid horizon')
parser.add_argument('--bottom_crop', type=int, required=False, default=226, help='Bottom crop to avoid front of car')
args = parser.parse_args()

def init_tensorflow_model(gpu, model_path):
    # Set enviroment variable to set the GPU to use
    if gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        print('Set tensorflow on CPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Build model and get references to placeholders
    model_in, model_out, labels_in, model_drop = model.build_graph_placeholder()

    # Load tensorflow model
    print("Loading model: %s" % model_path)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    return sess, model_in, model_out, labels_in, model_drop

def pre_proc_img(image, crop_start=126, crop_end=226):
    image = scipy.misc.imresize(np.array(image)[crop_start:crop_end], [66, 200]) / 255.0
    return image


# Initialize tensorflow
sess, model_in, model_out, labels_in, model_drop = init_tensorflow_model(args.gpu, args.model)

# Add app to use flask
app = Flask(__name__)

# This will fire when you access from the web browser the address 127.0.0.1:8090:/
@app.route('/', methods=['GET'])
def test():
    return jsonify({'message' : 'It works!'})


# Service that will return an angle given an image
# From matlab this would be
# webread('127.0.0.1:8090/angle_from_file);
@app.route('/angle_from_file', methods=['POST'])
def get_angle_from_file():
    # Get image file from json request
    imagefile = request.files['file']
    print (type(imagefile))

    # Convert file to a image (This way we don't need to save the image to disk)
    image_bytes = imagefile.stream.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = pil_image.convert('RGB')

    # Convert PIL image to numpy array
    img_array = np.asarray(pil_image)

    # Do some image processing
    img_array = pre_proc_img(img_array)

    # Get steering angle from tensorflow model (Also convert from rad to degree)
    degrees = sess.run(model_out, feed_dict={model_in: [img_array], model_drop: 1.0})[0][0]

    return jsonify(output=float(degrees))


# Service that will return an angle given some data
# From matlab this would be
# webread('127.0.0.1:8090/angle_from_file);
@app.route('/angle_from_data', methods=['POST'])
def get_angle_from_data():
    # Get input from json
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    return jsonify({'angle': '0.3'})

if __name__ == '__main__':
    app.run(debug=True, port=args.port)