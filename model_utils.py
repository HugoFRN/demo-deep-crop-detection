import glob
import os
import pickle
from keras.models import Model
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util
from keras.layers import interfaces
from keras.engine.topology import InputSpec
from keras.utils import conv_utils
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.engine import Layer
from tensorflow.python.lib.io import file_io


try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot
except ImportError:
    # Fall back on pydot if necessary.
    try:
        import pydot
    except ImportError:
        pydot = None


def io_write_from_temp(temp_pth, dest_pth):
    """
    Used to save any file with tensorflow lib io (gcloud compatibility)
    We first save it to temp path, then load it save it to dest path and delete temp file
    Args:
        temp_pth:
        dest_pth:

    Returns:

    """
    with file_io.FileIO(temp_pth, mode='r') as input_f:
        with file_io.FileIO(dest_pth, mode='w+') as output_f:
            output_f.write(input_f.read())
            file_io.delete_file(temp_pth)


class Identity(Layer):
    """
    Identity layer (with a name)
    """

    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Identity,
              self).build(input_shape)  # Be sure to call this somewhere!

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        return x


def resize_images_bilinear(x, height_factor, width_factor, data_format):
    """ Bilinear Resizes the images contained in a 4D tensor.
    # Arguments
        x: Tensor or variable to resize.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: One of `"channels_first"`, `"channels_last"`.
    # Returns
        A tensor.
    # Raises
        ValueError: if `data_format` is neither
            `channels_last` or `channels_first`.
    """
    if data_format == 'channels_first':
        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[2:]
        new_shape *= tf.constant(
            np.array([height_factor, width_factor]).astype('int32'))
        x = K.permute_dimensions(x, [0, 2, 3, 1])
        x = tf.image.resize_bilinear(x, new_shape)
        x = K.permute_dimensions(x, [0, 3, 1, 2])
        x.set_shape((None, None, original_shape[2] * height_factor
        if original_shape[2] is not None else None,
                     original_shape[3] * width_factor
                     if original_shape[3] is not None else None))
        return x
    elif data_format == 'channels_last':
        original_shape = K.int_shape(x)
        new_shape = tf.shape(x)[1:3]
        new_shape *= tf.constant(
            np.array([height_factor, width_factor]).astype('int32'))
        x = tf.image.resize_bilinear(x, new_shape)
        x.set_shape((None, original_shape[1] * height_factor
        if original_shape[1] is not None else None, original_shape[2] * width_factor
                     if original_shape[2] is not None else None, None))
        return x
    else:
        raise ValueError('Invalid data_format:', data_format)


class BilinearUpSampling2D(Layer):
    """Bilinear Upsampling layer for 2D inputs.
    Repeats the rows and columns of the data
    by size[0] and size[1] respectively.
    # Arguments
        size: int, or tuple of 0627_1357_unet_unet_elubn__ship_oneatlas_sigmoid_jaccard_sigmoid_crop_nadam integers.
            The upsampling factors for rows and columns.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`
    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, upsampled_rows, upsampled_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, upsampled_rows, upsampled_cols)`
    """

    @interfaces.legacy_upsampling2d_support
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[
                2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[
                3] if input_shape[3] is not None else None
            return input_shape[0], input_shape[1], height, width
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[
                1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[
                2] if input_shape[2] is not None else None
            return input_shape[0], height, width, input_shape[3]

    def call(self, inputs, **kwargs):
        return resize_images_bilinear(inputs, self.size[0], self.size[1],
                                         self.data_format)

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


custom_objects = {
    "BilinearUpSampling2D": BilinearUpSampling2D,
    "Identity": Identity
}


def load_json_and_weights(path_to_json, path_to_weights, custom_objects=None):
    """
    Define a non-compiled Keras models from json and weight files
    :param path_to_json:
    :param path_to_weights:
    :param custom_objects: Custom layers necessary to load models
    :return: An uncompiled models loaded from the json and weights files
    """
    from keras.models import model_from_json

    with open(path_to_json, 'r') as f:
        json_string = f.read()
        model = model_from_json(json_string, custom_objects=custom_objects)
        model.load_weights(path_to_weights)
        f.close()

    return model


def plot_model_if_pydot(model, file):
    """
    Plot models with graphivz library if pydot and graphviz are installed else do nothing
    :param model:
    :param file:
    :return:
    """
    try:
        # pydot-ng is a fork of pydot that is better maintained.
        import pydot_ng as pydot
    except ImportError:
        # Fall back on pydot if necessary.
        try:
            import pydot
        except ImportError:
            pydot = None

    from keras.utils.vis_utils import plot_model
    if pydot is not None and pydot.find_graphviz():
        plot_model(
            model, to_file=file, show_shapes=True, show_layer_names=True)
    else:
        print("Pydot is None or not graphviz")


def save_as_tensorflow_inference(path_to_model):
    """
    Converts keras model to .pb
    Args:
        path_to_model: Directory containing weights & config of keras model

    Returns:

    """
    K.clear_session()
    K.set_learning_phase(0)
    model, _ = load_keras_model(path_to_model, remove_cropping=True)
    pred = [None]
    pred_node_names = [None]
    pred_node_names[0] = 'output_node'
    pred[0] = tf.identity(model.output[0], name=pred_node_names[0])

    sess = K.get_session()

    bn = model.name

    constant_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), pred_node_names)

    graph_io.write_graph(constant_graph, "./", bn + '.pb', as_text=False)
    io_write_from_temp(
        os.path.join("./", bn + '.pb'), os.path.join(path_to_model,
                                                     bn + ".pb"))

    print('saved the constant graph (ready for inference) at: ',
          os.path.join(path_to_model, bn + '.pb'))


def load_tensorflow_inference(path_to_model, model_type):
    """
    Loads .pb inference model
    Args:
        path_to_model: Directory containing .pb

    Returns:

    """
    graph_filename = glob.glob(os.path.join(path_to_model, '*.pb'))

    if len(graph_filename) == 0:
        # Not generated. Try
        save_as_tensorflow_inference(path_to_model)
        graph_filename = glob.glob(os.path.join(path_to_model, '*.pb'))

    assert len(
        graph_filename
    ) > 0, "No .pb in dir %s and no hdf5/json combo to generate from" % path_to_model

    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(graph_filename[0], "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None)

    # We can verify that we can access the list of operations in the graph
    input_tensor_name = None
    output_tensor_name = None
    for i, op in enumerate(graph.get_operations()):
        if i == 0:
            input_tensor_name = op.name
    assert input_tensor_name is not None, "input_tensor is None"

    input_tensor = graph.get_tensor_by_name('%s:0' % input_tensor_name)
    if output_tensor_name is None:
        output_tensor = graph.get_tensor_by_name('prefix/output_node:0')
    else:
        output_tensor = graph.get_tensor_by_name('%s:0' % output_tensor_name)
    return graph, input_tensor, output_tensor


def load_keras_model(path_to_model, remove_cropping=False):
    """
    Load keras from .json config and .hdf5 wiehgts
    Args:
        path_to_model: Directorying containing files
        remove_cropping: remove cropping2d layer if present
        model_name: For multi-model, load the correct wheights
    Returns:

    """
    model_config = glob.glob(os.path.join(path_to_model, '*model_config.json'))
    final_weights = glob.glob(os.path.join(path_to_model, '*final.hdf5'))
    ckpts_weights = glob.glob(os.path.join(path_to_model, '*checkpoint.hdf5'))
    train_data = glob.glob(os.path.join(path_to_model, '*_train_data.pickle'))

    assert len(model_config) > 0
    assert len(final_weights) > 0 or len(ckpts_weights) > 0
    assert len(train_data) > 0

    json_file = model_config[0]
    train_data = train_data[0]
    train_data = pickle.load(open(train_data, 'rb'))

    if len(final_weights) > 0:
        weight_file = final_weights[0]
    else:
        weight_file = ckpts_weights[0]

    model = load_json_and_weights(
        json_file, weight_file, custom_objects=custom_objects)
    # Remove the last cropping layer if present
    if remove_cropping and model.layers[-1].__class__.__name__ == 'Cropping2D':
        model = Model(
            inputs=model.input,
            outputs=model.layers[-2].output,
            name=os.path.splitext(os.path.basename(weight_file))[0])

    else:
        model = Model(
            inputs=model.input,
            outputs=model.output,
            name=os.path.splitext(os.path.basename(weight_file))[0])

    return model, train_data
