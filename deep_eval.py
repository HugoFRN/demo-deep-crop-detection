from eolearn.core import FeatureType, EOTask
import numpy as np
import cv2
import os
import glob
import pickle
import tensorflow as tf
from model_utils import load_tensorflow_inference


def class_matrix_to_bgr_mask(matrix, bgr_class_dict, in_place=False):
    """
    Class matrix to bgr mask
    Args:
        matrix(np.ndarray):  class matrix
        bgr_class_dict(): List of class colors [[b,g,r]] or np.array (is recasted anyway)
        in_place(bool): modify element itself

    Returns:
        bgr mask
    """
    bgr_class_dict = np.asarray(bgr_class_dict, dtype=np.uint8)
    if in_place:
        matrix = np.uint8(bgr_class_dict[matrix])
        return matrix
    else:
        mask = np.uint8(bgr_class_dict[matrix])
        return mask


def softmax_to_class_matrix(softmax_output,
                            classes_to_ignore=np.asarray([]),
                            threshold=None):
    """
    Softmax to class matrix with threshold or argmax
    Args:
        softmax_output: Output from softmax
        classes_to_ignore: numpy ndarray containing indexes of classes to force at 0
        threshold: Threshold all activations < threshold to zero
                   Exclude class zero (apply threshold as if binary classification and takes argmax without class zero)
    Returns:
        Class matrix

    """
    classes_to_ignore = np.asarray(classes_to_ignore, dtype=np.int8)
    if classes_to_ignore.shape[0] > 0:
        softmax_output[:, :, classes_to_ignore] = 0
    if threshold is not None:
        if softmax_output.dtype == np.uint8 and threshold < 1.:
            threshold = int(255 * threshold)
        # We're thresholding so we're forcing to ignore class 0 before argmaxing
        softmax_output[:, :, 0] = 0
        # Compute argmax
        class_matrix = np.argmax(softmax_output, axis=-1).astype(np.uint8)
        # Now sets to background pixels where all classes are below threshold
        class_matrix[np.all(softmax_output <= threshold, axis=-1)] = 0
    else:
        class_matrix = np.argmax(softmax_output, axis=-1)
    return class_matrix


def bgr_mask_to_class_matrix(mask, bgr_class_list):
    h, w = mask.shape[:2]
    y = np.zeros((h, w), dtype=np.uint8)
    for i, pix in enumerate(bgr_class_list):
        y_c = np.all(mask == np.asarray(pix), axis=-1)
        y[y_c] = i
    return y


class DeepEval(EOTask):
    class Config:
        def __init__(self):
            self.to_hsv = False
            self.pooling_factor = None
            self.output_scale = None
            self.mean = None
            self.std = None
            self.class_dict = None
            self.class_labels = None
            self.num_classes = None
            self.encoding = 16
            self.n_bands = 13

    """
    Use our Deep Learning model to run inference.
    """

    @staticmethod
    def image_to_tensor(encoding, n_bands, to_hsv):
        def f(x):
            if x is not None:
                x = x.astype(np.float32)
                if x.shape[2] == 3 or n_bands <= 3:
                    # Scale image to tensor [0,1]
                    x /= (2 ** encoding - 1.)
                    if to_hsv:
                        # Go to HSV Color Space assuming a [0,1] image
                        x = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
                        # Scale Hue to [0,1]
                        x[:, :, 0] = x[:, :, 0] / 360.
                else:
                    # Scale image to tensor
                    x /= (2 ** encoding - 1.)
            return x

        return f

    def to_tensor(self, config, x):
        x = self.image_to_tensor(config.encoding, config.n_bands, config.to_hsv)(x)
        x -= config.mean
        x /= config.std
        return x


    @staticmethod
    def load_data(config, path_to_model):
        train_data = glob.glob(
            os.path.join(path_to_model, '*_train_data.pickle'))
        assert len(train_data) > 0
        train_data = train_data[0]
        ts_data = pickle.load(open(train_data, 'rb'), encoding='latin1')
        config.pooling_factor = int(ts_data['pooling_size'])
        config.output_scale = ts_data.get('output_scale') or 1
        config.mean = ts_data['mean'] if 'mean' in ts_data else np.zeros(
            (1, 1, config.n_bands))
        config.std = ts_data['std'] if 'std' in ts_data else np.ones(
            (1, 1, config.n_bands))
        config.class_dict = np.asarray(ts_data['class_dict'])
        config.class_labels = ts_data['class_labels']
        config.num_classes = len(config.class_labels)

        # Check if legacy imread was used.
        if np.all(config.mean > 1.) and config.encoding == 8:
            config.mean /= 255.
        if np.all(config.std > 1.) and config.encoding == 8:
            config.std /= 255.
        return config

    def __init__(self, feature, model_dir, gpu_mode=True):
        '''
        feature: EO-Patch feature to use for the inference
        model_dir: Path to the model to load
        gpu_mode: Whether to use the GPU for the inference
        '''
        self.feature = self._parse_features(feature, new_names=True,
                                            default_feature_type=FeatureType.DATA,
                                            rename_function='{}_CROP_MASK'.format)
        self.model_dir = model_dir
        self.GPU_MODE = gpu_mode
        if not self.GPU_MODE:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.graph, self.input_tensor, self.output_tensor = load_tensorflow_inference(self.model_dir, 'RefineNet')
        conf = self.Config()
        self.session = tf.Session(graph=self.graph)
        self.config = self.load_data(conf, self.model_dir)

    def execute(self, eopatch):
        if eopatch is None:
            return None
        else:
            eval = []
            eval_one_hot = []
            eval_proba = []
            feature_type, feature_name, new_feature_name = next(self.feature(eopatch))
            for i, im in enumerate(eopatch[feature_type][feature_name]):

                im = im.astype(np.uint16)
                pooling_factor = self.config.pooling_factor

                padding = 4 * pooling_factor
                padded_img_array = np.pad(
                    im, ((padding, padding), (padding, padding),
                         (0, 0)),
                    mode='reflect')

                h, w, d = padded_img_array.shape
                # Ensure that we test a multiple of pooling_factor for spatial size
                w_16 = int(
                    np.ceil(float(w) / pooling_factor) * pooling_factor)
                h_16 = int(
                    np.ceil(float(h) / pooling_factor) * pooling_factor)
                pad_l = int((w_16 - w) // 2)
                pad_r = int(w_16 - w - pad_l)
                pad_t = int((h_16 - h) // 2)
                pad_b = int(h_16 - h - pad_t)
                pad_img = np.pad(
                    padded_img_array, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                    mode='reflect')
                pad_img = self.to_tensor(self.config, pad_img)
                img = pad_img.reshape((1, h_16, w_16, d))
                try:
                    out = self.session.run(
                        self.output_tensor, feed_dict={
                            self.input_tensor: img
                        })

                    # Crop output as necessary
                    if len(out.shape) == 4:
                        out = out[0, pad_t:(h_16 - pad_b), pad_l:(w_16 - pad_r), :]
                    else:
                        out = out[pad_t:(h_16 - pad_b), pad_l:(w_16 - pad_r), :]
                    # Clip probabilities to [0-255]
                    out = out[padding:-padding, padding:-padding]
                    out *= 255.
                    out = np.clip(out, 0., 255.)
                    out = out.astype(np.uint8)

                    mask_one_hot = softmax_to_class_matrix(out)
                    mask = class_matrix_to_bgr_mask(mask_one_hot, self.config.class_dict, in_place=False)
                    if i == 0:
                        eval = np.array([mask])
                        eval_one_hot = np.array([mask_one_hot])
                        eval_proba = np.array([out])
                    else:
                        eval = np.append(eval, np.array([mask]), axis=0)
                        eval_one_hot = np.append(eval_one_hot, np.array([mask_one_hot]), axis=0)
                        eval_proba = np.append(eval_proba, np.array([out]), axis=0)
                except Exception as exception:
                    print(exception)
            eopatch.mask[new_feature_name] = eval
            eopatch.mask[new_feature_name + '_CLASS'] = eval_one_hot.reshape(eval_one_hot.shape + (1,))
            eopatch.mask[new_feature_name + '_PROBA'] = eval_proba
            eopatch.meta_info['class_rgb'] = self.config.class_dict
            eopatch.meta_info['class_labels'] = self.config.class_labels
            return eopatch