import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import xception
import matplotlib.pyplot as plt
from scipy import ndimage
from IPython.display import Image
from tensorflow import keras
from tensorflow.keras import layers

# def get_img_array(img_path, size=(299, 299)):
#     # `img` is a PIL image of size 299x299
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
#     # `array` is a float32 Numpy array of shape (299, 299, 3)
#     array = tf.keras.preprocessing.image.img_to_array(img)
#     # We add a dimension to transform our array into a "batch"
#     # of size (1, 299, 299, 3)
#     array = np.expand_dims(array, axis=0)
#     return array


# def get_gradients(img_input, model, top_pred_idx, img_size=None):
#     """Computes the gradients of outputs w.r.t input image.

#     Args:
#         img_input: 4D image tensor
#         top_pred_idx: Predicted label for the input image

#     Returns:
#         Gradients of the predictions w.r.t img_input
#     """
#     images = tf.cast(img_input, tf.float32)

#     with tf.GradientTape() as tape:
#         tape.watch(images)
#         preds = model(images)
# #         print(np.shape(images))
# #         print(np.shape(preds))
#         top_class = preds[:, top_pred_idx]

#     grads = tape.gradient(top_class, images)
#     return grads


def get_integrated_gradients(img_input, model, top_pred_idx, baseline=None, num_steps=50, img_size=None):
    """Computes Integrated Gradients for a predicted label.

    Args:
        img_input (ndarray): Original image
        top_pred_idx: Predicted label for the input image
        baseline (ndarray): The baseline image to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.

    Returns:
        Integrated gradients w.r.t input image
    """
    # If baseline is not provided, start with a black image
    # having same size as the input image.
    if baseline is None:
#         baseline = np.zeros(img_size).astype(np.float32)
        baseline = np.zeros(img_size).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    # 1. Do interpolation.
    img_input = img_input.astype(np.float32)
    interpolated_image = [
        baseline + (step / num_steps) * (img_input - baseline)
        for step in range(num_steps + 1)
    ]
    interpolated_image = np.array(interpolated_image).astype(np.float32)

    # 2. Preprocess the interpolated images
    interpolated_image = xception.preprocess_input(interpolated_image)

    # 3. Get the gradients
    grads = []
    for i, img in enumerate(interpolated_image):
        img = tf.expand_dims(img, axis=0)
        grad = get_gradients(img, model, top_pred_idx=top_pred_idx, img_size=img_size)
        grads.append(grad[0])
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)

    # 4. Approximate the integral using the trapezoidal rule
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)

    # 5. Calculate integrated gradients and return
    integrated_grads = (img_input - baseline) * avg_grads
    return integrated_grads


def random_baseline_integrated_gradients(
    img_input, model, top_pred_idx, num_steps=50, num_runs=2, img_size=None
):
    """Generates a number of random baseline images.

    Args:
        img_input (ndarray): 3D image
        top_pred_idx: Predicted label for the input image
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.
        num_runs: number of baseline images to generate

    Returns:
        Averaged integrated gradients for `num_runs` baseline images
    """
    # 1. List to keep track of Integrated Gradients (IG) for all the images
    integrated_grads = []

    # 2. Get the integrated gradients for all the baselines
    for run in range(num_runs):
#         baseline = np.random.random(img_size) * 255
        baseline = np.random.random(img_size) * 255
        igrads = get_integrated_gradients(
            img_input=img_input,
            model = model,
            top_pred_idx=top_pred_idx,
            baseline=baseline,
            num_steps=num_steps,
            img_size =img_size,
        )
        integrated_grads.append(igrads)

    # 3. Return the average integrated gradients for the image
    integrated_grads = tf.convert_to_tensor(integrated_grads)
    return tf.reduce_mean(integrated_grads, axis=0)