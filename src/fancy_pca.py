from typing import List

import tensorflow as tf


def cov(m: tf.Tensor, rowvar: bool = True, bias: bool = False) -> tf.Tensor:
    """Estimate a covariance matrix.
    Mimics the behaviour of `np.cov(m)`

    Parameters
    ----------
    m : tf.Tensor
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of m represents a variable, and each column a single
        observation of all those variables.

    rowvar: bool
        If rowvar is True (default), then each row represents a variable, with
        observations in the columns. Otherwise, the relationship is transposed:
        each column represents a variable, while the rows contain observations.

    Returns
    -------
    tf.Tensor
        The covariance matrix of the variables.
    """
    if rowvar:
        m = m - tf.reduce_mean(m, axis=1, keepdims=True)
        n = tf.shape(m)[1] if bias else tf.shape(m)[1] - 1
        covariance = tf.matmul(m, tf.transpose(m)) / tf.cast(n, tf.float32)
        return covariance
    else:
        m = m - tf.reduce_mean(m, axis=0, keepdims=True)
        n = tf.shape(m)[0] if bias else tf.shape(m)[0] - 1
        covariance = tf.matmul(tf.transpose(m), m) / tf.cast(n, tf.float32)
        return covariance


def fancy_pca(img: tf.Tensor, alphas: List[float]) -> tf.Tensor:

    """PCA Colour Augmentation as described in AlexNet paper.

    Parameters
    ----------
    img : tf.Tensor
        3-dimensional Tensor of shape (h, w, 3)
    alphas: List[float]
        The 3 random normal alpha values

    Returns
    -------
    tf.Tensor
        3-dimensional Tensor corresponding to the image with some noise added
        along the principal components of the colour channels.
    """
    rows, columns, _ = img.shape
    img = tf.reshape(img, (rows * columns, 3))
    img = tf.cast(img, "float32")

    mean = tf.reduce_mean(img, axis=0)
    std = tf.math.reduce_std(img, axis=0)
    img -= mean
    img /= std

    covariance = cov(img, rowvar=False, bias=True)
    lambdas, p, _ = tf.linalg.svd(covariance)

    alphas = tf.constant(alphas)
    delta = tf.tensordot(p, alphas * lambdas, axes=1)

    img = img + delta
    img = img * std + mean
    img = tf.clip_by_value(img, 0, 255)
    img = tf.cast(img, dtype=tf.uint8)

    img = tf.reshape(img, (rows, columns, 3))
    return img
