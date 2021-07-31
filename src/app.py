import streamlit as st
import tensorflow as tf

from fancy_pca import fancy_pca

st.title("PCA Colour Augmentation")
st.header("Demo")

alpha_std = st.slider(
    "Alpha Standard Deviation", min_value=0.0, max_value=1.0, step=0.01, value=0.1
)
alphas = tf.random.normal((3,), mean=0, stddev=alpha_std)
min_alpha = alpha_std * 3 * -1
max_alpha = alpha_std * 3
alphas = tf.clip_by_value(alphas, min_alpha, max_alpha)
col1, col2, col3 = st.beta_columns(3)
alpha_1 = col1.slider(
    "Alpha 1",
    min_value=min_alpha,
    max_value=max_alpha,
    step=0.01,
    value=float(alphas[0]),
)
alpha_2 = col2.slider(
    "Alpha 2",
    min_value=min_alpha,
    max_value=max_alpha,
    step=0.01,
    value=float(alphas[1]),
)
alpha_3 = col3.slider(
    "Alpha 3",
    min_value=min_alpha,
    max_value=max_alpha,
    step=0.01,
    value=float(alphas[2]),
)

image = tf.io.read_file("./images/bird3.jpg")
image = tf.image.decode_jpeg(image, channels=3)
pca_image = fancy_pca(image, [alpha_1, alpha_2, alpha_3])

col1, col2 = st.beta_columns(2)

col1.subheader("Original")
col1.image(image.numpy(), use_column_width=True)

col2.subheader("PCA Colour Augmented")
col2.image(pca_image.numpy(), use_column_width=True)
