import streamlit as st
import tensorflow as tf

from fancy_pca import fancy_pca

if "min_alpha" not in st.session_state:
    st.session_state.min_alpha = -3.0

if "max_alpha" not in st.session_state:
    st.session_state.max_alpha = 3.0

if "alpha_1" not in st.session_state:
    st.session_state.alpha_1 = float(tf.random.normal((1,), mean=0, stddev=1))

if "alpha_2" not in st.session_state:
    st.session_state.alpha_2 = float(tf.random.normal((1,), mean=0, stddev=1))

if "alpha_3" not in st.session_state:
    st.session_state.alpha_3 = float(tf.random.normal((1,), mean=0, stddev=1))

if "image" not in st.session_state:
    image = tf.io.read_file("./images/bird.jpg")
    image = tf.image.decode_jpeg(image, channels=3)
    st.session_state.image = image


def augment_image():
    pca_image = fancy_pca(
        st.session_state.image,
        [st.session_state.alpha_1, st.session_state.alpha_2, st.session_state.alpha_3],
    )
    st.session_state.pca_image = pca_image


if "pca_image" not in st.session_state:
    augment_image()


st.title("PCA Colour Augmentation")
st.header("Demo")

col1, col2 = st.beta_columns(2)

col1.subheader("Original")
col1.image(st.session_state.image.numpy(), use_column_width=True)

col2.subheader("PCA Colour Augmented")
col2.image(st.session_state.pca_image.numpy(), use_column_width=True)

col1, col2, col3 = st.beta_columns(3)
col1.slider(
    "Alpha 1",
    min_value=st.session_state.min_alpha,
    max_value=st.session_state.max_alpha,
    step=0.01,
    key="alpha_1",
    on_change=augment_image,
)
col2.slider(
    "Alpha 2",
    min_value=st.session_state.min_alpha,
    max_value=st.session_state.max_alpha,
    step=0.01,
    key="alpha_2",
    on_change=augment_image,
)
col3.slider(
    "Alpha 3",
    min_value=st.session_state.min_alpha,
    max_value=st.session_state.max_alpha,
    step=0.01,
    key="alpha_3",
    on_change=augment_image,
)
