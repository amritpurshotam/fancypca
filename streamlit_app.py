import streamlit as st
import tensorflow as tf
from PIL import Image

from fancy_pca import fancy_pca

st.title("PCA Colour Augmentation")
user_image = st.file_uploader("Try your own image...", type=["jpg"])

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


if user_image is not None:
    image = Image.open(user_image)
    image = tf.keras.preprocessing.image.img_to_array(image, dtype="int")
    image = tf.convert_to_tensor(image)
    st.session_state.image = image
    augment_image()


def reset_alphas():
    st.session_state.alpha_1 = 0.0
    st.session_state.alpha_2 = 0.0
    st.session_state.alpha_3 = 0.0
    augment_image()


def randomize_alphas():
    alpha_1, alpha_2, alpha_3 = list(tf.random.normal((3,), mean=0, stddev=1).numpy())
    st.session_state.alpha_1 = alpha_1
    st.session_state.alpha_2 = alpha_2
    st.session_state.alpha_3 = alpha_3
    augment_image()


if "pca_image" not in st.session_state:
    augment_image()

col1, col2 = st.columns(2)

col1.subheader("Original")
col1.image(st.session_state.image.numpy(), use_column_width=True)

col2.subheader("PCA Colour Augmented")
col2.image(st.session_state.pca_image.numpy(), use_column_width=True)

col1, col2, col3 = st.columns(3)
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

col1, col2 = st.columns([0.12, 1])
col1.button("Reset", on_click=reset_alphas)
col2.button("Randomize", on_click=randomize_alphas)

st.header("Explanation")
st.markdown(
    r"""
PCA Colour Augmentation is an image augmentation technique that consists of altering
the intensities in each of the RGB channels.

In short, the principal components of the image are calculated along the RGB colour
channels i.e. each colour channel is treated as an independent feature so an image
with shape $(h, w, 3)$ is reshaped to $(hw, 3)$. These principal components are then
added to the image with magnitudes proportional to their corresponding eigenvalue
times a random variable drawn from a Gaussian with mean zero and standard deviation 0.1.

More rigorously, to each RGB image pixel $I_{xy}=[I_{xy}^R, I_{xy}^G, I_{xy}^B]^T$ we
add

$$
[\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3]
[\alpha_1\lambda_1, \alpha_2\lambda_2, \alpha_3\lambda_3]^T
$$

where $\mathbf{p}_i$ and $\lambda_i$ are the $i$th eigenvector and eigenvalue of
the $3 \times 3$ covariance matrix of RGB pixel values, respectively, and
$\alpha_i$ is the random variable.

The intuitition behind this technique is that it captures an important property
of natural images that object identity is invariant to changes in the intensity
and colour of the illumination. In the AlexNet paper [1] the authors reduced their
top-1 error rate by over 1% with this scheme.
"""
)

st.header("References")
authors = "A. Krizhevsky, I. Sutskever, G. Hinton"
year = 2012
title = "ImageNet Classification with Deep Convolutional Neural Networks"
url = "https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf"  # noqa: B950
booktitle = "Advances in Neural Information Processing Systems"

st.markdown(
    f"""
1. {authors}. [{title}]({url}). In _{booktitle}_. {year}.

2. D. Clode. [A Black-capped Lory eating an apple](https://unsplash.com/photos/7JrLPQXPVCI). On _Unsplash_. 2021
"""
)
