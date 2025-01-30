import streamlit as st
import tensorflow as tf
import gcsfs
import numpy as np
from PIL import Image
import time


class Dataset:
    def __init__(self, name, gcs_url, text_key, image_key):
        self.name = name
        self.gcs_url = gcs_url
        self.text_key = text_key
        self.image_key = image_key

    def __repr__(self):
        return f"Dataset(name={self.name}, gcs_url={self.gcs_url}, text_key={self.text_key}, image_key={self.image_key})"

# Dictionary of datasets for dropdown selection
GCS_DATASETS = {
    "TacoPlay": Dataset(
        name="TacoPlay",
        gcs_url="gs://gresearch/robotics/taco_play/0.1.0/",
        text_key="steps/observation/natural_language_instruction",
        image_key="steps/observation/rgb_gripper"
    ),
    # "RT-1": Dataset(
    #     name="RT-1",
    #     gcs_url="gs://gresearch/robotics/fractal20220817_data/0.1.0/",
    #     text_key="steps/observation/natural_language_instruction",  # Update this key as necessary
    #     image_key="steps/observation/image"  # Update this key as necessary
    # ),
    # "Bridge": Dataset(
    #     name="Bridge",
    #     gcs_url="gs://gresearch/robotics/bridge/0.1.0/",
    #     text_key="steps/observation/natural_language_instruction",  # Update this key as necessary
    #     image_key="steps/observation/image"  # Update this key as necessary
    # ),
}

# Initialize the GCS file system
fs = gcsfs.GCSFileSystem()

@st.cache_resource
def list_tfrecord_files(dataset_path):
    """Lists all TFRecord files for the selected dataset."""
    files = fs.ls(dataset_path)
    return [f"gs://{file}" for file in files if "tfrecord" in file]

def parse_tfrecord(tfrecord_path, text_key="steps/observation/natural_language_instruction", image_key="steps/observation/rgb_gripper"):
    """Extracts and groups text and images by episode boundaries."""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    episodes = []
    current_texts = []
    current_images = []

    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        # Detect episode boundaries
        is_first = bool(example.features.feature["steps/is_first"].int64_list.value[0])
        is_last = bool(example.features.feature["steps/is_last"].int64_list.value[0])

        if is_first and current_texts:  # If a new episode starts, save the previous one
            episodes.append((current_texts, current_images))
            current_texts, current_images = [], []

        # Extract and decode texts
        current_texts.extend(text.decode("utf-8") for text in example.features.feature[text_key].bytes_list.value)
        
        # Extract and decode images
        current_images.extend([
            tf.image.resize(tf.image.decode_jpeg(img, channels=3), [224, 224]).numpy()
            for img in example.features.feature[image_key].bytes_list.value
        ])

        if is_last:  # If the episode ends, store it and reset
            episodes.append((current_texts, current_images))
            current_texts, current_images = [], []

    return episodes

def display_episode(episode_texts, episode_images, delay=0.05):
    """Displays text and images sequentially for an episode."""
    text_placeholder = st.empty()
    image_placeholder = st.empty()
    
    for step, (text, image) in enumerate(zip(episode_texts, episode_images)):
        text_placeholder.text(f"Step {step + 1}: {text}")
        image_placeholder.image(Image.fromarray(np.uint8(image)), use_container_width=True)
        time.sleep(delay)

# Streamlit UI
st.title("Google Robotics Dataset Viewer")

# Select dataset
selected_dataset_name = st.selectbox("Choose a dataset:", list(GCS_DATASETS.keys()))
selected_dataset = GCS_DATASETS[selected_dataset_name]

# Select a TFRecord file
tfrecord_files = list_tfrecord_files(selected_dataset.gcs_url)
selected_tfrecord = st.selectbox("Choose a TFRecord file:", tfrecord_files)

# Display data
if selected_tfrecord:
    episodes = parse_tfrecord(
        selected_tfrecord,
        text_key=selected_dataset.text_key,  # Use the selected dataset's text key
        image_key=selected_dataset.image_key  # Use the selected dataset's image key
    )
    
    st.write(f"Total Episodes Found: {len(episodes)}")

    # Ensure current episode index is within bounds
    if 'current_episode' not in st.session_state:
        st.session_state.current_episode = 0

    # Ensure current episode index is within bounds
    st.session_state.current_episode = max(0, min(st.session_state.current_episode, len(episodes) - 1))

    # Display current episode
    episode_texts, episode_images = episodes[st.session_state.current_episode]
    st.subheader(f"Episode {st.session_state.current_episode + 1}/{len(episodes)}")
    display_episode(episode_texts, episode_images)

    # Navigation buttons
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Previous Episode", disabled=st.session_state.current_episode == 0):
            st.session_state.current_episode -= 1  # Go to the previous episode
            # No rerun required

    with col2:
        if st.button("Next Episode", disabled=st.session_state.current_episode == len(episodes) - 1):
            st.session_state.current_episode += 1  # Go to the next episode
            # No rerun required