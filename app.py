import streamlit as st
import tensorflow as tf
import gcsfs
import numpy as np
from PIL import Image
import time
from rtxdataset import RTXDataset

RTX_DATASET_CONFIGS = {
    "TacoPlay": {
        "gcs_url": "gs://gresearch/robotics/taco_play/0.1.0/",
        "text_key": "steps/observation/natural_language_instruction",
        "image_key": "steps/observation/rgb_gripper",
        "is_first_key": "steps/is_first",
        "is_last_key": "steps/is_last",
    },
    "RT-1": {
        "gcs_url": "gs://gresearch/robotics/fractal20220817_data/0.1.0/",
        "text_key": "steps/observation/natural_language_instruction",
        "image_key": "steps/observation/image",
        "is_first_key": "steps/is_first",
        "is_last_key": "steps/is_terminal",
    },
    "Bridge": {
        "gcs_url": "gs://gresearch/robotics/bridge/0.1.0/",
        "text_key": "steps/observation/natural_language_instruction",
        "image_key": "steps/observation/image",
        "is_first_key": "steps/is_first",
        "is_last_key": "steps/is_last",
    },
}

# Instantiate RTXDataset objects based on the configurations
datasets = {}

for name, config in RTX_DATASET_CONFIGS.items():
    datasets[name] = RTXDataset(
        name=name,
        gcs_url=config["gcs_url"],
        text_key=config["text_key"],
        image_key=config["image_key"],
        is_first_key=config["is_first_key"],
        is_last_key=config["is_last_key"]
    )

@st.cache_resource
def list_tfrecord_files(dataset):
    return dataset.get_record_files()

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

# Streamlit selectbox for selecting a dataset by name
selected_dataset_name = st.selectbox("Choose a Dataset:", list(datasets.keys()))

# Retrieve the selected dataset object
selected_dataset = datasets[selected_dataset_name]

# Select a TFRecord file
tfrecord_files = selected_dataset.get_record_files()
selected_tfrecord = st.selectbox("Choose a TFRecord file:", tfrecord_files)

# Display data
if selected_tfrecord:
    episodes = selected_dataset.parse_tfrecord(selected_tfrecord)
    
    st.write(f"Total Episodes Found: {len(episodes)}")

    # Dropdown to select episode
    episode_choices = [f"Episode {i + 1}" for i in range(len(episodes))]
    selected_episode = st.selectbox("Select an episode", episode_choices)

    # Map selected episode to its index
    episode_idx = episode_choices.index(selected_episode)

    # Display selected episode
    episode_texts, episode_images = episodes[episode_idx]
    st.subheader(f"{selected_episode} / Total Episodes: {len(episodes)}")
    display_episode(episode_texts, episode_images)

