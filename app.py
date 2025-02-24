import streamlit as st
import numpy as np
from PIL import Image
import time
from rtxdataset import RTXDataset

# Dataset Configurations
RTX_DATASET_CONFIGS = {
    "TacoPlay": {
        # "gcs_url": "gs://gresearch/robotics/taco_play/0.1.0/",
        "local_data_path": "/mnt/nfs/datasets/selma/downloaded_datasets/taco_play/0.1.0",
        "text_key": "steps/observation/natural_language_instruction",
        "image_key": "steps/observation/rgb_gripper",
        "is_first_key": "steps/is_first",
        "is_last_key": "steps/is_last",
    },
    "RT-1": {
        # "gcs_url": "gs://gresearch/robotics/fractal20220817_data/0.1.0/",
        "local_data_path": "/mnt/nfs/datasets/selma/downloaded_datasets/rt-1/0.1.0",
        "text_key": "steps/observation/natural_language_instruction",
        "image_key": "steps/observation/image",
        "is_first_key": "steps/is_first",
        "is_last_key": "steps/is_terminal",
    },
    "Bridge": {
        # "gcs_url": "gs://gresearch/robotics/bridge/0.1.0/",
        "local_data_path": "/mnt/nfs/datasets/selma/downloaded_datasets/bridge/0.1.0",
        "text_key": "steps/observation/natural_language_instruction",
        "image_key": "steps/observation/image",
        "is_first_key": "steps/is_first",
        "is_last_key": "steps/is_last",
    },
}

@st.cache_resource
def load_dataset(name):
    """Load RTXDataset only when selected to optimize memory usage."""
    config = RTX_DATASET_CONFIGS[name]
    return RTXDataset(
        name=name,
        # gcs_url=config["gcs_url"],
        local_data_path=config["local_data_path"],
        text_key=config["text_key"],
        image_key=config["image_key"],
        is_first_key=config["is_first_key"],
        is_last_key=config["is_last_key"]
    )

def display_episode(episode_texts, episode_images, delay=0.15):
    """Displays text and images sequentially for an episode."""
    text_placeholder = st.empty()
    image_placeholder = st.empty()
    
    for step, (text, image) in enumerate(zip(episode_texts, episode_images)):
        text_placeholder.text(f"Step {step + 1}: {text}")
        image_placeholder.image(Image.fromarray(np.uint8(image)), use_container_width=True)
        time.sleep(delay)

# Streamlit UI
st.title("Google Robotics Dataset Viewer")

# Dataset Selection
selected_dataset_name = st.selectbox("Choose a Dataset:", list(RTX_DATASET_CONFIGS.keys()))

if selected_dataset_name:
    selected_dataset = load_dataset(selected_dataset_name)

    # Choose Lookup Method
    st.subheader("Find an Episode")
    lookup_mode = st.radio("Select how to find an episode:", ["Dropdown", "Search by Episode Number", "Search by Text"])

    if lookup_mode == "Dropdown":
        # Select a TFRecord file
        tfrecord_files = selected_dataset.get_record_files()
        selected_tfrecord = st.selectbox("Choose a TFRecord file:", tfrecord_files)

        if selected_tfrecord:
            episodes = selected_dataset.parse_tfrecord(selected_tfrecord)
            st.write(f"Total Episodes Found: {len(episodes)}")

            # Dropdown for episode selection
            episode_choices = [f"Episode {i + 1}" for i in range(len(episodes))]
            selected_episode = st.selectbox("Select an episode", episode_choices)

            # Get episode index
            episode_idx = episode_choices.index(selected_episode)
            episode_texts, episode_images = episodes[episode_idx]

            st.subheader(f"Episode {episode_idx + 1} / Total Episodes: {len(episodes)}")
            display_episode(episode_texts, episode_images)

    elif lookup_mode == "Search by Episode Number":
        # **Search for an Episode by Global Index**
        total_episodes = len(selected_dataset._episode_index_map)
        episode_idx = st.number_input(
            "Enter Episode Number:", min_value=1, max_value=total_episodes, step=1, value=1
        ) - 1

        if 0 <= episode_idx < total_episodes:
            episode_texts, episode_images = selected_dataset.get_episode_by_index(episode_idx)
            st.subheader(f"Episode {episode_idx + 1} / Total Episodes: {total_episodes}")
            display_episode(episode_texts, episode_images)
        else:
            st.error("Invalid Episode Number. Please enter a valid episode index.")

    else:  # "Search by Text"
        # **Search for Episodes by Text**
        search_query = st.text_input("Enter text to search in episodes:")
        
        if search_query:
            matching_episodes = selected_dataset.search_episodes_by_text(search_query)

            if matching_episodes:
                st.write(f"Found {len(matching_episodes)} episode(s) containing '{search_query}':")
                
                # Dropdown to select from matching episodes
                episode_options = [f"Episode {idx + 1}" for idx, _ in matching_episodes]
                selected_match = st.selectbox("Select an episode from search results:", episode_options)

                if selected_match:
                    match_idx = episode_options.index(selected_match)
                    episode_idx, (episode_texts, episode_images) = matching_episodes[match_idx]
                    
                    st.subheader(f"Episode {episode_idx + 1} / Total Episodes: {len(selected_dataset._episode_index_map)}")
                    display_episode(episode_texts, episode_images)
            else:
                st.warning(f"No episodes found containing '{search_query}'.")