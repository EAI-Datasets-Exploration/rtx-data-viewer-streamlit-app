from rtxdataset import RTXDataset

dataset = RTXDataset(
    name="TacoPlay",
    local_data_path="/mnt/nfs/datasets/selma/downloaded_datasets/taco_play/0.1.0/",
    text_key="steps/observation/natural_language_instruction",
    image_key="steps/observation/rgb_gripper",
    is_first_key="steps/is_first",
    is_last_key="steps/is_last"
)

# Force database rebuild
dataset.save_episode_index_sqlite()