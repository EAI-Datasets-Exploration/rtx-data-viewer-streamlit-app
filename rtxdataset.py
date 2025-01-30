import tensorflow as tf
import gcsfs

class RTXDataset:
    def __init__(self, name, gcs_url, text_key, image_key, is_first_key, is_last_key):
        self.name = name
        self.gcs_url = gcs_url
        self.text_key = text_key
        self.image_key = image_key
        self.is_first_key = is_first_key
        self.is_last_key = is_last_key

        self.image_h = 224
        self.image_w = 224

        self.record_files = self.create_tfrecord_files_list()

    def __repr__(self):
        return f"Dataset(name={self.name}, gcs_url={self.gcs_url}, text_key={self.text_key}, image_key={self.image_key})"
    
    def create_tfrecord_files_list(self):
        fs = gcsfs.GCSFileSystem()
        files = fs.ls(self.gcs_url)
        return [f"gs://{file}" for file in files if "tfrecord" in file]

    def get_record_files(self):
        return self.record_files

    def parse_tfrecord(self, record_path):
        raw_dataset = tf.data.TFRecordDataset(record_path)

        episodes = []
        current_texts = []
        current_images = []

        for raw_record in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            # Detect episode boundaries
            is_first = bool(example.features.feature[self.is_first_key].int64_list.value[0])
            is_last = bool(example.features.feature[self.is_last_key].int64_list.value[0])

            if is_first and current_texts:  # If a new episode starts, save the previous one
                episodes.append((current_texts, current_images))
                current_texts, current_images = [], []

            # Extract and decode texts
            current_texts.extend(text.decode("utf-8") for text in example.features.feature[self.text_key].bytes_list.value)
            
            # Extract and decode images
            current_images.extend([
                tf.image.resize(tf.image.decode_jpeg(img, channels=3), [self.image_h, self.image_w]).numpy()
                for img in example.features.feature[self.image_key].bytes_list.value
            ])

            if is_last:  # If the episode ends, store it and reset
                episodes.append((current_texts, current_images))
                current_texts, current_images = [], []

        return episodes