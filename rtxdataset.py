import concurrent.futures
import tensorflow as tf
import gcsfs
import sqlite3
import os

class RTXDataset:
    def __init__(self, name, gcs_url, text_key, image_key, is_first_key, is_last_key, db_path="sqlite3_files/"):
        self.name = name
        self.gcs_url = gcs_url
        self.text_key = text_key
        self.image_key = image_key
        self.is_first_key = is_first_key
        self.is_last_key = is_last_key
        self.db_path = db_path + self.name + "_episode_index.db"

        self.image_h = 224
        self.image_w = 224

        self.record_files = self.create_tfrecord_files_list()

        # Try to load from SQLite, otherwise build and cache the index
        if self._episode_index_exists():
            self._episode_index_map = self.load_episode_index_sqlite()
        else:
            self._episode_index_map = self._build_episode_index_parallel()
            self.save_episode_index_sqlite()

    def create_tfrecord_files_list(self):
        """List all TFRecord files from the given GCS bucket."""
        fs = gcsfs.GCSFileSystem()
        files = fs.ls(self.gcs_url)
        return [f"gs://{file}" for file in files if "tfrecord" in file]
    
    def get_record_files(self):
        return self.record_files

    def parse_tfrecord(self, record_path):
        """Parses a single TFRecord file and returns a list of episodes."""
        raw_dataset = tf.data.TFRecordDataset(record_path)
        episodes = []
        current_texts = []
        current_images = []

        for raw_record in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            is_first = bool(example.features.feature[self.is_first_key].int64_list.value[0])
            is_last = bool(example.features.feature[self.is_last_key].int64_list.value[0])

            if is_first and current_texts:
                episodes.append((current_texts, current_images))
                current_texts, current_images = [], []

            current_texts.extend(text.decode("utf-8") for text in example.features.feature[self.text_key].bytes_list.value)
            
            current_images.extend([
                tf.image.resize(tf.image.decode_jpeg(img, channels=3), [self.image_h, self.image_w]).numpy()
                for img in example.features.feature[self.image_key].bytes_list.value
            ])

            if is_last:
                episodes.append((current_texts, current_images))
                current_texts, current_images = [], []

        return episodes

    def _process_file_for_index(self, record_file):
        """Helper function to process a single TFRecord file for episode indexing."""
        episodes = self.parse_tfrecord(record_file)
        return [(record_file, i) for i in range(len(episodes))]

    def _build_episode_index_parallel(self):
        """Builds a mapping from global episode indices to (TFRecord file, local episode index) using parallel processing."""
        episode_index_map = {}
        episode_count = 0

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self._process_file_for_index, self.record_files)

        for episodes_in_file in results:
            for local_idx in episodes_in_file:
                episode_index_map[episode_count] = local_idx
                episode_count += 1

        return episode_index_map

    def save_episode_index_sqlite(self):
        """Saves the episode index to an SQLite database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Create table if it doesn't exist
        c.execute("CREATE TABLE IF NOT EXISTS episode_index (global_id INTEGER PRIMARY KEY, file_path TEXT, local_index INTEGER)")

        # Clear any existing data (useful if rebuilding index)
        c.execute("DELETE FROM episode_index")

        # Insert new index data
        c.executemany("INSERT INTO episode_index VALUES (?, ?, ?)", [(k, v[0], v[1]) for k, v in self._episode_index_map.items()])

        conn.commit()
        conn.close()
        print(f"Episode index saved to {self.db_path}")

    def load_episode_index_sqlite(self):
        """Loads the episode index from an SQLite database if it exists."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT global_id, file_path, local_index FROM episode_index")
        rows = c.fetchall()

        conn.close()
        print(f"Loaded episode index from {self.db_path}")

        return {row[0]: (row[1], row[2]) for row in rows}

    def _episode_index_exists(self):
        """Checks if an episode index exists in the SQLite database."""
        if not os.path.exists(self.db_path):
            return False

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='episode_index'")
        table_exists = c.fetchone()[0] > 0

        if table_exists:
            c.execute("SELECT COUNT(*) FROM episode_index")
            row_count = c.fetchone()[0]
        else:
            row_count = 0

        conn.close()

        return row_count > 0  # True if episodes are stored

    def get_episode_by_index(self, episode_index):
        """Retrieves an episode using its global index across all TFRecords."""
        if episode_index in self._episode_index_map:
            record_path, local_index = self._episode_index_map[episode_index]
            episodes = self.parse_tfrecord(record_path)
            return episodes[local_index]
        else:
            raise IndexError(f"Episode index {episode_index} is out of range. Max index: {len(self._episode_index_map) - 1}")