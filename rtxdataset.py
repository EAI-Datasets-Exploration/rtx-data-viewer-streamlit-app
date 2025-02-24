import concurrent.futures
import tensorflow as tf
import sqlite3
import os
import glob

class RTXDataset:
    def __init__(self, name, local_data_path, text_key, image_key, is_first_key, is_last_key, db_path="sqlite3_files/"):
        self.name = name
        self.local_data_path = local_data_path
        self.text_key = text_key
        self.image_key = image_key
        self.is_first_key = is_first_key
        self.is_last_key = is_last_key
        self.db_path = os.path.join(db_path, f"{self.name}_episode_index.db")

        self.image_h = 224
        self.image_w = 224

        self.record_files = self.create_tfrecord_files_list()
        self._cached_episodes = {}

        if self._episode_index_exists():
            self._episode_index_map = self.load_episode_index_sqlite()
        else:
            print("🛠 Building episode index...")
            self._episode_index_map = self._build_episode_index_parallel()
            self.save_episode_index_sqlite()

    def _episode_index_exists(self):
        """Checks if the SQLite database contains an episode index."""
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
        return row_count > 0

    def create_tfrecord_files_list(self):
        """Lists all files containing 'tfrecord' in their filename in the specified local directory."""
        all_files = glob.glob(os.path.join(self.local_data_path, "**/*"), recursive=True)  # Recursively list all files
        tfrecord_files = [file for file in all_files if "tfrecord" in os.path.basename(file)]  # Filter for 'tfrecord'

        if not tfrecord_files:
            print(f"❌ ERROR: No TFRecord files found in {self.local_data_path}. Check dataset path.")
        else:
            print(f"✅ Found {len(tfrecord_files)} TFRecord files in {self.local_data_path}")
        
        return tfrecord_files

    def get_record_files(self):
        return self.record_files

    def parse_tfrecord(self, record_path):
        """Parses a single TFRecord file and returns a list of episodes."""

        print(f"record path in parse_tfrecord(): {record_path}")

        raw_dataset = tf.data.TFRecordDataset(record_path)
        episodes = []
        current_texts = []
        current_images = []

        print(f"🔍 Parsing TFRecord: {record_path}")

        for raw_record in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            if self.text_key not in example.features.feature or self.image_key not in example.features.feature:
                print(f"❌ ERROR: Required keys {self.text_key}, {self.image_key} not found in {record_path}.")
                return []

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

        print(f"📊 Total Episodes Parsed: {len(episodes)} from {record_path}")
        return episodes

    def _build_episode_index_parallel(self):
        """Builds the episode index using parallel processing."""
        episode_index_map = {}
        episode_count = 0

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self._process_file_for_index, self.record_files)

        for episodes_in_file in results:
            for local_idx in episodes_in_file:
                episode_index_map[episode_count] = (local_idx[0], local_idx[1], local_idx[2])
                episode_count += 1

        if not episode_index_map:
            print("❌ ERROR: No episodes indexed. Check TFRecord parsing.")
        return episode_index_map

    def _process_file_for_index(self, record_file):
        """Processes a single TFRecord file and returns metadata for indexing."""
        episodes = self.parse_tfrecord(record_file)
        return [(record_file, i, " ".join(episodes[i][0])) for i in range(len(episodes))]

    def save_episode_index_sqlite(self):
        """Saves the episode index and text data to an SQLite database."""
        if not self._episode_index_map:
            print("❌ ERROR: No episodes found in index. Ensure TFRecord files are correctly parsed.")
            return

        print(f"🛠 Creating SQLite DB at: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Create episode index table
        c.execute("""
            CREATE TABLE IF NOT EXISTS episode_index (
                global_id INTEGER PRIMARY KEY,
                file_path TEXT,
                local_index INTEGER,
                text TEXT
            )
        """)

        # Create full-text search table
        c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS episode_texts USING fts5(global_id, text)")

        # Clear old data
        c.execute("DELETE FROM episode_index")
        c.execute("DELETE FROM episode_texts")

        episode_entries = []
        text_entries = []

        for global_id, (file_path, local_index, episode_text) in self._episode_index_map.items():
            print(f"📌 Storing Episode {global_id}: {episode_text[:100]}...")  # Debugging
            episode_entries.append((global_id, file_path, local_index, episode_text))
            text_entries.append((global_id, episode_text))

        if not episode_entries:
            print("❌ ERROR: No data to insert into SQLite!")
            conn.close()
            return

        try:
            c.executemany("INSERT INTO episode_index VALUES (?, ?, ?, ?)", episode_entries)
            c.executemany("INSERT INTO episode_texts VALUES (?, ?)", text_entries)
            conn.commit()
            print(f"✅ SQLite DB successfully created: {self.db_path} with {len(episode_entries)} episodes")
        except Exception as e:
            print(f"❌ SQLite Insert Error: {e}")
        finally:
            conn.close()

    def load_episode_index_sqlite(self):
        """Loads the episode index from SQLite."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("SELECT global_id, file_path, local_index, text FROM episode_index")
        rows = c.fetchall()
        conn.close()

        if rows:
            print(f"✅ Loaded {len(rows)} episodes from {self.db_path}")
        else:
            print(f"❌ ERROR: SQLite database is empty!")

        return {row[0]: (row[1], row[2], row[3]) for row in rows}

    def search_episodes_by_text(self, query):
        """Search for episodes containing the given text."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Using FTS5 full-text search
        c.execute("SELECT global_id FROM episode_texts WHERE text MATCH ?", (query,))
        matching_ids = [row[0] for row in c.fetchall()]

        conn.close()

        if not matching_ids:
            print(f"❌ No episodes found for query: {query}")
            return []

        matching_episodes = []
        for episode_index in matching_ids:
            if episode_index in self._cached_episodes:
                matching_episodes.append((episode_index, self._cached_episodes[episode_index]))
            else:
                record_path, local_index, _ = self._episode_index_map[episode_index]
                episodes = self.parse_tfrecord(record_path)
                episode_data = (episodes[local_index][0], episodes[local_index][1])

                # Cache for faster future retrieval
                self._cached_episodes[episode_index] = episode_data
                matching_episodes.append((episode_index, episode_data))

        return matching_episodes


    def get_episode_by_index(self, episode_index):
        """Retrieves an episode using its global index."""
        if episode_index in self._episode_index_map:
            record_path, local_index, _ = self._episode_index_map[episode_index]  # FIXED: Unpacking 3 values
            episodes = self.parse_tfrecord(record_path)
            return episodes[local_index]
        else:
            raise IndexError(f"Episode index {episode_index} is out of range. Max index: {len(self._episode_index_map) - 1}")
