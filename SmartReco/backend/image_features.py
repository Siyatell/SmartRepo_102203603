import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA


class ImageFeatureExtractor:
    """
    Lightweight image feature extractor using Pillow + NumPy + PCA.
    This replaces TensorFlow-based models for environments without TF support.
    """

    def __init__(self, image_dir="data/images", feature_dim=128):
        self.image_dir = image_dir
        self.feature_dim = feature_dim
        self.pca = PCA(n_components=feature_dim)
        self.features = {}
        self.image_names = []

    def _process_image(self, img_path):
        """Resize and flatten image into a simple numeric vector."""
        try:
            with Image.open(img_path).convert("RGB") as img:
                img = img.resize((64, 64))  # small, consistent size
                arr = np.array(img).astype("float32") / 255.0
                return arr.flatten()
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            return None

    def load_images(self):
        """Loads all images from the directory and computes PCA features."""
        if not os.path.exists(self.image_dir):
            print(f"❌ Directory not found: {self.image_dir}")
            return

        print("📸 Loading images...")
        vectors = []

        for filename in os.listdir(self.image_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(self.image_dir, filename)
                vec = self._process_image(path)
                if vec is not None:
                    vectors.append(vec)
                    self.image_names.append(filename)

        if len(vectors) == 0:
            print("⚠️ No valid images found.")
            return

        print("🔍 Computing PCA for dimensionality reduction...")
        reduced = self.pca.fit_transform(np.array(vectors))
        self.features = dict(zip(self.image_names, reduced))
        print(f"✅ Extracted features for {len(self.features)} images.")

    def get_feature(self, image_name):
        """Return feature vector for a given image."""
        return self.features.get(image_name, None)
