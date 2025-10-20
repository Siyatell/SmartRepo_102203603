import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from genai_description import enhance_description

class Recommender:
    def __init__(self, data_path="data/products.csv"):
        print("🔄 Loading product data and model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        try:
            self.products = pd.read_csv(data_path)
            print(f"✅ Loaded {len(self.products)} products.")
        except FileNotFoundError:
            raise Exception("❌ products.csv not found in /data folder!")

        required_cols = ["id","name","description","category","price"]
        for col in required_cols:
            if col not in self.products.columns:
                raise Exception(f"❌ Missing column: {col}")

        # Fill missing or weak descriptions
        for i, row in self.products.iterrows():
            if pd.isna(row["description"]) or len(str(row["description"]).strip()) < 10:
                self.products.at[i, "description"] = enhance_description(
                    row.get("name", ""),
                    row.get("category", ""),
                    row.get("material", ""),
                    row.get("brand", "")
                )

        print("🧠 Generating embeddings for all product descriptions...")
        self.embeddings = self.model.encode(
            self.products["description"].astype(str).tolist(), convert_to_tensor=True
        )
        print("✅ Model and embeddings ready!")

    def recommend(self, query: str, top_k: int = 3):
        query_emb = self.model.encode([query], convert_to_tensor=True)
        scores = cosine_similarity(query_emb.cpu(), self.embeddings.cpu())[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        return self.products.iloc[top_indices].to_dict(orient="records")
