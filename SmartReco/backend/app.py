from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from recommender import Recommender

app = FastAPI(title="SmartReco API", version="1.0")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommender once
recommender = Recommender()

@app.get("/")
def home():
    return {"message": "Welcome to SmartReco API!"}

@app.post("/recommend")
def recommend_product(query: str = Query(...), top_k: int = 3):
    try:
        recs = recommender.recommend(query, top_k)
        return {"query": query, "recommendations": recs}
    except Exception as e:
        return {"error": str(e)}
