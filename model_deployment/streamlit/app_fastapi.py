from fastapi import FastAPI
import pandas as pd
import pickle
from typing import List
import scipy.sparse as sparse

app = FastAPI()

# Load the trained model
with open("../final_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the metadata
product_metadata = pd.read_json("../data/meta_Movies_and_TV.json", lines=True)

df = pd.read_csv("../data/Movies_and_TV.csv", header=None)
df = df.sample(frac=0.2, random_state=42)
df.columns = ["asin", "reviewerId", "overallRating", "timestamp"]

df["user_id"] = df["reviewerId"]
df["item_id"] = df["asin"]

df["reviewerId"] = df["reviewerId"].astype("category").cat.codes.values.astype(int)
df["asin"] = df["asin"].astype("category").cat.codes.values.astype(int)
df["overallRating"] = df["overallRating"].astype(float)

user_id2token = pd.Series(df.user_id.values, index=df.reviewerId).to_dict()
item_id2token = pd.Series(df.item_id.values, index=df.asin).to_dict()

user_token2id = pd.Series(df.reviewerId.values, index=df.user_id).to_dict()
item_token2id = pd.Series(df.asin.values, index=df.item_id).to_dict()

sparse_user_item = sparse.csr_matrix(
    (df["overallRating"], (df["reviewerId"], df["asin"]))
)

def get_recommendations(user_id, N=10):
    user_id = user_token2id[user_id]
    recommended = model.recommend(user_id, sparse_user_item[user_id], N)[0]
    recommendations = []
    for r in recommended:
        asin = item_id2token[r]
        title = product_metadata.loc[product_metadata["asin"] == asin, "title"].values[0]
        try:
            url = product_metadata.loc[product_metadata["asin"] == asin, "imageURLHighRes"].values[0][0]
        except:
            url = None
        recommendations.append({"asin": asin, "title": title, "url": url})
    return recommendations

def get_similar_items(item_id, N=10):
    item_id = item_token2id[item_id]
    recommended = model.similar_items(item_id, N)[0]
    similar_items = []
    for r in recommended:
        asin = item_id2token[r]
        title = product_metadata.loc[product_metadata["asin"] == asin, "title"].values[0]
        try:
            url = product_metadata.loc[product_metadata["asin"] == asin, "imageURLHighRes"].values[0][0]
        except:
            url = None
        similar_items.append({"asin": asin, "title": title, "url": url})
    return similar_items

@app.get("/recommendations/{user_id}")
def recommendations(user_id: str, N: int = 10) -> List[dict]:
    return get_recommendations(user_id, N)

@app.get("/similar_items/{item_id}")
def similar_items(item_id: str, N: int = 10) -> List[dict]:
    return get_similar_items(item_id, N)

@app.get("/search/{query}")
def search(query: str) -> List[dict]:
    results = product_metadata[product_metadata["title"].str.contains(query, case=False, na=False)]
    return results.to_dict('records')

@app.get("/product_detail/{item_id}")
def product_detail(item_id: str) -> dict:
    product = product_metadata.loc[product_metadata["asin"] == item_id]
    if product.empty:
        return {"error": "Product not found"}
    product = product.iloc[0].to_dict()
    similar_items = get_similar_items(item_id)
    product["similar_items"] = similar_items
    return product