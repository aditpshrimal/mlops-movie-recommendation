import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"  # Change this to your FastAPI app's base URL

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

def display_recommendations(user_id):
    response = requests.get(f"{API_BASE_URL}/recommendations/{user_id}")
    recommendations = response.json()

    st.subheader("Recommendations for you:")
    for rec in recommendations:
        st.image(rec["url"] or "https://via.placeholder.com/350x500", width=350)
        if st.button(rec["title"], key="rec_" + rec["asin"]):
            display_product_detail(rec["asin"])



def display_product_detail(item_id):
    response = requests.get(f"{API_BASE_URL}/product_detail/{item_id}")
    product = response.json()

    if "error" in product:
        st.error(product["error"])
        return

    st.subheader(product["title"])
    st.image(product["imageURLHighRes"] or "https://via.placeholder.com/350x500", width=350)

    st.subheader("Similar items:")
    for item in product["similar_items"]:
        if st.button(item["title"], key=item["asin"]):
            display_product_detail(item["asin"])

page = st.sidebar.radio("Choose a page:", ["Login", "Recommendations", "Product Detail"])

if page == "Login":
    user_id = st.text_input("Enter your user ID:")
    if user_id:
        display_recommendations(user_id)
elif page == "Recommendations":
    user_id = st.text_input("Enter your user ID:")
    if user_id:
        display_recommendations(user_id)
elif page == "Product Detail":
    item_id = st.text_input("Enter an item ID:")
    if item_id:
        display_product_detail(item_id)
