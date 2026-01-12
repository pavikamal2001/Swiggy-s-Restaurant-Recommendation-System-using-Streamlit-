import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# 1️⃣ Load datasets
# -----------------------
cleaned_df = pd.read_csv("cleaned_data.csv")
encoded_df = pd.read_csv("encoded_datanew.csv")

# Clean column names
cleaned_df.columns = cleaned_df.columns.str.strip().str.lower()
encoded_df.columns = encoded_df.columns.str.strip()

# -----------------------
# 2️⃣ Load encoders
# -----------------------
with open("encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

city_encoder = encoders["city_encoder"]
cuisine_encoder = encoders["cuisine_encoder"]

# -----------------------
# 3️⃣ Streamlit UI
# -----------------------
st.set_page_config(page_title="Restaurant Recommendation", layout="wide")
st.title("🍽️ Restaurant Recommendation System")


st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color:  #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.sidebar.header("User Preferences")

city = st.sidebar.selectbox(
    "Select City",
    sorted(cleaned_df["city"].unique())
)

cuisine = st.sidebar.multiselect(
    "Select Cuisine",
    sorted(cleaned_df["cuisine"].unique())
)

min_rating = st.sidebar.slider(
    "Minimum Rating",
    0.0, 5.0, 4.0, 0.1
)

min_rating_count = st.sidebar.slider(
    "Minimum Rating Count",
    0,
    int(cleaned_df["rating_count_log"].max()),
    10
)

max_price = st.sidebar.slider(
    "Maximum Cost",
    int(cleaned_df["cost"].min()),
    int(cleaned_df["cost"].max()),
    600
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    1, 20, 5
)

# -----------------------
# 4️⃣ Recommendation Engine
# -----------------------
if st.sidebar.button("Get Recommendations"):

    if not cuisine:
        st.warning("Please select at least one cuisine")
    else:
        # ---- Build user vector ----
        user_vector = pd.DataFrame(
            np.zeros((1, encoded_df.shape[1])),
            columns=encoded_df.columns
        )

        # Encode city
        city_col = f"city_{city}"
        if city_col in user_vector.columns:
            user_vector[city_col] = 1

        # Encode cuisine(s)
        for c in cuisine:
            cuisine_col = f"cuisine_{c}"
            if cuisine_col in user_vector.columns:
                user_vector[cuisine_col] = 1

        # ---- Cosine Similarity ----
        similarity = cosine_similarity(user_vector, encoded_df)[0]

        # ---- Attach similarity ----
        cleaned_df["similarity"] = similarity

        # ---- Apply filters ----
        recommendations = cleaned_df[
            (cleaned_df["city"] == city) &
            (cleaned_df["rating"] >= min_rating) &
            (cleaned_df["rating_count_log"] >= min_rating_count) &
            (cleaned_df["cost"] <= max_price)
        ].copy()

        # ---- Sort results ----
        recommendations = recommendations.sort_values(
            by=["similarity", "rating", "rating_count_log"],
            ascending=False
        ).head(top_n)

        # -----------------------
        # 5️⃣ Output
        # -----------------------
        if recommendations.empty:
            st.error("No matching restaurants found.")
        else:
            st.success(f"Top {top_n} Recommended Restaurants")

            st.dataframe(
                recommendations[
                    [
                        "name",
                        "city",
                        "cuisine",
                        "rating",
                        "rating_count_log",
                        "cost",
                        "similarity"
                    ]
                ].reset_index(drop=True)
            )

            # 📊 visualization
            st.subheader("📊 Popularity (Rating Count)")
            st.bar_chart(
                recommendations.set_index("name")["rating_count_log"]
            )
