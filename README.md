# Movie Recommender App

This is a **Movie Recommendation System** built with **Streamlit**, **FAISS**, and **LangChain**. The app allows users to filter movies by genre, IMDb rating, and duration, and provides personalized recommendations based on a search query.

---

## Features

- **Search for Movies**: Use natural language queries to find movies (e.g., "A movie about superheroes").
- **Filter Options**:
  - Filter by **Genre** (e.g., Action, Comedy, Drama, etc.).
  - Filter by **IMDb Rating** (e.g., 6.0 - 6.9, 8.5 - 10.0).
  - Filter by **Duration** (e.g., 0-60 minutes, 121-180 minutes).
- **Movie Details**: Displays movie posters, titles, IMDb ratings, genres, directors, and overviews.
- **Load More**: Lazy loading to display more movies incrementally.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- `pip` for package management

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/dmquan1105/movies-recommender.git
   cd movie-recommender
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Open the app in your browser (Streamlit will provide a local URL, e.g., http://localhost:8501).
2. Use the search bar to enter a query (e.g., "A movie about superheroes").
3. Apply filters for Genre, IMDb Rating, and Duration using the sidebar.
4. View the recommended movies with their details and posters.
5. Click Load More to display additional recommendations.

---

## Dataset

The app uses the IMDb Dataset of Top 1000 Movies and TV Shows, you can find it [here](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows) which includes:

- Movie titles, genres, IMDb ratings, release years, overviews, and more.
- The dataset is preprocessed and stored in movies_cleaned.csv.

---

## Technologies Used

- Streamlit: For building the interactive web app.
- FAISS: For efficient similarity search and vector-based recommendations.
- LangChain: For handling embeddings and vector stores.
- Hugging Face Transformers: For generating embeddings using the sentence-transformers/all-MiniLM-L6-v2 model.
- Pandas: For data manipulation and preprocessing.
- NumPy: For numerical operations.

---

## File Structure

- app.py: Main Streamlit app file.
- requirements.txt: List of dependencies for the app.
- movies_cleaned.csv: Preprocessed movie dataset.
- tag_desc.txt: Text file containing movie descriptions for vector search.
- data-exploration.ipynb: Jupyter Notebook for dataset exploration and preprocessing.
- vector-search.ipynb: Jupyter Notebook for vector search implementation.

---

## Deployment

The app is deployed on Streamlit Cloud. You can access it [here](https://dmq-movies-recommender.streamlit.app).
