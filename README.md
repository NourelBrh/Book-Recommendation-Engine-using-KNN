# Book-Recommendation-Engine-using-KNN

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
books = pd.read_csv('books.csv')
ratings = pd.read_csv('ratings.csv')

# Filter users with fewer than 200 ratings
user_rating_counts = ratings['user_id'].value_counts()
valid_users = user_rating_counts[user_rating_counts >= 200].index
filtered_ratings = ratings[ratings['user_id'].isin(valid_users)]

# Filter books with fewer than 100 ratings
book_rating_counts = filtered_ratings['book_id'].value_counts()
valid_books = book_rating_counts[book_rating_counts >= 100].index
filtered_ratings = filtered_ratings[filtered_ratings['book_id'].isin(valid_books)]

# Join with book metadata
filtered_ratings = filtered_ratings.merge(books, on='book_id')

# Create a user-item matrix
user_item_matrix = filtered_ratings.pivot(index='user_id', columns='book_title', values='rating').fillna(0)

# Use TF-IDF Vectorizer to convert book titles to vectors (if book metadata is available)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(user_item_matrix.columns)


# Fit Nearest Neighbors model
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(X)


def get_recommends(book_title):
    # Get the index of the book title
    if book_title not in user_item_matrix.columns:
        return [book_title, []]  # Book not found in dataset
    
    book_index = user_item_matrix.columns.get_loc(book_title)
    book_vector = vectorizer.transform([book_title])
    
    # Find similar books
    distances, indices = knn.kneighbors(book_vector, n_neighbors=6)
    
    recommendations = []
    for i in range(1, len(distances[0])):  # Start from 1 to skip the book itself
        similar_book = user_item_matrix.columns[indices[0][i]]
        distance = distances[0][i]
        recommendations.append([similar_book, distance])
    
    return [book_title, recommendations]

# Example usage
print(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))


