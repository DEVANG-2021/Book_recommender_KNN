import pandas as pd

# Read Ratings csv file
ratings = pd.read_csv("data/Ratings.csv", encoding='latin-1')

# Show top-5 records
ratings.head()
# Read Books csv file
books = pd.read_csv("data/Books.csv", encoding='latin-1')

# Show top-5 records
books.head()

# Join ratings and books dataframes
rating_books=pd.merge(ratings,books,on="ISBN")

# Shape of the data
rating_books.shape
# Take 1 % data as sample  
rating_books_sample = rating_books.sample(frac=.01, random_state=1) 

# Shape of the sample data
rating_books_sample.shape
# Create Item-user matrix using pivot_table()
rating_books_pivot = rating_books_sample.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)

# Show top-5 records
rating_books_pivot.head()
# Import NearestNeighbors
from sklearn.neighbors import NearestNeighbors

# Build NearestNeighbors Object
model_nn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=7, n_jobs=-1)

# Fit the NearestNeighbor
model_nn.fit(rating_books_pivot)
# Get top 10 nearest neighbors 
indices=model_nn.kneighbors(rating_books_pivot.loc[['10 Secrets for Success and Inner Peace']], 10, return_distance=False)

# Print the recommended books
print("Recommended Books:")
print("==================")
for index, value in enumerate(rating_books_pivot.index[indices][0]):
    print((index+1),". ",value)