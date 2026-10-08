# Hybrid Recommendation Engine with Cold-Start Handling

## 1. Project Overview

Recommendation systems are widely used by platforms such as Netflix, Amazon, Spotify, and YouTube to suggest relevant content to users.

However, a major challenge is the **Cold-Start Problem**.

When a new user joins a platform, the system has little or no information about their preferences. Traditional collaborative filtering methods struggle in this situation because they depend on previous user-item interactions.

This project develops a **Hybrid Recommendation Engine** that combines:

* **Collaborative Filtering using SVD**
* **Content-Based Filtering using TF-IDF**
* **Dynamic weighting based on user interaction history**
* **Cold-start handling for users with fewer than 5 interactions**

The system dynamically changes the importance of collaborative and content-based recommendations depending on how much information is available about a user.

---

## 2. Problem Statement

Traditional collaborative filtering performs well when users have sufficient interaction history.

However:

* New users have little or no rating history.
* New users cannot be represented accurately by collaborative filtering.
* Recommendations can therefore become inaccurate for new users.

The goal of this project is to build a recommendation system that can **degrade gracefully when user interaction history is limited**.

The system should:

1. Combine collaborative filtering and content-based recommendations.
2. Dynamically adjust their contribution.
3. Give more importance to content when user history is limited.
4. Give more importance to collaborative filtering when sufficient interaction history is available.
5. Evaluate performance separately for cold-start and warm users.

---

## 3. Proposed Solution

The proposed system combines two recommendation approaches.

### Collaborative Filtering

Collaborative filtering learns user preferences from historical ratings.

We use **Singular Value Decomposition (SVD)** to learn latent relationships between users and movies.

When a user has many interactions, collaborative filtering becomes more reliable.

### Content-Based Filtering

Content-based filtering recommends movies based on their characteristics.

In this project, movie **genres** are converted into numerical TF-IDF vectors.

Cosine similarity is then used to find movies that are similar to movies the user has previously liked.

### Dynamic Hybrid Model

Instead of using a fixed 50/50 combination, the system dynamically changes the weights.

The collaborative filtering weight is calculated as:

`CF Weight = interactions / (interactions + 5)`

The content-based weight is:

`Content Weight = 1 - CF Weight`

Therefore:

| User Interactions | CF Weight | Content Weight |
| ----------------: | --------: | -------------: |
|                 0 |      0.00 |           1.00 |
|                 1 |      0.17 |           0.83 |
|                 2 |      0.29 |           0.71 |
|                 3 |      0.38 |           0.62 |
|                 5 |      0.50 |           0.50 |
|                10 |      0.67 |           0.33 |
|                20 |      0.80 |           0.20 |
|                50 |      0.91 |           0.09 |
|               100 |      0.95 |           0.05 |

This allows the system to automatically shift from **content-heavy recommendations for sparse users** to **collaborative-filtering-heavy recommendations for experienced users**.

---

## 4. System Architecture

```text
                    MovieLens Dataset
                           |
              +------------+------------+
              |                         |
              v                         v
        Rating Data                Movie Data
              |                         |
              v                         v
      Collaborative              Content Features
       Filtering (SVD)              (TF-IDF)
              |                         |
              v                         v
         CF Score                Content Score
              |                         |
              +------------+------------+
                           |
                           v
                  Dynamic Weighting
                           |
               +-----------+-----------+
               |                       |
        Few Interactions        Many Interactions
               |                       |
        Content-heavy              CF-heavy
               |                       |
               +-----------+-----------+
                           |
                           v
                    Hybrid Score
                           |
                           v
                   Movie Ranking
                           |
                           v
                     Top-K Movies
                           |
                           v
                    Recommendation
```

---

## 5. Dataset

This project uses the **MovieLens 1M dataset** provided by GroupLens.

The dataset contains approximately:

* 1 million movie ratings
* 6,000 users
* 4,000 movies
* Ratings from 1 to 5
* Movie genre information

### Files Used

#### `ratings.dat`

Contains:

* User ID
* Movie ID
* Rating
* Timestamp

This file is primarily used for collaborative filtering.

#### `movies.dat`

Contains:

* Movie ID
* Movie title
* Movie genres

This file is used for content-based filtering.

#### `users.dat`

Contains demographic information about users.

It was not required for the main recommendation model.

---

## 6. Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Scikit-Surprise
* Matplotlib

### Machine Learning Techniques

* Singular Value Decomposition (SVD)
* TF-IDF Vectorization
* Cosine Similarity
* Hybrid Recommendation
* Dynamic Weighting

---

## 7. Data Preprocessing

The MovieLens `.dat` files use `::` as their separator.

The ratings data was loaded into a Pandas DataFrame with the following columns:

```text
userId
movieId
rating
timestamp
```

The movie data was loaded as:

```text
movieId
title
genres
```

For content-based filtering, the `|` separator between genres was replaced with spaces.

For example:

```text
Action|Adventure|Sci-Fi
```

becomes:

```text
Action Adventure Sci-Fi
```

This allows the genres to be processed using TF-IDF.

---

## 8. Collaborative Filtering — SVD

Collaborative filtering uses user rating patterns to predict how much a user may like an unseen movie.

The project uses the **SVD algorithm** from the Surprise library.

The main configuration is:

```python
SVD(
    n_factors=100,
    n_epochs=20,
    random_state=42
)
```

The ratings dataset was divided into:

* 80% training data
* 20% testing data

The SVD model was trained using the training ratings.

The predicted rating is then normalized to a 0–1 range so that it can be combined with the content-based score.

---

## 9. Content-Based Filtering

The content-based component uses movie genres to identify similar movies.

### Step 1 — TF-IDF

Movie genres are converted into TF-IDF vectors.

For example:

```text
Toy Story → Animation Comedy Children's
```

The TF-IDF representation allows the system to compare movies mathematically.

### Step 2 — Cosine Similarity

Cosine similarity measures how similar two movie genre vectors are.

A similarity closer to `1` means the movies are more similar.

For a user, the system:

1. Finds movies the user rated 4 or 5.
2. Finds movies similar to those liked movies.
3. Calculates the average similarity.
4. Uses this as the content-based recommendation score.

---

## 10. Hybrid Recommendation

The final recommendation score combines collaborative filtering and content-based scores.

The basic hybrid formula is:

```text
Hybrid Score =
(CF Weight × CF Score)
+
(Content Weight × Content Score)
```

The weights depend on the number of interactions available for the user.

This prevents the system from relying heavily on collaborative filtering when there is not enough user history.

---

## 11. Cold-Start Handling

### What is Cold Start?

A cold-start user is a user who has very few or no previous interactions with the system.

Collaborative filtering struggles with such users because there is insufficient information to learn their preferences.

### Problem with MovieLens 1M

The MovieLens 1M dataset does not naturally contain users with fewer than 5 ratings.

Therefore, an artificial cold-start evaluation scenario was constructed.

### Cold-Start Experiment

Selected users with sufficient original rating history were chosen.

For each selected user:

* Only **3 ratings** were exposed to the recommendation system.
* The remaining ratings were hidden.
* The hidden ratings were used as ground truth during evaluation.

Therefore, from the model's perspective:

```text
Original User
      |
      v
Many historical ratings
      |
      v
Only 3 ratings exposed
      |
      v
Cold-start user
```

The system then generated recommendations using the limited information.

Movies that the user rated **4 or 5** in the hidden portion were considered relevant recommendations.

---

## 12. Cold-Start Hybrid Strategy

For cold-start users, collaborative filtering has limited personalization capability.

Therefore, the system uses:

* Content similarity as the main personalized signal.
* Movie popularity as a fallback collaborative signal.
* Dynamic weighting to give greater importance to content when interactions are limited.

For example, with 3 interactions:

```text
CF Weight       ≈ 37.5%
Content Weight  ≈ 62.5%
```

As the number of interactions increases, the collaborative component becomes more important.

---

## 13. Evaluation Metrics

The recommendation system is evaluated using:

### Precision@10

Measures how many of the top 10 recommended movies were relevant.

```text
Precision@10 =
Relevant Recommended Movies / 10
```

A higher value indicates that more recommended movies were relevant.

---

### Recall@10

Measures how many of the user's relevant movies were successfully recommended.

```text
Recall@10 =
Relevant Recommended Movies /
Total Relevant Movies
```

---

### NDCG@10

Normalized Discounted Cumulative Gain considers the position of relevant recommendations.

A relevant movie appearing near the top of the recommendation list receives more importance than one appearing near the bottom.

NDCG ranges from:

```text
0 → Poor ranking
1 → Ideal ranking
```

---

## 14. Evaluation Setup

The system was evaluated separately on:

### Cold-Start Users

Users with fewer than 5 visible interactions.

In the experiment, users were provided with exactly 3 known interactions.

### Warm Users

Users with 5 or more available interactions.

This allows us to compare how recommendation performance changes as more user information becomes available.

---

## 15. Results

The final evaluation compares:

1. Content-Based model
2. Hybrid model for cold-start users
3. Hybrid model for warm users

### Final Results

Replace the values below with the actual values obtained from the notebook.

| Model             | Precision@10 |    Recall@10 |      NDCG@10 |
| ----------------- | -----------: | -----------: | -----------: |
| Content-Based     | `YOUR_VALUE` | `YOUR_VALUE` | `YOUR_VALUE` |
| Hybrid Cold-Start | `YOUR_VALUE` | `YOUR_VALUE` | `YOUR_VALUE` |
| Hybrid Warm Users | `YOUR_VALUE` | `YOUR_VALUE` | `YOUR_VALUE` |

The results demonstrate how the hybrid recommendation approach behaves under different amounts of available user information.

---

## 16. Dynamic Weighting Results

The dynamic weighting mechanism demonstrates the intended behavior:

```text
Low interaction history
        ↓
More Content-Based Weight
        ↓
Better use of available movie information

High interaction history
        ↓
More Collaborative Filtering Weight
        ↓
Better use of learned user preferences
```

This allows the system to adapt its recommendation strategy based on the amount of information available.

---

## 17. Failure Case Analysis

Cold-start recommendation remains challenging because the system has very little information about a new user's preferences.

A failure case was analyzed by selecting a cold-start user with a low NDCG@10 score.

The analysis compared:

* The user's 3 known movies
* Movies recommended by the system
* Movies the user actually liked in the hidden evaluation data

### Possible Reasons for Failure

1. Only three interactions were available.
2. The known movies may not represent the user's complete interests.
3. Genre similarity may not capture deeper movie preferences.
4. Popularity can introduce movies that are popular but not personally relevant.
5. Movie descriptions, actors, directors, and tags were not included in the current content model.

These limitations can cause the system to recommend movies that are technically similar but not necessarily preferred by the user.

---

## 18. Limitations

The current implementation has several limitations:

* Content-based filtering uses mainly movie genres.
* Movie descriptions, actors, directors, and tags are not included.
* The cold-start experiment is simulated because MovieLens 1M does not naturally contain users with fewer than 5 ratings.
* Popularity is used as a fallback signal for cold users.
* The system is currently designed for movie recommendation.
* The evaluation is performed offline rather than using real-time user feedback.

---

## 19. Future Improvements

The system can be improved by adding:

### Better Content Features

Use:

* Movie descriptions
* Keywords
* Actors
* Directors
* Tags

instead of relying mainly on genres.

### Advanced Embeddings

Use models such as:

* Sentence Transformers
* BERT-based embeddings
* Other semantic text embeddings

to understand movie descriptions more deeply.

### Advanced Collaborative Filtering

Experiment with:

* ALS
* Matrix Factorization
* Neural Collaborative Filtering

### New-Item Cold Start

Extend the system to handle movies that have very few or no ratings.

### Real-Time Feedback

Allow users to provide:

* Likes
* Dislikes
* Ratings
* Watch history

and update recommendations dynamically.

---

## 20. Project Structure

```text
hybrid-recommendation-engine/
│
├── README.md
│
├── notebook/
│   └── hybrid_recommender.ipynb
│
├── results/
│   ├── performance.png
│   └── dynamic_weighting.png
│
├── data/
│   └── README.md
│
└── requirements.txt
```

The MovieLens dataset itself does not need to be stored in the repository. Users can download it separately.

---

## 21. Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd hybrid-recommendation-engine
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

The notebook can then be opened using Jupyter Notebook, JupyterLab, or Google Colab.

---

## 22. Requirements

The main Python dependencies are:

```text
pandas
numpy
scikit-learn
scikit-surprise
matplotlib
```

---

## 23. How to Run

1. Download the MovieLens 1M dataset.
2. Place `ratings.dat` and `movies.dat` in the expected data location.
3. Open the notebook.
4. Install the required Python libraries.
5. Run the cells sequentially.
6. Train the SVD model.
7. Build the TF-IDF content model.
8. Generate hybrid recommendations.
9. Construct the cold-start evaluation users.
10. Evaluate Precision@10, Recall@10, and NDCG@10.
11. Compare cold-start and warm-user performance.

---

## 24. Example Workflow

```text
Load MovieLens Dataset
          ↓
Data Preprocessing
          ↓
      ┌───┴───┐
      ↓       ↓
     SVD     TF-IDF
      ↓       ↓
   CF Score  Content Score
      ↓       ↓
      └───┬───┘
          ↓
 Dynamic Weighting
          ↓
    Hybrid Score
          ↓
   Rank Candidate Movies
          ↓
      Top 10
          ↓
   Recommendation
```

---

## 25. Conclusion

This project demonstrates a hybrid recommendation system designed to address the cold-start problem.

The system combines collaborative filtering using SVD with content-based filtering using TF-IDF and cosine similarity. Instead of using fixed weights, the system dynamically adjusts the contribution of each component according to the user's interaction history.

For users with limited history, the system shifts toward content-based recommendations, while users with more interactions receive greater influence from collaborative filtering.

The system was evaluated using Precision@10, Recall@10, and NDCG@10 on both cold-start and warm-user scenarios.

The project demonstrates how combining multiple recommendation strategies can provide a more flexible recommendation system capable of handling different levels of available user information.

---

**Project:** Hybrid Recommendation Engine with Cold-Start Handling

**Dataset:** MovieLens 1M
