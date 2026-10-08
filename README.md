# Hybrid Recommendation Engine with Cold-Start Handling

A hybrid movie recommendation system that combines **Collaborative Filtering** and **Content-Based Filtering** with **dynamic weighting** to handle both warm users and cold-start users.

The project is built using the **MovieLens 1M dataset** and evaluates recommendations using **Precision@10, Recall@10, and NDCG@10**.

---

## 1. Project Overview

Recommendation systems are widely used by platforms such as Netflix, Amazon, Spotify, and YouTube to personalize content for users.

A major challenge is the **cold-start problem**:

> How can a recommendation system provide useful recommendations when a user has very little interaction history?

This project addresses this problem using a **Hybrid Recommendation Engine**.

The system combines:

* **Collaborative Filtering using SVD**
* **Content-Based Filtering using TF-IDF**
* **Dynamic weighting based on user interactions**
* **Popularity-based fallback for cold-start users**
* **Precision@10, Recall@10, and NDCG@10 evaluation**

The main idea is simple:

**Less user history → rely more on content**

**More user history → rely more on collaborative filtering**

---

## 2. Problem Statement

Traditional collaborative filtering works better when sufficient user interaction data is available.

However, for a new user with very few interactions, collaborative filtering has limited information to work with.

This creates a cold-start problem.

The objective of this project is to build a recommendation system that:

1. Provides personalized movie recommendations.
2. Combines collaborative and content-based approaches.
3. Dynamically changes the contribution of each approach.
4. Handles users with very limited interaction history.
5. Evaluates performance separately for cold-start and warm users.
6. Analyzes failure cases of the recommendation system.

---

## 3. Proposed Solution

The system uses three main components:

### Collaborative Filtering

We use **SVD (Singular Value Decomposition)** to learn relationships between users and movies from their ratings.

This component becomes more useful as a user interacts with more movies.

### Content-Based Filtering

Movie genres are converted into TF-IDF vectors.

Cosine similarity is then used to find movies that are similar to movies the user previously liked.

This is especially useful when there is little interaction history.

### Dynamic Hybrid Weighting

Instead of using a fixed 50/50 combination, the system dynamically changes the weights according to the number of user interactions.

The collaborative filtering weight is calculated as:

```text
CF Weight = interactions / (interactions + 5)
```

The content-based weight is:

```text
Content Weight = 1 - CF Weight
```

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

This allows the system to gradually transition from content-based recommendations to collaborative filtering as more user data becomes available.

---

## 4. System Architecture

```text
                 MovieLens 1M Dataset
                         |
              +----------+----------+
              |                     |
          Ratings Data          Movie Data
              |                     |
              ↓                     ↓
     Collaborative Filtering   Content-Based
            (SVD)              (TF-IDF)
              |                     |
              ↓                     ↓
        CF Prediction         Similarity Score
              |                     |
              +----------+----------+
                         |
                  Dynamic Weighting
                         |
              +----------+----------+
              |                     |
       More interactions      Fewer interactions
              |                     |
         More CF weight       More content weight
              |                     |
              +----------+----------+
                         |
                         ↓
                Final Recommendation
                         |
                         ↓
              Precision / Recall / NDCG
```

---

## 5. Dataset

This project uses the **MovieLens 1M dataset** provided by GroupLens.

The dataset contains approximately:

* 1 million ratings
* 6,000 users
* 4,000 movies

### Download Dataset

Download the MovieLens 1M dataset from the official GroupLens website:

https://grouplens.org/datasets/movielens/1m/

Download:

```text
ml-1m.zip
```

After extracting it, the dataset contains:

```text
ml-1m/
├── ratings.dat
├── movies.dat
├── users.dat
└── README
```

### Required Files

The notebook uses:

```text
ratings.dat
movies.dat
users.dat
```

`users.dat` is included as part of the original dataset, although the current recommendation model primarily uses `ratings.dat` and `movies.dat`.

### Google Colab Setup

If running the notebook in Google Colab:

1. Download `ml-1m.zip`.
2. Extract the ZIP file.
3. Upload `ratings.dat`, `movies.dat`, and `users.dat` to the Colab session.
4. Run the notebook from the beginning.

The dataset files are **not included in this GitHub repository**. This keeps the repository lightweight and allows users to obtain the dataset directly from its original source.

---

## 6. Technologies Used

| Technology      | Purpose                                   |
| --------------- | ----------------------------------------- |
| Python          | Main programming language                 |
| Pandas          | Data processing                           |
| NumPy           | Numerical operations                      |
| Scikit-learn    | TF-IDF and cosine similarity              |
| Scikit-Surprise | SVD collaborative filtering               |
| Matplotlib      | Visualization                             |
| Google Colab    | Development and experimentation           |
| GitHub          | Version control and project documentation |

---

## 7. Project Workflow

### Step 1 — Load Dataset

The MovieLens ratings and movie information are loaded using Pandas.

### Step 2 — Train/Test Split

The ratings are divided into training and testing data.

The training data is used to build the recommendation model.

The test data is used to evaluate recommendations.

### Step 3 — Collaborative Filtering

An SVD model is trained using the training ratings.

The model predicts how strongly a user may prefer a particular movie.

### Step 4 — Content-Based Filtering

Movie genres are converted into TF-IDF vectors.

Cosine similarity is used to measure the similarity between movies.

If a user has previously liked a movie, similar movies receive higher content-based scores.

### Step 5 — Dynamic Hybrid Scoring

The system combines:

```text
Hybrid Score =
CF Weight × CF Score
+
Content Weight × Content Score
```

The weights depend on the number of interactions available for the user.

### Step 6 — Cold-Start Simulation

The original MovieLens 1M dataset does not naturally contain users with fewer than five ratings.

Therefore, an artificial cold-start evaluation scenario is created.

For selected users:

* Only 3 ratings are exposed as the user's known history.
* Their remaining ratings are hidden.
* The hidden ratings are used as ground truth for evaluation.

This simulates a new user who has only a few interactions with the platform.

### Step 7 — Cold-Start Recommendation

For users with very limited history, content-based recommendations become more important.

A popularity score is also used as a fallback signal on the collaborative side because a reliable personalized SVD prediction cannot be assumed for an artificially cold user.

### Step 8 — Evaluation

Recommendations are evaluated using:

* Precision@10
* Recall@10
* NDCG@10

Performance is analyzed separately for:

* Cold-start users
* Warm users

---

## 8. Evaluation Metrics

### Precision@10

Measures how many of the top 10 recommended movies were relevant.

```text
Precision@10 =
Relevant Recommended Movies / 10
```

Higher is better.

### Recall@10

Measures how many of the user's relevant movies were successfully recommended.

```text
Recall@10 =
Relevant Recommended Movies / Total Relevant Movies
```

Higher is better.

### NDCG@10

NDCG considers the position of relevant recommendations.

A relevant movie appearing near the top of the recommendation list contributes more than one appearing near the bottom.

Higher NDCG means better ranking quality.

---

## 9. Results

The project compares:

1. Content-Based Filtering
2. Cold-Start Hybrid Model
3. Hybrid Model for Warm Users

### Performance Comparison

![Performance Comparison](results/performance.png)

The graph compares the models using:

* Precision@10
* Recall@10
* NDCG@10

The exact numerical results are generated directly from the notebook.

### Dynamic Weighting

![Dynamic Weighting](results/dynamic_weighting.png)

This graph demonstrates how the recommendation strategy changes as the user's interaction history increases.

The system gives greater importance to content-based filtering for users with limited interactions and gradually increases the contribution of collaborative filtering for users with more interactions.

---

## 10. Cold-Start Evaluation

The cold-start experiment uses selected MovieLens users who originally have sufficient ratings.

Only **3 ratings are exposed** to the recommendation system.

The remaining ratings are hidden and treated as future interactions.

This allows us to simulate:

```text
New User
   ↓
Only 3 known interactions
   ↓
Generate Top-10 Recommendations
   ↓
Compare with hidden liked movies
   ↓
Calculate Precision@10
Recall@10
NDCG@10
```

This is a **simulated cold-start experiment**, not naturally occurring cold-start data.

---

## 11. Dynamic Weighting Strategy

The dynamic weighting mechanism is one of the main features of the project.

For example:

### User with 1 interaction

```text
CF Weight      = 0.17
Content Weight = 0.83
```

The system relies mainly on movie similarity.

### User with 10 interactions

```text
CF Weight      = 0.67
Content Weight = 0.33
```

The system now relies more heavily on collaborative filtering.

### User with 100 interactions

```text
CF Weight      = 0.95
Content Weight = 0.05
```

The system strongly favors collaborative filtering because sufficient interaction data is available.

This creates a smooth transition instead of using a fixed weighting scheme.

---

## 12. Failure Case Analysis

The project also identifies the worst-performing cold-start user based on NDCG@10.

For this user, the following are inspected:

* Known movies
* Recommended movies
* Hidden movies that the user actually liked

This helps identify cases where the system fails to understand the user's preferences.

Possible causes include:

* Only three known interactions are available.
* The user's preferences may span very different genres.
* Genre information alone may not capture the user's actual taste.
* Popularity-based fallback can favor generally popular movies rather than highly personalized choices.
* Limited metadata reduces the effectiveness of content-based filtering.

---

## 13. Limitations

The current implementation has several limitations:

1. Movie descriptions, actors, directors, and tags are not currently used.
2. Content-based filtering mainly uses movie genres.
3. The cold-start scenario is simulated because MovieLens 1M does not naturally contain users with fewer than five interactions.
4. The cold-start collaborative component uses popularity as a fallback rather than personalized SVD.
5. The recommendation system has not been deployed as a production API or web application.
6. The current implementation evaluates a selected sample of users for computational efficiency.

---

## 14. Future Improvements

Possible improvements include:

* Use movie descriptions and tags for richer content embeddings.
* Use neural embeddings instead of only TF-IDF.
* Experiment with ALS or other collaborative filtering algorithms.
* Develop a more sophisticated cold-start strategy.
* Use popularity calculated strictly from training data.
* Tune the dynamic weighting parameter using validation data.
* Build a Streamlit or web-based recommendation interface.
* Deploy the model using AWS services.
* Add real-time user interaction updates.
* Experiment with larger datasets.
* Perform hyperparameter tuning and cross-validation.

---

## 15. Project Structure

```text
hybrid-recommendation-engine/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── notebook/
│   └── hybrid_recommender.ipynb
│
└── results/
    ├── performance.png
    └── dynamic_weighting.png
```

---

## 16. Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project directory:

```bash
cd hybrid-recommendation-engine
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

For Google Colab, the notebook already contains the required installation command for `scikit-surprise`.

---

## 17. How to Run

### Google Colab

1. Download or clone this repository.
2. Open `notebook/hybrid_recommender.ipynb` in Google Colab.
3. Download the MovieLens 1M dataset.
4. Extract the dataset.
5. Upload:

   * `ratings.dat`
   * `movies.dat`
   * `users.dat`
6. Run the notebook from top to bottom.
7. View the recommendation results and evaluation metrics.
8. The notebook generates the performance and dynamic weighting graphs.

---

## 18. Example Recommendation Workflow

For a warm user:

```text
User History
     ↓
SVD Prediction + Content Similarity
     ↓
Dynamic Weight Calculation
     ↓
Hybrid Score
     ↓
Rank Movies
     ↓
Top-10 Recommendations
```

For a cold-start user:

```text
Limited User History
       ↓
Content Similarity
       +
Popularity Fallback
       ↓
Dynamic Weighting
       ↓
Hybrid Score
       ↓
Top-10 Recommendations
```

---

## 19. Reproducibility

To reproduce the experiment:

1. Use the MovieLens 1M dataset.
2. Install the dependencies from `requirements.txt`.
3. Upload the required dataset files to the Colab environment.
4. Run all notebook cells sequentially.
5. Use the generated metrics and graphs to verify the results.

The project uses fixed random seeds where applicable to make the experimental setup more reproducible.

---

## 20. Conclusion

This project demonstrates how a recommendation system can combine different recommendation strategies instead of relying on a single algorithm.

The key idea is **adaptive recommendation**:

> When user information is limited, rely more on content. As interaction history grows, rely more on collaborative filtering.

This approach provides a practical framework for handling both **cold-start and warm-user recommendation scenarios**.

---

## 21. Author

**Kunal Kotwani**

B.Tech Computer Science and Engineering
VIT Vellore

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
