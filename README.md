# song-popularity-analysis
A Probability &amp; Statistics project analyzing Spotify song popularity using audio features. Includes summary statistics, boxplots, scatter/correlation analysis, Multiple Linear Regression, Random Forest prediction, model comparison, and visualizations using Python.


# Spotify Song Popularity Prediction

This project analyzes and predicts Spotify song popularity using audio features from the Kaggle Spotify Tracks Dataset. The dependent variable is `popularity`, while the independent variables are `danceability`, `energy`, `loudness`, `valence`, `tempo`, and `duration_min`.

The project was developed for a Probability and Statistics course and includes descriptive statistics, visualizations, Multiple Linear Regression, and a Random Forest machine learning model.

---

## Project Objective

The main objective of this project is to study how different audio features affect the Spotify popularity score of a song.

Research Question:

**To what extent do danceability, energy, loudness, valence, tempo, and duration explain Spotify Song Popularity Score?**

---

## Dataset

Source: Kaggle Spotify Tracks Dataset

The dataset contains Spotify track information and audio features. For this project, 500 observations were selected after cleaning the data.

### Dataset Split

| Data Part | Observations |
|---|---:|
| Training Set | 400 |
| Test Set | 100 |
| Total | 500 |

---

## Variables Used

| Variable | Type | Description | Expected Effect |
|---|---|---|---|
| popularity | Dependent | Spotify popularity score from 0 to 100 | N/A |
| danceability | Independent | Suitability of a song for dancing, from 0 to 1 | Positive |
| energy | Independent | Intensity and activity level of a song, from 0 to 1 | Positive |
| loudness | Independent | Overall loudness in decibels | Positive |
| valence | Independent | Musical positiveness or mood, from 0 to 1 | Ambiguous |
| tempo | Independent | Estimated speed of the song in BPM | Ambiguous |
| duration_min | Independent | Track length in minutes | Negative |

---

## Tasks Covered

### Task 1: Data Collection and Cleaning

The dataset is loaded from `spotify.csv`. Required variables are selected, missing values are removed, zero-popularity rows are excluded, and duration is converted from milliseconds to minutes.

The cleaned dataset is saved as:

```text
spotify_data.xlsx
