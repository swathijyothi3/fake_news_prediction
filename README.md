# FakeScope AI

**Fake News Detection System using Natural Language Processing**

---

## Overview

FakeScope AI is a machine learning application that classifies news articles as real or fake. It is trained on the ISOT Fake News Dataset and offers two text vectorisation strategies — Bag of Words and TF-IDF — each paired with a K-Nearest Neighbours classifier, served through a Streamlit web interface.

---

## Dataset

| File       | Description                                            | Articles |
|------------|--------------------------------------------------------|----------|
| `Fake.csv` | Fabricated articles sourced from unreliable websites   | 23,481   |
| `True.csv` | Genuine articles from Reuters                          | 21,417   |

Both files contain the columns: `title`, `text`, `subject`, `date`.  
Labels are assigned as `0` (fake) and `1` (real).  
After removing duplicates, the working dataset contains approximately 39,000 articles.

---

## Project Structure

```
fakescope/
├── app.py               Web application (Streamlit)
├── NLP_Project.ipynb    Training notebook — data prep, modelling, evaluation
├── bow_model.pkl        Trained BOW + KNN bundle
├── tfidf_model.pkl      Trained TF-IDF + KNN bundle
├── requirements.txt     Python dependencies
├── Fake.csv             Fake news dataset
├── True.csv             Real news dataset
└── README.md            This file
```

---

## Setup and Installation

**Prerequisites:** Python 3.10 or higher

### Step 1 — Clone or download the project

Place all files in the same directory.

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — (Optional) Retrain the models

Open and run `NLP_Project.ipynb` from top to bottom. This will regenerate `bow_model.pkl` and `tfidf_model.pkl` using your local environment.

### Step 4 — Launch the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## How to Use the App

1. **Select a vectoriser** using one of the two buttons at the top of the input panel.
2. Enter an optional news headline and paste the article body into the text area.
3. Click **Analyze Article**.
4. The result card displays whether the article is classified as REAL or FAKE, along with a confidence score.
5. The word cloud and top-words chart show which terms drove the classification.

---

## Vectorisation Options

| Option | Name           | Description |
|--------|----------------|-------------|
| 1      | BOW (Bag of Words) | Counts how many times each of the top 5,000 words appears in the article. Simple and interpretable. Test accuracy: ~80%. |
| 2      | TF-IDF         | Weights each word by how unique it is across the entire training corpus. Rare but meaningful words receive higher scores. Test accuracy: ~70%. |

---

## Text Preprocessing Pipeline

Each article passes through the following steps before classification:

1. Lowercase conversion
2. URL and HTML tag removal
3. Non-alphabetic character removal
4. Stopword removal (NLTK English stopwords)
5. Lemmatisation using NLTK WordNetLemmatizer

---

## Model Performance

| Model        | Accuracy | Precision (avg) | Recall (avg) | F1 Score (avg) |
|--------------|----------|-----------------|--------------|----------------|
| BOW + KNN    | ~80%     | ~80%            | ~80%         | ~80%           |
| TF-IDF + KNN | ~70%     | ~70%            | ~70%         | ~70%           |

Evaluated on a held-out test set (20% of the full dataset, ~7,800 articles).

---

## Limitations

- The model was trained exclusively on English-language political news from 2015 to 2018. Performance on other domains (sport, science, entertainment) or more recent articles may be lower.
- KNN is a distance-based method and works well on this dataset but does not provide direct feature importance. Use the word cloud and frequency chart in the app to inspect which terms are present in the input.
- The confidence score reflects the proportion of the 5 nearest neighbours belonging to each class — it is not a calibrated probability.

---

## Dependencies

| Package        | Purpose                            |
|----------------|------------------------------------|
| streamlit      | Web interface                      |
| scikit-learn   | Vectorisation and KNN classifier   |
| nltk           | Stopwords, lemmatisation           |
| numpy / pandas | Numerical computing and data handling |
| matplotlib     | Word cloud rendering               |
| wordcloud      | Word cloud generation              |
| plotly         | Confidence gauge and bar charts    |
| Pillow         | Image processing (required by wordcloud) |
| joblib         | Model serialisation                |

---

## License

This project is for educational and portfolio purposes.  
The ISOT Fake News Dataset was created by the Information Security and Object Technology (ISOT) Research Lab at the University of Victoria.
