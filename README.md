# AI-Driven Brand Trust & Sentiment Analyzer

## Overview

This project is a Python tool that analyzes customer complaints to find out why they are unhappy with a brand. It reads text reviews, calculates how negative or positive they are, figures out the main emotion behind them, and classifies them into specific business problems like technical failures or ethical issues.

It is designed to bridge the gap between software development and market research by automating how we understand customer frustration.

---

## Description

When customers leave bad reviews, it is usually for one of two main reasons: the product didn't work properly (**Competence Violation**), or the company treated them unfairly or lied to them (**Integrity Violation**).

This script uses Natural Language Processing (NLP) to read through text data, analyze the sentiment intensity using a library called VADER, and then use a custom keyword-matching approach to sort the complaints into categories. To make the data useful for business research, the project generates a realistic sample dataset modeled after real financial sector complaints, complete with company names, product categories, and text lengths. Finally, it builds a 6-panel visual dashboard showing the data trends.

---

## Features

* **Sentiment Scoring:** Measures how positive, negative, or neutral a review is on a scale from -1 to +1.
* **Trust Violation Sorting:** Automatically flags whether a complaint is about system bugs/crashes (Competence) or hidden fees/deception (Integrity).
* **Emotion Mapping:** Tracks four core consumer emotions: Anger, Fear, Sadness, and Trust.
* **Data Simulation:** Creates a structured, research-ready dataset with random company names and products using fixed data seeds so the results can be easily reproduced.
* **Visual Dashboard:** Generates a clean, 6-chart data dashboard and saves it automatically as a high-resolution PNG file.

---

## Tech Stack

* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **NLP & Text Analysis:** VADER Sentiment
* **Data Visualization:** Matplotlib, Seaborn

---

## Workflow

```
[Customer Review Text]
         │
         ▼
[VADER Sentiment Engine] ──► Calculates Sentiment Score
         │
         ▼ (If Score is Negative)
[Keyword Matching Logic] ──► Extracts Core Emotion (Anger, Fear, etc.)
         │               ──► Classifies Trust Issue (Integrity vs. Competence)
         │
         ▼
[Data Aggregator]        ──► Combines text analysis with company/product metadata
         │
         ▼
[Visualization Engine]   ──► Generates and saves 6-panel analytical dashboard

```

---

## Libraries

* `pandas`: For creating dataframes, cleaning data, and handling column outputs.
* `numpy`: For generating predictable, structured mock data using random choice selection.
* `vaderSentiment`: For extracting sentiment scores from text without needing complex neural network training.
* `matplotlib.pyplot`: For structuring the core grid layout of the final dashboard.
* `seaborn`: For styling the charts and making the bar, pie, and box plots easy to read.

---

## Installation and Setup

### 1. Clone or Open the Project Folder

Make sure all your files are inside a single directory on your machine.

### 2. Install Required Python Packages

Open your terminal or command prompt and run the following command to install all dependencies at once:

```bash
pip install pandas vaderSentiment matplotlib seaborn numpy

```

### 3. Run the Script

Execute the main script to process the data and generate the graphs:

```bash
python main.py

```

---

## Project Structure

```text
AI-Driven Sentiment & Brand Trust Violation Analysis/
│
├── main.py                         # The main Python script with data, logic, and plotting
├── trust_analysis_dashboard.png    # The generated 6-chart dashboard output (saved after running)
└── README.md                       # Documentation file

```

---

## Future Improvements

* **Real-World Scrapers:** Add a feature to pull real-time reviews directly from Twitter/X, Reddit, or Amazon using APIs.
* **Machine Learning Transition:** Train a custom classification model (like Random Forest or a small Transformer) on labeled data instead of relying on fixed keyword lists.
* **Interactive Dashboard:** Convert the static Matplotlib charts into an interactive web dashboard using Streamlit or Plotly Dash.
