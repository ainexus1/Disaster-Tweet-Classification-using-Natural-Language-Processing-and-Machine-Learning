# Twitter Disaster Prediction

## Overview
This project focuses on predicting whether a tweet is related to a disaster or not, utilizing various natural language processing (NLP) techniques and machine learning models. The dataset used contains labeled tweets, and the goal is to build a robust classifier that can accurately distinguish between tweets related to disasters and those that are not.

## Features
- **Data Cleaning:** The project involves thorough data cleaning, including removing URLs, emails, and special characters, as well as handling duplicates and accents.
- **Exploratory Data Analysis (EDA):** EDA is performed using visualizations, including word count distributions and average word length comparisons between disaster and non-disaster tweets.
- **WordCloud Visualization:** WordClouds are generated to visualize the most frequent words in disaster and non-disaster tweets separately.
- **Text Vectorization:** Different text vectorization techniques are explored, including TF-IDF and word embeddings using spaCy.
- **Machine Learning Models:** Linear Support Vector Classifier (LinearSVC) and a deep learning model with Convolutional Neural Network (CNN) architecture are employed for classification tasks.
- **BERT Model:** A BERT-based model is implemented using the ktrain library for more advanced natural language understanding and classification.
- **Performance Metrics:** The models' performance is evaluated using accuracy, precision, recall, and F1-score metrics.

## Dependencies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- TensorFlow
- Scikit-learn
- spaCy
- WordCloud
- ktrain

## Usage

1. **Run Jupyter Notebooks:**
    - Execute the Jupyter notebooks in the given order to perform data analysis, cleaning, and model training.

2. **BERT Model Training:**
    - Train the BERT-based model using ktrain and evaluate its performance.

3. **Run SVM Model:**
    - Train and evaluate the Linear Support Vector Classifier (SVM) model.

4. **Run CNN Model:**
    - Train and evaluate the Convolutional Neural Network (CNN) model.

5. **Explore Results:**
    - Analyze the results, compare model performances, and choose the most suitable model for your use case.

## Future Enhancements
- Experiment with different deep learning architectures and hyperparameters.
- Fine-tune the BERT model for better performance.
- Deploy the best-performing model to production for real-time predictions.

Feel free to contribute, report issues, or suggest improvements!
