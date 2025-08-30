# Text Similarity Detection with Siamese Networks

## 📋 Project Overview

This project implements a deep learning solution for detecting duplicate questions using Siamese neural networks. The model is trained on the Quora Question Pairs dataset to identify whether two questions are semantically similar or not.

## 🎯 Problem Statement

The goal is to build a model that can determine if two questions are duplicates, which is valuable for:
- Reducing redundancy in Q&A platforms
- Improving search functionality
- Enhancing user experience by grouping similar questions

## 📊 Dataset

The project uses the **Quora Question Pairs dataset** from Kaggle, which contains:
- Over 400,000 question pairs
- Binary labels indicating whether questions are duplicates
- Various question topics and styles

## 🛠️ Implementation Steps

### 1. Data Acquisition and Preparation
- Downloaded the dataset from Kaggle using the official API
- Extracted and loaded the training data
- Performed initial data exploration

### 2. Data Preprocessing
- Removed unnecessary columns (`id`, `qid1`, `qid2`)
- Handled missing values and duplicates
- Applied text cleaning:
  - Converted to lowercase
  - Removed special characters, URLs, and HTML tags
  - Eliminated numbers and extra whitespace

### 3. Data Analysis and Balancing
- Analyzed class distribution (duplicate vs non-duplicate questions)
- Balanced the dataset to address class imbalance
- Split data into training and testing sets (80/20 split)

### 4. Text Tokenization
- Created a tokenizer with 10,000 word vocabulary
- Converted questions to sequences
- Padded sequences to ensure uniform length

### 5. Model Architecture (Siamese Network)

The Siamese network consists of:
- **Embedding Layer**: Converts words to 128-dimensional vectors
- **Bidirectional GRU Layer**: Captures contextual information from both directions
- **Dense Layers**: Process concatenated question representations
- **Dropout Layer**: Prevents overfitting
- **Output Layer**: Sigmoid activation for binary classification

### 6. Model Training
- Compiled with Adam optimizer and binary cross-entropy loss
- Trained for 5 epochs
- Achieved 78% accuracy on the test set

## 🧠 Key Features

1. **Siamese Architecture**: Uses shared weights to process both questions simultaneously
2. **Bidirectional GRU**: Captures context from both directions for better understanding
3. **Text Preprocessing**: Comprehensive cleaning for better model performance
4. **Class Balancing**: Addressed dataset imbalance for improved learning

## 📈 Performance

The model achieved:
- Training accuracy: 92.3%
- Validation accuracy: 78.1%
- Test accuracy: 78.1%

## 💡 Potential Improvements

1. Experiment with different embedding techniques (Word2Vec, GloVe, BERT)
2. Try alternative architectures (CNN, Transformer-based models)
3. Implement more sophisticated attention mechanisms
4. Use hyperparameter tuning to optimize performance
5. Apply more advanced text preprocessing and feature engineering

## 🚀 How to Use

1. Ensure you have the required dependencies (TensorFlow, pandas, numpy)
2. Download the Quora dataset from Kaggle
3. Run the notebook cells in sequence
4. The model will train and evaluate automatically
5. Use the trained model to predict similarity for new question pairs

## 📚 Dependencies

- TensorFlow 2.x
- pandas
- numpy
- scikit-learn
- Kaggle API

This project demonstrates the application of deep learning and natural language processing techniques to solve real-world text similarity problems, with potential applications in search engines, Q&A platforms, and content recommendation systems.
