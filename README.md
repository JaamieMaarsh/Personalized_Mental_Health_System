# Personalized_Mental_Health_System


### Overview

This project aims to develop a Mental Health Q&A system that detects human emotions based on text input and provides relevant do's, don’ts, and remedies for specific mental health conditions. The system leverages NLP techniques, sentiment analysis, and Retrieval-Augmented Generation (RAG) using Langchain to fetch and summarize mental health resources.

### Dataset

The system uses a dataset containing 50,000 text records labeled with seven major emotions:

Normal

Depression

Suicidal

Anxiety

Stress

Bi-Polar

Personality Disorder

### Dataset Source

Additional documents for RAG contain information on symptoms, causes, diagnosis, treatment, and resources for mental health conditions:

National Institute of Mental Health (NIMH) Publications

Depression Fact Sheet

### Workflow

#### 1. User Input Processing

The user enters a text query describing their feelings or symptoms.

The input is tokenized and transformed into vector embeddings using Sentence Transformers to capture semantic meaning.

#### 2. Sentiment Analysis & Emotion Classification

A pre-trained LLM is fine-tuned for mental health classification.

The trained model predicts the most probable mental health category.

Performance metrics are evaluated to ensure accuracy.

#### 3. Retrieval-Augmented Generation (RAG) for Recommendations

Document embeddings are generated and stored in vector databases.

User query embeddings are matched with the closest document vectors to fetch relevant content.

Summarization techniques are applied to generate concise responses.

#### 4. Output Generation

The system provides structured recommendations based on the predicted mental health condition.

##### Example:

User: I am not able to make decisions over small things such as deciding over food.
Predicted Problem: Anxiety
Suggested Remedy: Ensure at least 8 hours of sleep and maintain a positive mindset.

### Implementation Steps

##### Data Preprocessing

Clean and normalize text input.

Tokenize and convert into vector embeddings.

##### Model Training & Evaluation

Fine-tune a pre-trained LLM on the mental health dataset.

Train using cross-entropy loss and evaluate using precision, recall, and F1-score.

##### Vector Database for RAG

Convert mental health documents into embeddings.

Store in a vector database (e.g., FAISS, Pinecone) for similarity retrieval.

Inference & Response Generation

Classify input into one of the seven mental health conditions.

Retrieve relevant mental health content.

Summarize and display user-friendly recommendations.

##### Future Improvements

Expand emotion categories for better granularity.

Enhance response personalization based on user history.

Integrate voice-based input for accessibility.

Develop a chatbot interface for real-time support.

#### Conclusion

This project leverages NLP, emotion classification, and RAG to assist users in understanding their mental health and accessing valuable resources. By fine-tuning LLMs and integrating retrieval-based recommendations, the system aims to provide personalized and meaningful mental health support.


