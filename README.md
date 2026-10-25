# Week 7 – Hugging Face Pipelines

## 1. Project Overview

This project demonstrates different Natural Language Processing (NLP) tasks using pre-trained models from Hugging Face Transformers. The project uses an article about the benefits of Artificial Intelligence (AI) as its input dataset.

## 2. Objectives

- Perform text summarization.
- Translate English text into French and Spanish.
- Perform sentiment analysis.
- Answer questions based on the article.
- Compare the outputs of different NLP pipelines.
- Save the results in CSV files for further analysis.

## 3. Technologies Used

- Python
- Hugging Face Transformers
- PyTorch
- Pandas
- Jupyter Notebook

## 4. NLP Tasks Performed

### Text Summarization
Generates short and long summaries of the input article using a pre-trained summarization model.

### Translation
Translates the English article into French and Spanish using pre-trained translation models.

### Sentiment Analysis
Identifies the sentiment of the article and produces a sentiment label and confidence score.

### Question Answering
Answers questions based on the information available in the input article.

### Pipeline Output Comparison
Compares the original article with the outputs produced by the different NLP tasks.

## 5. Dataset

The input dataset is stored in `dataset/input_text.csv`. It contains an article titled "Benefits of Artificial Intelligence".

## 6. Project Structure

- `pipelines.py` – Main Python program for running the NLP tasks.
- `dataset/input_text.csv` – Input article.
- `dataset/sentiment_results.csv` – Sentiment analysis results.
- `dataset/summarized_text.csv` – Text summarization results.
- `dataset/translated_text.csv` – French and Spanish translations.
- `dataset/qa_results.csv` – Question-answering results.
- `dataset/pipeline_comparison.csv` – Comparison of pipeline outputs.
- `notebook/Week7_HuggingFace_Pipelines.ipynb` – Jupyter Notebook.
- `requirements.txt` – Required Python libraries.

## 7. How to Run the Project

1. Open the project folder in Visual Studio Code.
2. Install the required libraries using:

   `pip install -r requirements.txt`

3. Run the Python program:

   `python pipelines.py`

4. Wait for the NLP tasks to finish.
5. Open the `dataset` folder to view the generated CSV results.

## 8. Results

The program generates CSV files containing sentiment analysis results, short and long summaries, French and Spanish translations, question-answering results, and a comparison of the different NLP outputs.

## 9. Conclusion

This project demonstrates how Hugging Face Pipelines can perform different Natural Language Processing tasks using pre-trained models. The generated CSV files store the results for analysis and comparison. This project provides practical experience in using transformer-based models for text processing.