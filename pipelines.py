import os
import pandas as pd
from transformers import pipeline

# Project dataset folder
DATA_DIR = "dataset"
os.makedirs(DATA_DIR, exist_ok=True)

# Load input article
input_file = os.path.join(DATA_DIR, "input_text.csv")
df = pd.read_csv(input_file)

if df.empty or "text" not in df.columns:
    raise ValueError("input_text.csv must contain a non-empty 'text' column.")

article = str(df.iloc[0]["text"])
title = str(df.iloc[0].get("article_title", "Article"))

print("Loading Hugging Face pipelines...")
print("The first run may take time to download models.")

# 1. Sentiment Analysis
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
sentiment = sentiment_pipe(
    article[:1500],
    truncation=True,
    max_length=512
)[0]

sentiment_df = pd.DataFrame([{
    "article_title": title,
    "sentiment": sentiment["label"],
    "score": sentiment["score"]
}])
sentiment_df.to_csv(
    os.path.join(DATA_DIR, "sentiment_results.csv"),
    index=False
)
print("\nSentiment:", sentiment)

# 2. Text Summarization
summary_pipe = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

summary_short = summary_pipe(
    article,
    max_length=60,
    min_length=10,
    do_sample=False,
    truncation=True
)[0]["summary_text"]
summary_long = summary_pipe(
    article,
    max_length=100,
    min_length=20,
    do_sample=False,
    truncation=True
)[0]["summary_text"]

summary_df = pd.DataFrame([
    {"article_title": title, "max_length": 60,
     "summary": summary_short},
    {"article_title": title, "max_length": 100,
     "summary": summary_long}
])
summary_df.to_csv(
    os.path.join(DATA_DIR, "summarized_text.csv"),
    index=False
)
print("\nShort summary:\n", summary_short)
print("\nLong summary:\n", summary_long)

# 3. Translation: English to French and Spanish
translation_results = []

for language, model_name in [
    ("French", "Helsinki-NLP/opus-mt-en-fr"),
    ("Spanish", "Helsinki-NLP/opus-mt-en-es")
]:
    translator = pipeline(
        "translation",
        model=model_name
    )
    translated = translator(article[:1500])[0]["translation_text"]

    translation_results.append({
        "article_title": title,
        "language": language,
        "original_text": article[:1500],
        "translated_text": translated
    })

    print(f"\nTranslation to {language}:\n", translated)

pd.DataFrame(translation_results).to_csv(
    os.path.join(DATA_DIR, "translated_text.csv"),
    index=False
)

# 4. Question Answering
qa_pipe = pipeline(
    "question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad"
)

# Edit these questions to match your selected article.
questions = [
    "What is the main topic of the article?",
    "What are the main benefits discussed?",
    "What challenges are mentioned?",
    "Who can benefit from the topic discussed?",
    "What is important for responsible use?"
]

qa_results = []

for question in questions:
    result = qa_pipe(
        question=question,
        context=article[:4000]
    )

    qa_results.append({
        "article_title": title,
        "question": question,
        "answer": result["answer"],
        "score": result["score"]
    })

    print("\nQuestion:", question)
    print("Answer:", result["answer"])

pd.DataFrame(qa_results).to_csv(
    os.path.join(DATA_DIR, "qa_results.csv"),
    index=False
)

# 5. Compare pipeline outputs
comparison = pd.DataFrame([
    {
        "task": "Original Text",
        "output": article
    },
    {
        "task": "Summarization (max_length=60)",
        "output": summary_short
    },
    {
        "task": "Summarization (max_length=100)",
        "output": summary_long
    },
    {
        "task": "Sentiment Analysis",
        "output": sentiment["label"]
    },
    {
        "task": "French Translation",
        "output": translation_results[0]["translated_text"]
    },
    {
        "task": "Spanish Translation",
        "output": translation_results[1]["translated_text"]
    },
    {
        "task": "Question Answering",
        "output": " | ".join(
            item["question"] + ": " + item["answer"]
            for item in qa_results
        )
    }
])

comparison.to_csv(
    os.path.join(DATA_DIR, "pipeline_comparison.csv"),
    index=False
)

print("\nAll NLP tasks completed successfully.")
print("CSV results saved in the dataset folder.")