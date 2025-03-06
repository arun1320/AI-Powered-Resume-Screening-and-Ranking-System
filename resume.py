# Install required libraries
!pip install spacy PyPDF2 scikit-learn python-docx
!python -m spacy download en_core_web_sm

# Import necessary libraries
import os
import PyPDF2
from docx import Document
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import numpy as np
import spacy
from google.colab import drive

# Mount Google Drive to access files
drive.mount('/content/drive')

# Define paths for data directories
data_dir = "/content/drive/My Drive/data/"
resumes_dir = os.path.join(data_dir, "resumes")  # Folder containing resumes
jobs_path = os.path.join(data_dir, "jobs - Sheet1.csv")  # CSV file containing job descriptions
print("Files in data directory:", os.listdir(data_dir))

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    doc = Document(docx_path)
    return ' '.join([para.text for para in doc.paragraphs])

# Extract text from all resumes in the resumes directory
resume_texts = []
for file in os.listdir(resumes_dir):
    file_path = os.path.join(resumes_dir, file)
    if file.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file.endswith('.docx'):
        text = extract_text_from_docx(file_path)
    else:
        continue  # Skip non-PDF/DOCX files
    resume_texts.append((file, text))  # Store filename and text

# Print extracted resumes (first 200 characters for brevity)
print("\nExtracted Resumes:")
for file, text in resume_texts:
    print(f"File: {file}")
    print(f"Text: {text[:200]}...")  # Print first 200 characters of each resume
    print("-" * 50)

# Load job descriptions from the CSV file
jobs_df = pd.read_csv(jobs_path)
print("\nJob Descriptions:")
print(jobs_df.head())

# Prepare data for BERT by pairing each resume with each job description
data = []
for file, resume_text in resume_texts:
    for _, job_row in jobs_df.iterrows():
        data.append({
            'resume_file': file,
            'resume_text': resume_text,
            'job_desc_text': job_row['job_description'],
            'label': 1  # Assume all pairs are matches for now
        })

# Convert the data to a DataFrame
data_df = pd.DataFrame(data)
print("\nData for BERT:")
print(data_df.head())

# Load BERT tokenizer and tokenize the inputs
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(
    data_df['resume_text'].tolist(),
    data_df['job_desc_text'].tolist(),
    padding=True,
    truncation=True,
    return_tensors="pt",
    max_length=512
)
print("\nTokenized Inputs:")
print(inputs)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Prepare labels (all set to 1 for now)
labels = torch.tensor(data_df['label'].tolist())

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Train the BERT model for 3 epochs
print("\nTraining BERT Model...")
for epoch in range(3):  # 3 epochs for example
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch + 1}: Loss {loss.item()}")

# Generate predictions using the trained model
print("\nGenerating Predictions...")
with torch.no_grad():
    logits = model(**inputs).logits
    scores = torch.softmax(logits, dim=1)[:, 1].numpy()  # Probability of match

# Rank resumes based on their scores
ranked_indices = np.argsort(scores)[::-1]  # Sort in descending order
ranked_resumes = [data_df.iloc[i] for i in ranked_indices]

# Load SpaCy model for skill extraction
nlp = spacy.load("en_core_web_sm")

# Function to extract skills from resume text using SpaCy
def extract_skills(text):
    """Extract skills from resume text using SpaCy."""
    doc = nlp(text)
    skills = [token.text for token in doc if token.ent_type_ == "SKILL"]
    return skills

# Display the top 5 ranked resumes with their skills
print("\nTop 5 Ranked Resumes:")
for i, idx in enumerate(ranked_indices[:5]):
    resume_data = data_df.iloc[idx]
    skills = extract_skills(resume_data['resume_text'])
    print(f"Rank {i + 1} (Score: {scores[idx]:.2f}):")
    print(f"Resume File: {resume_data['resume_file']}")
    print(f"Skills: {', '.join(skills)}")
    print(f"Resume Text: {resume_data['resume_text'][:200]}...\n")  # Display first 200 characters
