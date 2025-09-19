Odia Turn Detection (QLoRA Fine-tuned)
📌 Overview

This project builds a Turn Detection model for the Odia language to classify conversational utterances into three categories:

F (Finished) → Speaker’s sentence is complete.
U (Unfinished) → Speaker’s sentence is incomplete or cut off.
W (Waiting) → Speaker indicates a pause or waiting turn.
Turn detection is a crucial step in dialogue systems, conversational AI, and speech processing. Since Odia is a low-resource language, this project contributes by creating a dataset and fine-tuning a transformer model for this task.

📊 Dataset
Language: Odia
Classes: Finished (F), Unfinished (U), Waiting (W)
Data Source: Custom-created sentences based on real-world conversations.

Preprocessing:
Removed duplicates & missing values
Normalized text format
Encoded labels as integers (W=0, U=1, F=2)

🏗️ Methodology

Dataset Preparation
Split into train/validation/test sets (80/10/10).
Converted to Hugging Face Dataset format.

Model Selection
Base model: livekit/turn-detector
Task: Sequence Classification (num_labels=3)
Fine-Tuning with QLoRA
Used LoRA (Low-Rank Adaptation) adapters.
4-bit quantization with BitsAndBytesConfig for memory efficiency.
Fine-tuned using Hugging Face Trainer.

Evaluation:
Metric: Accuracy
Compared predictions with gold labels for F/U/W.

⚙️ Training Setup:
Frameworks: PyTorch, Hugging Face Transformers, PEFT, Datasets
Batch Size: 16
Epochs: 3
Learning Rate: 2e-5

Optimizer: AdamW with weight decay
Hardware: Designed to run on GPUs with limited memory using QLoRA

📈 Results:
Model successfully learned to classify Odia conversational turns into F/U/W categories.
Demonstrates the potential of efficient fine-tuning for low-resource language NLP.
Training loss = 0.234500, 	validation loss = 0.228927, and accuracy = 0.973000

▶️ How to Run:
Clone the repo:
git clone https://github.com/your-username/odia-turn-detection-qlora.git
cd odia-turn-detection-qlora


Install dependencies:
pip install -r requirements.txt

Train the model:
python train.py


Run inference:
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "your-username/odia-turn-detection-qlora"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "ମୁଁ ଆସୁଛି"  # Odia sentence
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1).item()
print("Predicted label:", prediction)  # 0=W, 1=U, 2=F

🚀 Future Work
Extend dataset with real conversational transcripts.
Apply to speech-to-text pipelines for live dialogue systems.
Experiment with multilingual pre-trained models for cross-lingual transfer.

📌 Tech Stack
Python
PyTorch
Hugging Face Transformers
PEFT (LoRA)
BitsAndBytes (4-bit quantization)

✨ Acknowledgements
This project is inspired by ongoing work in dialogue systems and low-resource NLP. Special thanks to the open-source community for tools and frameworks.
