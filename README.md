# Healthcare Chatbot

A healthcare question chatbot, built using a fine-tuned Hugging Face Transformer model. It is an end-to-end project from data preprocessing to deployment with generative question-answering for medical and health questions.

---
##  Project Overview

The healthcare chatbot is designed to:
- Parse and respond to health-related questions
- Provide relevant medical information and advice
- Reasonably reject out-of-domain questions
- Offer an intuitive web interface for interaction with users

*** Disclaimer ***: This chatbot provides general health information only. Always medical professionals should be consulted for medical advice, diagnosis, or treatment.

---
## Features

- **Domain-Specific Responses**: Trained on healthcare data only
- **Out-of-Domain Detection**: Politely rejects non-healthcare queries
- **Web Interface**: Simple Gradio-based UI for ease of interaction
- **Evaluation Metrics**: BLEU and ROUGE-L for quantitative assessment

---
## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional) or CPU

Install dependencies:
```sh
pip install -r requirements.txt
```

---

## Dataset

Project uses a healthcare Q&A dataset (`data/ai-medical-chatbot.csv`) with columns:
- `Description` (optional/context)
- `Patient` (user question)
- `Doctor` (reference answer)

obtained from kaggle datasets

---

## Training

Train the chatbot using:
```sh
python train.py
```
The model and tokenizer will be saved in `./models/simple_chatbot`.

---

## Usage

### Web Interface

Launch the Gradio web interface:
```sh
python web_chatbot.py
```
- Enter your health question in the textbox.
- The chatbot will respond with a relevant answer.
- Out-of-domain questions will be courteously declined.

---

## Evaluation

Evaluate your chatbot using BLEU and ROUGE-L:
```sh
python evaluate.py
```
**Results on example data:**
- **BLEU score:** 0.50
- **ROUGE-L F1:** 0.08

---

## Hyperparameter Tuning

| Hyperparameter | Tried Values      | Final Value | Notes                                      |
|----------------|------------------|-------------|--------------------------------------------|
| Model          | distilgpt2, gpt2 | distilgpt2  | Used distilgpt2 for faster training        |
| Learning rate  | 5e-5 (default)   | 5e-5        | Used default for AdamW optimizer           |
| Batch size     | 2, 4             | 4           | Chose 4 for balance of speed and memory    |
| Epochs         | 1, 3, 5          | 3           | 3 gave best qualitative results            |
| Max length     | 100, 150         | 100         | 100 tokens for generated responses         |
| Temperature    | 0.7, 1.0         | 0.7         | 0.7 for more focused, relevant answers     |
| do_sample      | True, False      | True        | Sampling enabled for more diverse outputs  |
| pad_token_id   | eos_token_id     | eos_token_id| Used to prevent warnings in generation       |

---
## Example Interactions

**Patient:** What are the symptoms of diabetes?
**Chatbot:** Some common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, and fatigue.

**Patient:** How do I fix my car engine?
**Chatbot:** Sorry, I can only answer healthcare-related questions.

---
## Limitations

- Only answers healthcare-related questions
- May not address rare diseases or emergencies
- BLEU and ROUGE-L scores are low due to limited data and model size

---
## Project Structure

```
Summative_Domain_specific_chatbot/
├── data/
│   └── ai-medical-chatbot.csv
├── models/
│   └── simple_chatbot/
├── train.py
├── web_chatbot.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---
## Acknowledgments

- **Hugging Face**: For the Transformers library and pre-trained models
- **Gradio**: For the web interface framework

---

**Happy Healthcare Chatbotting! **
