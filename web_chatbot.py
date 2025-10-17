# web_chatbot.py
import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "Amandine0610/simple_chatbot"  # or your actual model path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def clean_response(text):
    return text.replace('-->', '').replace('>', '').replace('<', '').replace('-', '').replace('  ', ' ').strip()

def is_in_domain(user_input):
    medical_keywords = [
        "symptom", "treatment", "doctor", "medicine", "pain", "fever", "diabetes", "heart", "blood", "health",
        "disease", "infection", "cough", "headache", "nurse", "hospital", "clinic", "prescription", "dose","virus",
        "bacteria", "allergy", "asthma", "cancer", "surgery", "therapy", "mental health","flu", "cold", "injury", 
        "emergency", "vaccination", "immunization", "nutrition", "exercise","eye", "ear", "nose", "throat", "skin", 
        "bone", "joint", "muscle", "pregnancy", "childbirth","infant", "child", "adolescent", "adult", "elderly", "geriatrics",
        "pediatrics","cardiology", "neurology", "orthopedics", "dermatology", "psychiatry","radiology", "pathology", "anatomy", 
        "physiology","pharmacology"
    ]
    return any(word in user_input.lower() for word in medical_keywords)

def chat_with_bot(user_input):
    if not is_in_domain(user_input):
        return "Sorry, I can only answer healthcare-related questions."
    
    prompt = f"Patient: {user_input}\nDoctor:"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    doctor_response = response.split("Doctor:")[-1].strip()
    doctor_response = clean_response(doctor_response)
    return doctor_response
custom_css = """
#component-0 {background: #f0f4f8;}
#component-1 textarea {background: #fffbe7; border-radius: 8px;}
#component-3 textarea {background: #e6f7fa; border-radius: 8px;}
#component-5 {background: #f0f4f8;}
.gr-button-primary {background: #ff6600; color: white;}
"""

iface = gr.Interface(
    fn=chat_with_bot,
    inputs=gr.Textbox(lines=2, label="Your Medical question"),
    outputs=gr.Textbox(label="Assistant"),
    title="🏥 Healthcare Chatbot",
        description=(
        "Ask a medical question and get a response from the chatbot.<br>"
        "<b>Note:</b> This chatbot provides general health information only. "
        "For emergencies or specific medical advice, consult a healthcare professional."
    ),
    theme="soft",
    css=custom_css,
    allow_flagging="never",
    examples=[
        ["What are the symptoms of diabetes?"],
        ["How can I prevent heart disease?"],
        ["I have a headache and fever, what should I do?"],
        ["How do I fix my car engine?"]
    ]
)


if __name__ == "__main__":
    # create the public link
    iface.launch( 
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7861)),
        share=False
        )
