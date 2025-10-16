#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./models/simple_chatbot"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

print("🏥 Healthcare Chatbot Ready!")
print("Type 'quit' to exit")

while True:
    user_input = input("\nPatient: ")
    if user_input.lower() == 'quit':
        break
    
    prompt = f"Patient: {user_input}\nDoctor:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
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
    print(f"Doctor: {doctor_response}")
