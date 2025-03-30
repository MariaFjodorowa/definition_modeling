import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="Unbabel/TowerInstruct-7B-v0.2", torch_dtype=torch.bfloat16, device_map="auto")
# We use the tokenizer’s chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {"role": "user",
      "content": "Объедини определения слова барсук в одно с таким же смыслом: хищное животное; "
      "род млекопитающих, обитающий в лесах; ",
      "род птиц, обитающих в лесах; ",
      "хищное животное, обитающее в лесу; ",
      "хищное животное, обитающее в норах. "
      },
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
print(outputs[0]["generated_text"])