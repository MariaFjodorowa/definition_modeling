import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="Unbabel/TowerInstruct-7B-v0.2", torch_dtype=torch.bfloat16, device_map="auto")
# We use the tokenizer’s chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {"role": "user",
      "content": "Пример вопроса и ответа: Вопрос - Объедини определения слова барсук в одно с таким же смыслом: хищное животное; "
      "род млекопитающих, обитающий в лесах; "
      "род птиц, обитающих в лесах; "
      "хищное животное, обитающее в лесу; "
      "хищное животное, обитающее в норах.  Ответ - барсук - хищное животное, обитающее в лесу в норах."
      "Вопрос - Объедини определения слова чайник в одно с таким же смыслом: "
      "Жарг., пренебр. о плохом, неопытном в каком-либо деле человеке; "
      "Разг., неодобр. тот, кто профессионально занимается какой-либо деятельностью, часто тайно; "
      "Воен. жарг., пренебр. о курсантах, не приняв присягу.  Ответ - чайник - "
      },
]
# messages = [
#   {"role": "user", "content": "Суммаризируй следующие предложения: хищное животное; "
#        "род млекопитающих, обитающий в лесах; "
#        "род птиц, обитающих в лесах; "
#        "хищное животное, обитающее в лесу; "
#        "хищное животное, обитающее в норах. "
# }
# ]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
print(outputs[0]["generated_text"])