from openai import OpenAI
import jsonlines
import random
import json 
n = 800
def load_seed_tasks(file_path):
    seed_tasks = []
    with jsonlines.open(file_path, "r") as reader:
        for task in reader:
            seed_tasks.append(task)
    return seed_tasks

def generate_prompt():
    task = random.choice(load_seed_tasks("seed_tasks.jsonl"))
    prompt = """You are asked to come up with 5 task instructions. These task instructions will be given to an LLM and we will evaluate the LLM for completing the instructions. Generate 5 similar tasks in this format:"""
    prompt += "{"+ f"\"instruction\" : \"{task['instruction']}\","
    prompt += f"\"Input\": \"{task['instances'][0]['input']}\","
    prompt += f"\"Output\":\"{task['instances'][0]['output']}\""
    prompt+="}"
    return prompt

def save_model_output_to_json(output, filename = "output.json"):
    with open(filename, 'a') as file:
        #json.dump(output, file, indent=4)
        file.write(json.dumps(output,indent=4) + '\n')



client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
sysmsg = "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."

for i in range(n):
    print("Currently running the seed task: ",i)
    prompt = generate_prompt()
    completion = client.chat.completions.create(
      messages=[
        {"role": "system", "content": sysmsg},
        {"role": "user", "content": prompt}
      ],
      model = "phi-2.Q5_K_M.gguf",
      temperature=0.7,
    )
    
    #print(completion.choices[0].message.content)

    save_model_output_to_json(completion.choices[0].message.content)