import concurrent.futures
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import openai
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

CWD = os.path.abspath(os.path.dirname(__file__))
load_dotenv(override=True)
# Setting API parameters
openai.api_base = "https://api.deepseek.com"
openai.api_key = os.environ.get("OPENAI_API_KEY")
dataset = load_dataset("openai_humaneval", split="test")
dataset = [entry for entry in dataset]

prompt_path = f"{CWD}/prompts/coder_prompt_humaneval.txt"
with open(prompt_path, "r") as f:
    construct_few_shot_prompt = f.read()


def preprocess_data(completion_string: str, language: str = "python"):
    """Get the code block from the LLM answer

    Args:
        completion_string (str): LLM answer

    Returns:
        str: code block
    """
    if f"```{language}" in completion_string:
        completion_string = completion_string[
            completion_string.find(f"```{language}") + len(f"```{language}") :
        ]
        completion_string = completion_string[: completion_string.find("```")]
    else:
        print("Error: No code block found")
    return completion_string


def fetch_completion(
    data_entry: dict, model: str, language: str = "python", times: int = 5
):
    global construct_few_shot_prompt
    if (
        "need_reproduce" in data_entry.keys()
        and data_entry["need_reproduce"] == False
    ):
        return data_entry
    prompt = data_entry["prompt"]
    text = f"""
{construct_few_shot_prompt}

**Input Code Snippet**:
```{language}
{prompt}
```
## Completion 3:
"""
    completions_code = []
    for i in range(times):
        while True:
            try:
                completions = openai.ChatCompletion.create(
                    model=model,
                    stream=False,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a software programmer.",
                        },
                        {"role": "user", "content": text},
                    ],
                    request_timeout=100,
                )
                completion = completions.choices[0]["message"]["content"]
                completion = preprocess_data(completion)
            except Exception as e:
                print(repr(e))
                time.sleep(10)
                completion = ""
            if completion != "":
                break
        completions_code.append(completion)
    data_entry["completion_list"] = completions_code
    return data_entry


def call_fetch_completion_helper(
    dataset: List[dict], model: str, language: str
):
    print("Fixing bug...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_entry = {
            executor.submit(
                fetch_completion, copy.deepcopy(entry), model, language
            ): entry
            for entry in tqdm(dataset, total=len(dataset))
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_entry),
            desc="Updating code",
            total=len(dataset),
        ):
            entry = future_to_entry[future]
            try:
                updated_entry = future.result()
                idx = dataset.index(entry)
                dataset[idx] = updated_entry
            except Exception as e:
                print(repr(e))
    return dataset


if __name__ == "__main__":
    model = "deepseek-coder"
    language = "python"

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_entry = {
            executor.submit(
                fetch_completion, copy.deepcopy(entry), model, language, 1
            ): entry
            for entry in tqdm(dataset)
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
            entry = future_to_entry[future]
            try:
                updated_entry = future.result()
                idx = dataset.index(entry)
                dataset[idx] = updated_entry
            except Exception as e:
                print(repr(e))
    print("Done!")
    with open(f"{CWD}/data/{model}_{language}.json", "w") as f:
        json.dump(dataset, f, indent=4)
