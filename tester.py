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
load_dotenv()


def preprocess_data(test_case_string: str, language: str):
    if f"```{language}" in test_case_string:
        test_case_string = test_case_string[
            test_case_string.find(f"```{language}") + len(f"```{language}") :
        ]
        test_case_string = test_case_string[: test_case_string.find("```")]

    return test_case_string


# Function to fetch completion
def fetch_completion(
    data_entry: dict, model: str, language: str, times: int = 10
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
"""
    test_case_list = []
    for _ in range(times):
        while True:
            try:
                completions = openai.ChatCompletion.create(
                    model=model,
                    stream=False,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a code developer assistant.",
                        },
                        {"role": "user", "content": text},
                    ],
                    request_timeout=100,
                )
                test_case = completions.choices[0]["message"]["content"]
                test_case = preprocess_data(test_case)
            except Exception as e:
                time.sleep(20)
                print(e)
                test_case = ""
            if test_case != "":
                break
        test_case_list.append(test_case)
    data_entry["test_case_list"] = test_case_list
    return data_entry


def call_fetch_test_completion_helper(
    dataset: List[dict], model: str, language: str
):
    print("Fixing bug...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_entry = {
            executor.submit(
                fetch_completion, copy.deepcopy(entry), model, language
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
    return dataset


if __name__ == "__main__":
    # Setting API parameters
    openai.api_base = "https://api.deepseek.com"
    openai.api_key = os.getenv("OPENAI_API_KEY")

    dataset = load_dataset("openai_humaneval", split="test")
    dataset = [entry for entry in dataset]

    prompt_path = f"{CWD}/prompts/tester_prompt_humaneval.txt"
    with open(prompt_path, "r") as f:
        construct_few_shot_prompt = f.read()
    model = "deepseek-coder"
    language = "{language}"
    with open(f"{CWD}/data/{model}_{language}.json", "r") as f:
        dataset = json.load(f)
    dataset = [entry for entry in dataset]
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_entry = {
            executor.submit(
                fetch_completion, copy.deepcopy(entry), model, language
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

    with open(f"{CWD}/data/{model}_{language}.json", "w") as f:
        json.dump(dataset, f, indent=4)
