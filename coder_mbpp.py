import concurrent.futures
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List

import openai
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

CWD = os.path.abspath(os.path.dirname(__file__))
load_dotenv(override=True)

# Setting API parameters
openai.api_base = "https://api.deepseek.com"
openai.api_key = os.environ.get("OPENAI_API_KEY")

prompt_path = f"{CWD}/prompts/coder_prompt_mbpp.txt"
with open(prompt_path, "r") as f:
    construct_few_shot_prompt = f.read()

dataset = load_dataset("mbpp", name="sanitized", split="test")
dataset = [entry for entry in dataset]


def preprocess_data(data: dict, language: str = "python"):
    if f"```{language}" in data["completion"]:
        data["completion"] = data["completion"][
            data["completion"].find(f"```{language}") + len(f"```{language}") :
        ]
        data["completion"] = data["completion"][
            : data["completion"].find("```")
        ]
    else:
        print(data["task_id"])
    return data


# Function to fetch completion
def fetch_completion(data_entry: dict, model: str, language: str = "python"):
    global construct_few_shot_prompt
    language = "python"
    if "passed" in data_entry.keys() and data_entry["passed"] == True:
        return data_entry
    prompt = data_entry["prompt"]
    test_case = data_entry["test_list"]
    tests = ""
    # for test in test_case:
    #     tests += "\n"+test
    tests = "\n".join(test_case)
    text = f"""
{construct_few_shot_prompt}

**Task**:
```{language}
{prompt}
```
Your code should pass these tests:
```{language}
{tests}
```
"""
    try:
        completions = openai.ChatCompletion.create(
            model=model,
            stream=False,
            messages=[
                {"role": "system", "content": "You are a code developer."},
                {"role": "user", "content": text},
            ],
            request_timeout=100,
        )
        data_entry["completion"] = completions.choices[0]["message"]["content"]
        data_entry = preprocess_data(data_entry, language)
        return data_entry
    except Exception as e:
        print(repr(e))
        data_entry["completion"] = ""
        return data_entry


def fix_bug(
    data_entry: dict,
    model: str,
    language: str = "python",
    preprocess_data: Callable = preprocess_data,
):
    if "passed" in data_entry.keys() and data_entry["passed"] == True:
        return data_entry
    else:
        gpt_prompt = (
            "Please re-completion the code to fix the error message. "
            + f"\nHere is the previous version:\n```{language}\n"
            + data_entry["completion"]
            + f"\n```\nWhen we use this test cases: ```{language}\n"
            + data_entry["test_case"]
            + f"\n``` to evaluate the code. It raise the error:\n```{language}\n"
            + data_entry["result"]
            + f"\n```\nPlease fix the bug and return the code. The re-completion code should in triple backticks format(i.e., in ```{language} ```)."
        )
        try:
            completions = openai.ChatCompletion.create(
                model=model,
                stream=False,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a code developer assistant.",
                    },
                    {"role": "user", "content": gpt_prompt},
                ],
                request_timeout=100,
            )
            data_entry["completion"] = completions.choices[0]["message"][
                "content"
            ]
            data_entry = preprocess_data(data_entry, language)
        except Exception as e:
            print(repr(e))
    return data_entry


def call_fix_bug(dataset: List[dict], model: str, language: str = "python"):
    print("Fixing bug...")
    with ThreadPoolExecutor() as executor:
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


def call_completion(dataset: List[dict], model: str, language: str = "python"):
    print("Fixing bug...")
    with ThreadPoolExecutor() as executor:
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
    model = "deepseek-coder"
    language = "python"

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_entry = {
            executor.submit(
                fetch_completion, copy.deepcopy(entry), model, language
            ): entry
            for entry in tqdm(
                dataset, total=len(dataset), desc="Generating code"
            )
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_entry),
            total=len(dataset),
            desc="Updating code",
        ):
            entry = future_to_entry[future]
            try:
                updated_entry = future.result()
                idx = dataset.index(entry)
                dataset[idx] = updated_entry
            except Exception as e:
                print(repr(e))

    with open(f"{CWD}/data/{model}_mbpp.json", "w") as f:
        json.dump(dataset, f, indent=4)
