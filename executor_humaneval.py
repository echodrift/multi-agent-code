import contextlib
import io
import json
import os
import sys

CWD = os.path.abspath(os.path.dirname(__file__))
sys.path.append(f"{CWD}/CodeGeeX/")
import concurrent.futures
import signal

from codegeex.benchmark.execution import check_correctness
from codegeex.benchmark.utils import IMPORT_HELPER
from tqdm import tqdm

from coder_humaneval import call_fetch_completion_helper
from tester_humaneval import call_fetch_test_completion_helper


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def process_humaneval_test(
    sample, problems, example_test=False, language="python", test_case=True
):
    task_id = sample["task_id"]
    task_id = problems.index(sample)
    prompt = sample["prompt"]
    if (
        example_test
        and "example_test" in problems[task_id]
        and problems[task_id]["example_test"] != ""
    ):
        test = problems[task_id]["example_test"]
    else:
        test = problems[task_id]["test"]
    if test_case:
        test = problems[task_id]["test_case"]
    code = sample["completion"]
    # Pre-process for different languages
    if language == "python":
        test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
        if f"class {sample['entry_point']}" in code:
            test_string = (
                test_setup
                + code
                + "\n"
                + test
                + "\n"
                + f"check({sample['entry_point']})"
            )
        else:
            test_string = (
                test_setup
                + prompt
                + code
                + "\n"
                + test
                + "\n"
                + f"check({sample['entry_point']})"
            )
    return test_string


def preprocess_data(task, language: str = "python"):
    if f"```{language}" in task["completion"]:
        task["completion"] = task["completion"][
            task["completion"].find(f"```{language}") + len(f"```{language}") :
        ]
        task["completion"] = task["completion"][
            : task["completion"].find("```")
        ]
    elif "```" in task["completion"]:
        task["completion"] = task["completion"][
            task["completion"].find("```") + 3 :
        ]
        task["completion"] = task["completion"][
            : task["completion"].find("```")
        ]

    if f"```{language}" in task["prompt"]:
        task["prompt"] = task["prompt"][
            task["prompt"].find(f"```{language}") + len(f"```{language}") :
        ]
        task["prompt"] = task["prompt"][: task["prompt"].find("```")]
    elif "```" in task["prompt"]:
        task["prompt"] = task["prompt"][task["prompt"].find("```") + 3 :]
        task["prompt"] = task["prompt"][: task["prompt"].find("```")]

    if "assert" in task["prompt"]:
        task["prompt"] = task["prompt"][: task["prompt"].find("assert")]
    return task


def test_report(dataset, language="python"):
    correct = 0
    test_setup = "\n".join(IMPORT_HELPER[language]) + "\n"
    for i in tqdm(range(len(dataset)), desc="Executing code"):
        try:
            with swallow_io():
                with time_limit(2.0):
                    exec(
                        test_setup
                        + "\n"
                        + dataset[i]["completion"]
                        + "\n"
                        + dataset[i]["test"]
                        + "\n"
                        + f"check({dataset[i]['entry_point']})"
                    )
                correct += 1
        except Exception as exc:
            pass
    print("==============Start Report Testing==============")
    print(f"test_report: {(correct/len(dataset)*100):.1f}")


def test_agent_concurrency(dataset, language):
    test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
    _for_completion = 0

    def process_item(i):
        if (
            "need_reproduce" in dataset[i].keys()
            and dataset[i]["need_reproduce"] == False
        ):
            # dataset[i]["need_reproduce"] = True
            return dataset[i]["max_correct"], dataset[i]["idx"]
        completion_list = dataset[i]["completion_list"]
        test_case_list = dataset[i]["test_case_list"]
        correct_list = []

        for j in range(len(completion_list)):
            correct = 0
            if f"def {dataset[i]['entry_point']}" not in completion_list[j]:
                correct_list.append(correct)
                continue
            for k in range(len(test_case_list)):
                if (
                    f"assert {dataset[i]['entry_point']}("
                    not in test_case_list[k]
                ):
                    continue
                # dataset[i]["full_code"] = test_setup + "\n" + completion_list[j] + "\n" + test_case_list[k]
                dataset[i]["test_code"] = (
                    test_setup
                    + "\n"
                    + completion_list[j]
                    + "\n"
                    + test_case_list[k]
                )
                # print(dataset[i]["task_id"], dataset[i], language, 3, "../tmp")
                result = check_correctness(
                    dataset[i]["task_id"], dataset[i], language, 3, f"{CWD}/tmp"
                )
                if result["passed"]:
                    correct += 1
            correct_list.append(correct)

        max_correct = max(correct_list)
        idx = correct_list.index(max_correct)

        return max_correct, idx

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_item, i) for i in range(len(dataset))
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(dataset)
        ):
            max_correct, idx = future.result()
            if (
                max_correct >= 3
            ):  # GPT-3.5-turbo-1106's test case accuracy is about 67%. So we choice 60% as the bar.
                i = futures.index(future)
                dataset[i]["completion"] = dataset[i]["completion_list"][idx]
                dataset[i]["need_reproduce"] = False
                dataset[i]["idx"] = idx
                dataset[i]["max_correct"] = max_correct
                _for_completion += 1
            else:
                i = futures.index(future)
                dataset[i]["completion"] = dataset[i]["completion_list"][idx]

    print("==============Start Agent Testing==============")
    print(f"test_for_completion: {(_for_completion/len(dataset)*100):.1f}")
    return dataset


if __name__ == "__main__":
    model = "deepseek-coder"
    language = "python"

    path = f"{CWD}/data/{model}_{language}.json"

    with open(path, "r") as f:
        dataset = json.load(f)
    epoch = 5
    for current_epoch in tqdm(range(3, epoch), desc="Epoch"):
        dataset = test_agent_concurrency(dataset, language)
        test_report(dataset, language)
        dataset = call_fetch_completion_helper(dataset, model, language)
        dataset = call_fetch_test_completion_helper(dataset, model, language)
        with open(f"{CWD}/data/{model}_{current_epoch}.json", "w") as f:
            json.dump(dataset, f, indent=4)

    dataset = test_agent_concurrency(dataset, language)
    test_report(dataset, language)
    with open(f"{CWD}/data/deepseek-coder_python.json", "r") as f:
        dataset = json.load(f)
    for i in range(len(dataset)):
        dataset[i]["completion"] = dataset[i]["completion_list"][0]
    language = "python"
    test_report(dataset, language)
