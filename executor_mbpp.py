import contextlib
import io
import json
import os
import signal
import sys

CWD = os.path.abspath(os.path.dirname(__file__))
sys.path.append(f"{CWD}/CodeGeeX/")
from typing import List

from codegeex.benchmark.execution import check_correctness
from codegeex.benchmark.utils import IMPORT_HELPER
from tqdm import tqdm

from coder_mbpp import call_fetch_completion
from tester_mbpp import call_fetch_test_completion


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


def test_report(dataset: dict, language: str = "python"):
    test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
    correct = 0
    for i in tqdm(range(len(dataset)), desc="Test report"):
        dataset[i]["test_code"] = (
            test_setup
            + "\n"
            + dataset[i]["completion"].strip()
            + "\n"
            + '\n'.join(dataset[i]["test_list"]).strip()
        )
        result = check_correctness(
            dataset[i]["task_id"], dataset[i], language, 5, f"{CWD}/tmp"
        )
        if result["passed"] == True:
            correct += 1
        dataset[i]["report_passed"] = result["passed"]
    print("==============Start Report Testing==============")
    correct_percent = correct / len(dataset) * 100
    print(f"test_report, {correct_percent:0.2f}")
    return dataset


def test_agent(dataset: List[dict], language: str = "python"):
    test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
    cnt_correct = 0
    for i in tqdm(range(len(dataset)), desc="Test agent"):
        completion_list = dataset[i]["completion_list"]
        test_case_list = dataset[i]["test_case_list"]
        correct_list = []
        for completion in completion_list:
            correct = 0
            for tests in test_case_list:
                dataset[i]["test_code"] = (
                    test_setup
                    + "\n"
                    + completion.strip()
                    + "\n"
                    + tests.strip()
                )
                print(dataset[i]["test_code"])
                result = check_correctness(
                    dataset[i]["task_id"], dataset[i], language, 5, f"{CWD}/tmp"
                )
                if result["passed"] == True:
                    correct += 1
            
            correct_list.append(correct)
        best_completion = completion_list[correct_list.index(max(correct_list))]
        if max(correct_list) >= 1:
            cnt_correct += 1
        dataset[i]["completion"] = best_completion 
        dataset[i]["passed"] = result["passed"]
    print("============Start Agent Testing=================")
    print("test_agent:", "{:.2f}".format(cnt_correct / len(dataset) * 100))
    return dataset


if __name__ == "__main__":
    model = "deepseek-coder"
    language = "python"

    path = f"{CWD}/data/{model}_mbpp.json"
    with open(path, "r") as f:
        dataset = json.load(f)
    epoch = 3
    for current_epoch in tqdm(range(epoch), desc="Epoch"):
        dataset = test_agent(dataset, language)
        test_report(dataset, language)
        dataset = call_fetch_completion(dataset, model, language)
        dataset = call_fetch_test_completion(dataset, model, language)
        with open(
            f"{CWD}/data/{model}_{current_epoch}_mbpp.json",
            "w",
        ) as f:
            json.dump(dataset, f, indent=4)
    with open(
        f"{CWD}/data/{model}_{current_epoch}_mbpp_total.json",
        "w",
    ) as f:
        json.dump(dataset, f, indent=4)
