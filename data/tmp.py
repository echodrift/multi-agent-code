import json

with open("deepseek-coder_mbpp.json", "r") as f:
    test = json.load(f)
print(test[0].keys())
# with open("deepseek-coder_mbpp_code.json", "r") as f:
#     code = json.load(f)

# for i in range(257):
#     code[i]["test_case_list"] = test[i]["test_case_list"]

# with open("deepseek-coder_mbpp.json", 'w') as f:
#     json.dump(code, f)
