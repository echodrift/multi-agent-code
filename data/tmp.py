import json

with open("deepseek-coder_2_humaneval.json", "r") as f:
    humaneval2 = json.load(f)

with open("deepseek-coder_3_humaneval.json", "r") as f:
    humaneval3 = json.load(f)

for i in range(164):
    if ("need_reproduce" not in humaneval2[i].keys() 
        and "need_reproduce" in humaneval3[i].keys()):
        print(i)
