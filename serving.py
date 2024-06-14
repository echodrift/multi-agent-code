from vllm import LLM, SamplingParams

llm = LLM(model="deepseek-ai/deepseek-coder-6.7b-base")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
