from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "deepseek-ai/deepseek-coder-6.7b-base"
quant_path = "lvdthieu/deepseek-coder-6.7b-base-quantized"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
