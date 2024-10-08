from llama_dummy_cache import LlamaForCausalLM_DummyKV
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
# from transformers import BitsAndBytesConfig
from datasets import load_dataset
import torch
import argparse


#### Config for KIVI model
# import os
# my_token = os.environ.get("HUGGINGFACE_API_TOKEN")
# config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf", token=my_token)
config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

config.k_bits = 2# current support 2/4 bit for KV Cache
config.v_bits = 2 # current support 2/4 bit for KV Cache
config.group_size = 64
config.residual_length = 64 # the number of recent fp16 tokens

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
parser = argparse.ArgumentParser(description="Evaluate AQuA Tasks")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
# parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b", help="Model name or path.")
parser.add_argument("--model", type=str, default="None", help="Model name or path.")
args = parser.parse_args()

max_token = 1000 ### prefill_length
max_generation_length = 1500 ### geneate 500
batch_size = args.batch_size

DEVICE = "cuda:1"

if "None" in args.model:
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16, # FP16 으로 불러오도록 추가됨.
        device_map = DEVICE)
elif "dummy" in args.model:
    model = LlamaForCausalLM_DummyKV.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        config = config,
        torch_dtype=torch.float16, # FP16 으로 불러오도록 추가됨.
        device_map = DEVICE
    )
else:
    print("args model is not supported. args.model: ", args.model) # 인자가 무조건 필요함.
    exit(1)

model = model.half() # it is model.to(torch.float16)

peak_memory = torch.cuda.max_memory_allocated(device="cuda") / (1024**2)
print(f"Peak memory usage on GPU for model loading: {peak_memory} MB")


tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-hf', 
    model_max_length=max_token,
    max_length=max_token,
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')
tokenizer.pad_token = tokenizer.eos_token
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text_combined = test["text"]

sentence_group = []
for i in range(batch_size):
    # sentence_group.append(str(text_combined[i*max_token:(i+1)*max_token]))
    sentence_group.append(str(text_combined[0:max_token]))
inputs = tokenizer(
    sentence_group,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
)
print("begin")
inputs = inputs.to(DEVICE)
print(inputs.input_ids.shape)
import time

start = time.time()
generate_results = model.generate(**inputs, max_length=max_generation_length, use_cache=True)
torch.cuda.synchronize()
end = time.time()
peak_memory = torch.cuda.max_memory_allocated(device=DEVICE) / (1024**2)  # 转换为MB单位

print(f"Peak memory usage on GPU: {peak_memory} MB")
print("time",end - start, 's')
# result = tokenizer.batch_decode(result, skip_special_tokens=True)
# print(result)
# model = model.cuda()
