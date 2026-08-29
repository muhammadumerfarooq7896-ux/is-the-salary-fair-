import modal
from modal import Image



app = modal.App("salary-service")
image = Image.debian_slim().pip_install(
    "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)


secrets = [modal.Secret.from_name("huggingface-secret")]

# Constants

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "toughfigure"  
RUN_NAME = "2025-11-28_18.47.07"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd"
FINETUNED_MODEL = "toughfigure/llama-3.2-qlora"

@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def price(description: str) -> float:
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel

    PREFIX = "salary is $"
    QUESTION = "What is the estimated salary in us dollars for this role"

    prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

    # Quant Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    # Load model and tokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_config, device_map="auto"
    )

    fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL, revision=REVISION)

    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = fine_tuned_model.generate(inputs, max_new_tokens=5)
    result = tokenizer.decode(outputs[0])
    contents = result.split("Price is $")[1]
    contents = contents.replace(",", "")
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
    return float(match.group()) if match else 0
