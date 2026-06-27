import ast

# Sample machine learning code
ml_code = """


from datasets import load_dataset
ds = load_dataset("wmt16", "ro-en")

print(ds)

print(ds["train"][0])

from transformers import AutoTokenizer

model_checkpoint = "Helsinki-NLP/opus-mt-en-ro"
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if "mbart" in model_checkpoint:
    tokenizer.src_lang = "en-XX"
    tokenizer.tgt_lang = "ro-RO"
    

if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "translate English to Romanian: "
else:
    prefix = ""
    
tokenizer("Hello, this one sentence!")

"""
{'input_ids': [125, 778, 3, 63, 141, 9191, 23, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
"""

tokenizer(["Hello, this one sentence!", "This is another sentence."])

"""
{'input_ids': [[125, 778, 3, 63, 141, 9191, 23, 0], [187, 32, 716, 9191, 2, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]}
"""

max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "ro"


def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

preprocess_function(ds['train'][:2])

from datasets import DatasetDict

small_train = ds["train"].shuffle(seed=42).select(range(2000))

small_ds = DatasetDict({
    "train": small_train,
    "validation": ds["validation"],
    "test": ds["test"],
})

tokenized_datasets = small_ds.map(preprocess_function, batched=True)


import evaluate

metric = evaluate.load("sacrebleu")

fake_preds = ["hello there", "general kenobi"]
fake_labels = [["hello there"], ["general kenobi"]]

metric.compute(predictions=fake_preds, references=fake_labels)

"""
{'score': 0.0,
 'counts': [4, 2, 0, 0],
 'totals': [4, 2, 0, 0],
 'precisions': [100.0, 100.0, 0.0, 0.0],
 'bp': 1.0,
 'sys_len': 4,
 'ref_len': 4}
"""


import numpy as np

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode token IDs to text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 (used to mask loss calculation) with padding token ID for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean up spacing and format references properly
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Compute BLEU score
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    # Track average length of generated sequences
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    # Round values for readability
    result = {k: round(v, 4) for k, v in result.items()}
    return result


import torch

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


batch_size = 16
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
    eval_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

import time    
import cProfile
import pstats
profiler = cProfile.Profile()
    
start_time = time.time() 
profiler.enable() # Start profiling
trainer.train()
profiler.disable() # Stop profiling
end_time = time.time()
training_duration = end_time - start_time

stats = pstats.Stats(profiler)
stats.strip_dirs() # Remove directory paths for cleaner output
stats.sort_stats('cumulative') # Sort by cumulative time spent in function
stats.print_stats() # Print the profiling results

print(f"Total training time: {training_duration:.2f} seconds")


output_dir = f"{model_name}-finetuned-{source_lang}-to-{target_lang}"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_inf = AutoModelForSeq2SeqLM.from_pretrained(output_dir).to(device)
tokenizer_inf = AutoTokenizer.from_pretrained(output_dir)

sentence = "The quick brown fox jumps over the lazy dog."

inputs = tokenizer_inf(sentence, return_tensors="pt", truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}

generated_ids = model_inf.generate(**inputs, max_length=128, num_beams=4)
print(tokenizer_inf.decode(generated_ids[0], skip_special_tokens=True))


"""


class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.function_calls = 0
        self.load_statements = 0
        self.store_statements = 0
        self.loops = 0
        self.control=0
        

    def visit_Call(self, node):
        self.function_calls += 1
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.load_statements += 1
        elif isinstance(node.ctx, ast.Store):
            self.store_statements += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.loops += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.loops += 1
        self.generic_visit(node)
   
    def visit_If(self, node):
        self.control += 1
        self.generic_visit(node)  
        
    def visit_Function(self, node):
        self.control += 1
        self.generic_visit(node) 
   

# Parse the code into an AST
tree = ast.parse(ml_code)

# Analyze the AST
analyzer = CodeAnalyzer()
analyzer.visit(tree)

function_count = 0
for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_count += 1

print(f"Number of control calls: {analyzer.control}")
print(f"Number of function calls: {analyzer.function_calls}")
print(f"Number of load statements: {analyzer.load_statements}")
print(f"Number of store statements: {analyzer.store_statements}")
print(f"Number of loops: {analyzer.loops}")
print(f"Number of function: {function_count}")
