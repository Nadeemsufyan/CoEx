

import ast

# Sample machine learning code
ml_code = """
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1. Load the pre-trained tokenizer and model
# For a classification task, you might use BertForSequenceClassification
# For other tasks, you'd load the appropriate model (e.g., BertForQuestionAnswering)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# 2. Prepare your input text
text = "This is an example sentence for BERT inference."

# 3. Tokenize and encode the input
# `encode_plus` handles tokenization, adding special tokens ([CLS], [SEP]),
# and creating attention masks and token type IDs.
encoded_input = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=128,  # Or a suitable maximum sequence length
    padding='max_length',
    truncation=True,
    return_tensors='pt'  # Return PyTorch tensors
)

input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']
token_type_ids = encoded_input['token_type_ids'] # For tasks involving two sentences

# 4. Perform inference
with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

# 5. Process the output
# For sequence classification, outputs typically include logits
logits = outputs.logits
predictions = torch.argmax(logits, dim=1) # Get the predicted class index

print(f"Input text: {text}")
print(f"Predicted class index: {predictions.item()}")
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
