

import ast

# Sample machine learning code
ml_code = """
import torch

def translate_sentence(model, sentence, source_vocab, target_vocab, device, max_len=50):
    model.eval() # Set model to evaluation mode

    # Preprocess the input sentence
    # (Assuming sentence is a list of tokens, e.g., ['hello', 'world'])
    indexed = [source_vocab.stoi[token] for token in sentence]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1) # Add batch dimension

    # Encode the input
    with torch.no_grad():
        hidden, cell = model.encoder(tensor)

    # Decode to generate the output
    trg_indexes = [target_vocab.stoi['<sos>']] # Start with <sos> token
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == target_vocab.stoi['<eos>']:
            break

    # Postprocess the output
    trg_tokens = [target_vocab.itos[idx] for idx in trg_indexes]
    return trg_tokens[1:] # Exclude <sos> token
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
