

import ast

# Sample machine learning code
ml_code = """
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load the ResNet50 model with pre-trained ImageNet weights
model = models.resnet50(pretrained=True)
model.eval() # Set the model to evaluation mode

# Define image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess an image
img_path = 'path/to/your/image.jpg'
img = Image.open(img_path).convert('RGB')
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model

# Make predictions
with torch.no_grad():
    output = model(input_batch)

# Get the predicted class
_, predicted_idx = torch.max(output, 1)
# You would typically have a list of ImageNet class labels to map predicted_idx to a human-readable label
print(f"Predicted class index: {predicted_idx.item()}")

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
