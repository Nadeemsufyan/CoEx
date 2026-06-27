

import ast

# Sample machine learning code
ml_code = """
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 1. Load the pre-trained VGG16 model
# Use VGG16_Weights.IMAGENET1K_V1 for pretrained weights
vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
vgg16.eval() # Set the model to evaluation mode

# 2. Define the image transformation
# The required transforms are available with the weights object
preprocess = models.VGG16_Weights.IMAGENET1K_V1.transforms()

# 3. Load and preprocess an image
image_path = 'your_image.jpg' # Replace with your image path
input_image = Image.open(image_path).convert('RGB')
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model

# 4. Run inference
with torch.no_grad():
    output = vgg16(input_batch)

# 5. Post-process the output to get the predicted class
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Load ImageNet class names
# You would need a file (e.g., "imagenet_classes.txt") containing the 1000 class names
# Example of how to get top 5 categories
top5_prob, top5_catid = torch.topk(probabilities, 5)
print("Top 5 predictions:")
for i in range(top5_prob.size(0)):
    print(f"Class ID: {top5_catid[i].item()}, Probability: {top5_prob[i].item():.4f}")

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
