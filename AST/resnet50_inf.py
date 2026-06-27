import ast

# Sample machine learning code
ml_code = """
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 1. Load a pre-trained ResNet model
# You can choose different versions like resnet18, resnet34, resnet50, etc.
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# 2. Define image transformations
# These transformations should match what the model was trained on (e.g., ImageNet)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Load and preprocess an image
image_path = "path/to/your/image.jpg"  # Replace with your image path
img = Image.open(image_path).convert('RGB')
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# 4. Perform inference
with torch.no_grad():  # Disable gradient calculation for inference
    output = model(input_batch)

# 5. Process the output (e.g., get the predicted class)
# The output will be a tensor of logits; apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Load ImageNet class labels (if needed for interpretation)
# You would typically have a file mapping indices to class names
# For example, a list of 1000 class names
# with open("imagenet_classes.txt") as f:
#     classes = [line.strip() for line in f.readlines()]

# Get the top predicted class
top_probability, top_category_id = torch.max(probabilities, 0)
print(f"Predicted class ID: {top_category_id.item()}")
print(f"Probability: {top_probability.item():.4f}")
# If you have class names, you can print:
# print(f"Predicted class: {classes[top_category_id.item()]}")

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
