import torch
from transformers import BertForSequenceClassification, BertTokenizer
import os
import numpy as np

model_path = "/kaggle/working/final_model"

onnx_output_path = "/kaggle/working/results/model.onnx"

print("Loading model and tokenizer...")
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Set the model to evaluation mode
model.eval()

# Create dummy inputs for ONNX export
dummy_input_ids = torch.ones(1, 128, dtype=torch.long)
dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)
dummy_token_type_ids = torch.zeros(1, 128, dtype=torch.long)

# For older transformers versions
if hasattr(model, "config") and hasattr(model.config, "output_attentions") and model.config.output_attentions:
    model.config.output_attentions = False
if hasattr(model, "config") and hasattr(model.config, "output_hidden_states") and model.config.output_hidden_states:
    model.config.output_hidden_states = False

dynamic_axes = {
    'input_ids': {0: 'batch_size', 1: 'sequence'},
    'attention_mask': {0: 'batch_size', 1: 'sequence'},
    'token_type_ids': {0: 'batch_size', 1: 'sequence'},
    'output': {0: 'batch_size'}
}

print("Converting model to ONNX format...")
try:
    # Export the model to ONNX format
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
        onnx_output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

    if os.path.exists(onnx_output_path):
        print(f"Model successfully exported to {onnx_output_path}")
        print(f"File size: {os.path.getsize(onnx_output_path) / (1024 * 1024):.2f} MB")
    else:
        print(f"ERROR: Export appeared to succeed but file {onnx_output_path} was not created!")
except Exception as e:
    print(f"Error during ONNX export: {e}")

# Verify the exported model if it exists
if os.path.exists(onnx_output_path):
    try:
        import onnx

        onnx_model = onnx.load(onnx_output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
    except Exception as e:
        print(f"Error verifying ONNX model: {e}")
else:
    print("Skipping verification as ONNX file doesn't exist.")


# Save the tokenizer vocabulary and configuration
tokenizer_output_path = "/kaggle/working/results/tokenizer"
os.makedirs(tokenizer_output_path, exist_ok=True)
tokenizer.save_pretrained(tokenizer_output_path)
print(f"Tokenizer saved to {tokenizer_output_path}")


# Test the ONNX model if it exists
def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


model_to_test = None

if os.path.exists(onnx_output_path):
    model_to_test = onnx_output_path
    print("\nTesting the ONNX model...")
else:
    print("\nNo ONNX model available for testing.")

if model_to_test:
    try:
        import onnxruntime as ort

        print(f"Loading ONNX model for testing from: {model_to_test}")
        session = ort.InferenceSession(model_to_test, providers=['CPUExecutionProvider'])

        # Test text
        test_text = "This is a sample text to test the ONNX model."

        # Tokenize the text
        tokens = tokenizer(
            test_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='np'
        )

        ort_inputs = {
            'input_ids': tokens['input_ids'].astype(np.int64),
            'attention_mask': tokens['attention_mask'].astype(np.int64),
        }

        if 'token_type_ids' in tokens:
            ort_inputs['token_type_ids'] = tokens['token_type_ids'].astype(np.int64)

        # Get the actual output node name
        output_node_name = session.get_outputs()[0].name
        print(f"Model output node name: {output_node_name}")

        ort_outputs = session.run([output_node_name], ort_inputs)

        logits = ort_outputs[0]
        probabilities = softmax(logits)
        predicted_class = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0][predicted_class] * 100

        print(f"Test prediction: Class {predicted_class} with {confidence:.2f}% confidence")
        print("ONNX model test successful!")

    except ImportError:
        print("onnxruntime not found. Skipping testing step.")
        print("To test the model, install onnxruntime: pip install onnxruntime")
    except Exception as e:
        print(f"An error occurred during ONNX model testing: {e}")
        print("Skipping testing step.")

print("\nConversion process completed!")
print("Files for your Chrome extension (if conversion was successful):")
if os.path.exists(optimized_output_path):
    print(f"- Optimized ONNX Model: {optimized_output_path}")
elif os.path.exists(onnx_output_path):
    print(f"- ONNX Model: {onnx_output_path}")
else:
    print("- No ONNX model was successfully created")

if os.path.exists(tokenizer_output_path):
    print(f"- Tokenizer files: {tokenizer_output_path}")
else:
    print("- Tokenizer files were not successfully saved")

# List all files in the results directory to confirm what was created
results_dir = "/kaggle/working/results"
if os.path.exists(results_dir):
    print("\nFiles created in results directory:")
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            filepath = os.path.join(root, file)
            print(f"- {filepath} ({os.path.getsize(filepath) / 1024:.2f} KB)")
else:
    print("\nResults directory does not exist.")