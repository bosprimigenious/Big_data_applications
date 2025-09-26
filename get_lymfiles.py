import kagglehub

# Download latest version
path = kagglehub.dataset_download("navjotkaushal/human-vs-ai-generated-essays")

print("Path to dataset files:", path)