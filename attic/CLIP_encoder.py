#!/usr/bin/env python3
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Example image and text inputs
image = Image.open("./attic/cat.png")  # Provide an image file path
text = ["a photo of a cat", "a photo of a dog"]

# Preprocess inputs
inputs = processor(
    text=text,
    images=image,
    return_tensors="pt",
    padding=True
)

# Encode text and image into latent space
outputs = model(**inputs)

# Get image and text embeddings
image_embeddings = outputs.image_embeds  # Shape: [batch_size, embedding_dim]
text_embeddings = outputs.text_embeds    # Shape: [batch_size, embedding_dim]

# Normalize embeddings for cosine similarity comparison
image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

# Compute similarity between image and text
similarity = (image_embeddings @ text_embeddings.T)  # Cosine similarity matrix
print("Similarity scores:", similarity)
