import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

def caption_image(input_image:np.ndarray):

    # Load the pretrained processor and model
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    # Load your image, DON'T FORGET TO WRITE YOUR IMAGE NAME
    img_path = "photo.png"
    # convert it into an RGB format 
    image = Image.open(img_path).convert('RGB')
    # You do not need a question for image captioning
    text = "the image of"
    inputs = processor(images=image, text=text, return_tensors="pt")
    # Generate a caption for the image
    outputs = model.generate(**inputs, max_length=50)
    # Decode the generated tokens to text
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    # Print the caption
    return caption

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
)

iface.launch('share=True')