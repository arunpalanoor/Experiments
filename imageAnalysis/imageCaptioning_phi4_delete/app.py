import gradio as gr
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForMultimodal, AutoTokenizer

# Load Phi-4 Multimodal
model_id = "microsoft/Phi-4-multimodal-instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodal.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Captioning function
def caption_image(image: Image.Image):
    inputs = processor(text="Describe this image in detail.", images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    caption = tokenizer.decode(output[0], skip_special_tokens=True)
    return caption

# Gradio UI
demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="🖼️ Phi-4 Multimodal Image Captioning",
    description="Upload an image and let Phi-4 Multimodal generate a vivid caption."
)

if __name__ == "__main__":
    demo.launch()