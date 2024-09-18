import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from fpdf import FPDF
from PIL import Image
import os

# Load the Stable Diffusion pipeline (ensure that you have GPU if possible)
@st.cache_resource
def load_pipeline():
    return StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda" if torch.cuda.is_available() else "cpu")

pipe = load_pipeline()

# Streamlit App Layout
st.title("Story to Image Generator")
st.write("Enter your story (up to 6 paragraphs) and generate images for each paragraph to create an illustrated storybook.")

# Text Input for the story
story = st.text_area("Enter your story (each paragraph should be separated by a double newline):", height=300)
paragraphs = story.split("\n\n")

if st.button("Generate Images"):

    if len(paragraphs) > 6:
        st.error("Please limit the story to 6 paragraphs.")
    else:
        # Generate images for each paragraph
        images = []
        for idx, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                st.write(f"Generating image for paragraph {idx + 1}...")
                image = pipe(paragraph).images[0]
                images.append(image)
            else:
                st.error(f"Paragraph {idx + 1} is empty.")

        if images:
            st.success("Images generated successfully!")

            # Display images alongside their respective paragraphs
            st.write("### Story with Images")
            for paragraph, image in zip(paragraphs, images):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, use_column_width=True)
                with col2:
                    st.write(paragraph)

            # Function to save images and text to a PDF
            def save_storybook(paragraphs, images, output_path="storybook.pdf"):
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                
                for paragraph, image in zip(paragraphs, images):
                    pdf.add_page()
                    # Add text to the PDF
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, paragraph)

                    # Save image temporarily
                    image_path = f"image_{hash(paragraph)}.png"
                    image.save(image_path)

                    # Add image to PDF
                    pdf.image(image_path, x=10, y=60, w=100)
                    
                    # Cleanup temp image
                    os.remove(image_path)

                # Save the PDF
                pdf.output(output_path)
                return output_path

            # Save the storybook and create a download button
            storybook_path = save_storybook(paragraphs, images)
            with open(storybook_path, "rb") as file:
                st.download_button("Download Storybook (PDF)", file, file_name="storybook.pdf", mime="application/pdf")
