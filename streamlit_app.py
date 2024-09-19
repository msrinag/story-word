
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

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

            
