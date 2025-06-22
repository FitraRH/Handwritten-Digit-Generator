import streamlit as st
import torch 
import torch.nn as nn
import numpy as np
from PIL import Image
import io


class Generator(nn.Module):
    def __init__(self):
        super (Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.ReLU(),
        )

    def forward(self, noise, labesl):
        x = torch.cat((noise, labesl), dim=1)
        return x.view(-1, 1, 28, 28)
    
    @st.cache_resource
    def load_generator():
        model = Generator()
        for param in model.parameters():
            nn.init.normal_(param, 0, 0.02)
        return model
    
    def generate_image(model, digit):
        with torch.no_grad():
            noise = torch.randn(5, 100)
            labels = torch.zeros(5, 10)
            labels[:, digit] = 1
            fake_imgs = model(noise, labels)
            fake_imgs = (fake_imgs + 1) / 2 * 225
            return fake_imgs.numpy().astype(np.uint8)
        
    st.title("Handwritten Digit Generator")
    
    digit = st.selectbox("Select a digit (0-9):", range (10), index=2)
    
    if st.button("Generate Image"):
        model = load_generator()
        images = generate_image(model, digit)
        
        st.subheader(f"Generated Images for Digit {digit}")
        cols = st.columns(5)
        
        for i, img in enumerate(images):
            with cols[i]:
                pil_img = Image.fromarray(img, mode='L')
                st.image(pil_img, width=100)
                
                buf = io.BytesIO()
                pil_img.save(buf, format='PNG')
                st.download_button(
                    f"Download Image {i+1}",
                    buf.getvalue(),
                    f"digit_{digit}_image_{i+1}.png",
                    "image/png"
                )
        
    