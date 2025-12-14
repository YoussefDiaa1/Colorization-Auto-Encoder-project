import streamlit as st
import torch
from models import VAE, Colorizer
import torchvision.utils as vutils
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Face App", page_icon="🎨", layout="wide")


@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(256)
    colorizer = Colorizer()

    vae.load_state_dict(torch.load('vae_FINAL_50epochs.pth',
                        map_location=device)['model_state_dict'])
    colorizer.load_state_dict(torch.load(
        'colorizer_FINAL_30epochs.pth', map_location=device)['model_state_dict'])

    vae.eval()
    colorizer.eval()
    return vae, colorizer, device


st.title("🎨 AI Face Generation & Colorization")
vae, colorizer, device = load_models()
st.success(f"✅ Models loaded! Device: {device}")

tab1, tab2, tab3 = st.tabs(["🎭 Generator", "🌈 Colorizer", "🔗 Pipeline"])

with tab1:
    st.header("Generate Faces")
    num = st.slider("Number:", 1, 64, 16)
    if st.button("Generate"):
        with torch.no_grad():
            z = torch.randn(num, 256).to(device)
            imgs = vae.decoder(z).cpu()
        grid = vutils.make_grid(
            imgs, nrow=8, normalize=True).permute(1, 2, 0).numpy()
        st.image(grid, use_container_width=True)

with tab2:
    st.header("Colorize Image")
    file = st.file_uploader("Upload", type=['png', 'jpg', 'jpeg'])
    if file:
        img = Image.open(file).convert('RGB').resize((64, 64))
        col1, col2 = st.columns(2)
        col1.image(img, caption="Input", use_container_width=True)
        if col2.button("Colorize"):
            arr = np.array(img)/255.0
            tensor = torch.from_numpy(arr).permute(
                2, 0, 1).unsqueeze(0).float().to(device)
            gray = tensor.mean(1, keepdim=True)
            with torch.no_grad():
                result = colorizer(gray)
            out = result[0].cpu().permute(1, 2, 0).numpy()
            out = ((out-out.min())/(out.max()-out.min())*255).astype(np.uint8)
            col2.image(out, caption="Colorized", use_container_width=True)

with tab3:
    st.header("Complete Pipeline")
    if st.button("Run"):
        with torch.no_grad():
            z = torch.randn(8, 256).to(device)
            gen = vae.decoder(z)
            gray = gen.mean(1, keepdim=True)
            color = colorizer(gray)

        col1, col2, col3 = st.columns(3)

        grid1 = vutils.make_grid(
            gen.cpu(), 4, normalize=True).permute(1, 2, 0).numpy()
        grid2 = vutils.make_grid(gray.repeat(1, 3, 1, 1).cpu(
        ), 4, normalize=True).permute(1, 2, 0).numpy()
        grid3 = vutils.make_grid(
            color.cpu(), 4, normalize=True).permute(1, 2, 0).numpy()

        col1.image(grid1, caption="Generated", use_container_width=True)
        col2.image(grid2, caption="Grayscale", use_container_width=True)
        col3.image(grid3, caption="Colorized", use_container_width=True)
