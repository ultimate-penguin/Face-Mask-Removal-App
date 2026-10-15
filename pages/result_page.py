
import streamlit as st
from PIL import Image, ImageOps
import io
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
st.title("Result")

chosen_model = st.session_state.get("model_data", "Default")#
PROJECT_ROOT = Path(__file__).parent.parent

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(4,64,4,2,1),
            nn.ReLU(True),

            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64,3,4,2,1),

            nn.Tanh()
        )

    def forward(self,x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

class Generator_u(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.ReLU(True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )



        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),   # 128+128
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),    # 64+64
            nn.Tanh()
        )

    def forward(self, x):

        e1 = self.enc1(x)   # 64
        e2 = self.enc2(e1)  # 128
        e3 = self.enc3(e2)  # 256

        d3 = self.dec3(e3)               # 128
        d3 = torch.cat([d3, e2], dim=1)  # 256

        d2 = self.dec2(d3)               # 64
        d2 = torch.cat([d2, e1], dim=1)  # 128

        out = self.dec1(d2)

        return out
st.write(chosen_model)

MODELS = {
    "Encoder–Decoder": (Generator, "generator.pth"),
    "U-NET": (Generator_u, "generator_unet.pth"),
}

if chosen_model not in MODELS:
    st.error(f"Unknown model: {chosen_model!r}")
    st.stop()

model_cls, weight_file = MODELS[chosen_model]

model = model_cls()
PATH = PROJECT_ROOT / weight_file
model.load_state_dict(torch.load(PATH, map_location="cpu"))


image_size=256
threshold=25
def denormalize(img):
    return (img + 1) / 2

transform = transforms.Compose([
                                  transforms.Resize((image_size, image_size)),
                                  transforms.ToTensor() ])


# Verify both images exist in session_state
has_img1 = "img1_data" in st.session_state and st.session_state["img1_data"] is not None
has_img2 = "img2_data" in st.session_state and st.session_state["img2_data"] is not None




if has_img1 and has_img2:
    # Load raw PIL images from bytes
    raw1 = Image.open(io.BytesIO(st.session_state["img1_data"])).convert("RGB")
    raw2 = Image.open(io.BytesIO(st.session_state["img2_data"])).convert('1')

    
    # Grayscale on Image 1, Inversion on Image 2
    masked_tensor = transform(raw1)
    mask_tensor = transform(raw2).float()
    input_tensor = torch.cat([masked_tensor, mask_tensor], dim=0)
    # ----------------------------------
    # Predict the missing region
    pred_region = model(input_tensor.unsqueeze(0))

    # Composite the final image
    output = masked_tensor * (1 - mask_tensor) + pred_region * mask_tensor

    generated = output[0]
    generated = denormalize(generated)

    

    # Display Results
    st.subheader("Image without facemask")
    # 1. Permute to (H, W, C) and convert to NumPy array
    np_image = generated.permute(1, 2, 0).detach().cpu().numpy()

    # Note: If pixel values are floats in range [0, 1], st.image renders them directly.
    # If values are normalized or standard 0-255 uint8, adjust accordingly.

    # 2. Display directly in Streamlit
    
    st.image(np_image, caption="removed facemask", use_container_width=True)

    

    
else:
    st.warning("Missing images! Please upload both images from the home page first.")

st.divider()

# Back button with session state reset
if st.button("← Upload New Images"):
    st.session_state["img1_data"] = None
    st.session_state["img2_data"] = None
    st.switch_page("app.py")
