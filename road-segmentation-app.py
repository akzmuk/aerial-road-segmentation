import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch.nn as nn

# Model Definition (same as in the original script)
class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()

        self.backbone = smp.Unet(
            encoder_name = "timm-efficientnet-b0",
            encoder_weights = "imagenet", 
            in_channels = 3,
            classes = 1,
            activation = None
        )

    def forward(self, images, masks = None):
        logits = self.backbone(images)
        return logits

# Image Preprocessing Functions
def preprocess_image(image):
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 512x512
    image = cv2.resize(image, (512, 512))
    
    # Normalize and convert to tensor
    image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
    image = torch.Tensor(image).unsqueeze(0)
    
    return image

def postprocess_mask(mask):
    # Convert mask to numpy and squeeze dimensions
    mask = mask.detach().cpu().numpy().squeeze()
    
    # Threshold and convert to uint8
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    return mask

# Streamlit App
def main():
    st.title('Road Segmentation Model')
    st.write('Upload an image to perform road segmentation')

    # Model loading
    @st.cache_resource
    def load_model():
        model = SegmentationModel()
        model.load_state_dict(torch.load("/content/best-model.pth", map_location=torch.device('cpu')))
        model.eval()
        return model

    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image
        st.subheader('Original Image')
        st.image(image, channels='BGR')

        # Preprocess image
        input_tensor = preprocess_image(image)

        # Perform inference
        with torch.no_grad():
            logits = model(input_tensor)
            pred_mask = torch.sigmoid(logits)

        # Postprocess mask
        mask = postprocess_mask(pred_mask)

        # Resize mask to original image size
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Display segmentation mask
        st.subheader('Road Segmentation Mask')
        st.image(mask, channels='GRAY')

        # Optional: Overlay mask on original image
        st.subheader('Segmentation Overlay')
        overlay = image.copy()
        overlay[mask == 255] = [0, 255, 0]  # Green color for road
        st.image(overlay, channels='BGR', caption='Green indicates road')

if __name__ == "__main__":
    main()
