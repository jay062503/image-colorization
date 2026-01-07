# ensemble for photos (aspect ratio corrected)
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from tqdm import tqdm # Make sure tqdm is imported if used elsewhere
import os

# ---------------- 1. DEFINE BASE CLASS ----------------
# (Using modern Python 3 'super()')
class BaseColor(nn.Module):
    def __init__(self):
        super().__init__() # Use modern super()
        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.

    def normalize_l(self, in_l): return (in_l - self.l_cent) / self.l_norm
    def unnormalize_l(self, in_l): return in_l * self.l_norm + self.l_cent
    def normalize_ab(self, in_ab): return in_ab / self.ab_norm
    def unnormalize_ab(self, in_ab): return in_ab * self.ab_norm

# ---------------- 2. DEFINE ECCVGenerator ----------------
class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64)
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128)
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256)
        )
        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512)
        )
        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512)
        )
        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),
            nn.ReLU(True),
            norm_layer(512)
        )
        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512)
        )
        self.model8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))
        return self.unnormalize_ab(self.upsample4(out_reg))

# ---------------- 3. DEFINE SIGGRAPHGenerator ----------------
class SIGGRAPHGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d, classes=529):
        super().__init__()

        # Re-using the clean definition from your previous script
        def C(in_c, out_c, k=3, s=1, p=1, d=1):
            return nn.Conv2d(in_c, out_c, k, s, p, dilation=d, bias=True)

        self.model1 = nn.Sequential(C(4,64), nn.ReLU(True), C(64,64), nn.ReLU(True), norm_layer(64))
        self.model2 = nn.Sequential(C(64,128), nn.ReLU(True), C(128,128), nn.ReLU(True), norm_layer(128))
        self.model3 = nn.Sequential(C(128,256), nn.ReLU(True), C(256,256), nn.ReLU(True), C(256,256), nn.ReLU(True), norm_layer(256))
        self.model4 = nn.Sequential(C(256,512), nn.ReLU(True), C(512,512), nn.ReLU(True), C(512,512), nn.ReLU(True), norm_layer(512))
        self.model5 = nn.Sequential(C(512,512,3,1,2,2), nn.ReLU(True), C(512,512,3,1,2,2), nn.ReLU(True), C(512,512,3,1,2,2), nn.ReLU(True), norm_layer(512))
        self.model6 = nn.Sequential(C(512,512,3,1,2,2), nn.ReLU(True), C(512,512,3,1,2,2), nn.ReLU(True), C(512,512,3,1,2,2), nn.ReLU(True), norm_layer(512))
        self.model7 = nn.Sequential(C(512,512), nn.ReLU(True), C(512,512), nn.ReLU(True), C(512,512), nn.ReLU(True), norm_layer(512))

        self.model8up = nn.Sequential(nn.ConvTranspose2d(512,256,4,2,1))
        self.model3short8 = nn.Sequential(C(256,256))
        self.model8 = nn.Sequential(nn.ReLU(True), C(256,256), nn.ReLU(True), C(256,256), nn.ReLU(True), norm_layer(256))

        self.model9up = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1))
        self.model2short9 = nn.Sequential(C(128,128))
        self.model9 = nn.Sequential(nn.ReLU(True), C(128,128), nn.ReLU(True), norm_layer(128))

        self.model10up = nn.Sequential(nn.ConvTranspose2d(128,128,4,2,1))
        self.model1short10 = nn.Sequential(C(64,128))
        self.model10 = nn.Sequential(nn.ReLU(True), C(128,128), nn.LeakyReLU(0.2, True))

        self.model_out = nn.Sequential(C(128,2,1,1,0), nn.Tanh())

    def forward(self, input_L, input_ab=None, mask=None):
        if input_ab is None:
            input_ab = torch.cat((input_L*0, input_L*0), dim=1)
        if mask is None:
            mask = input_L*0

        x = torch.cat((self.normalize_l(input_L), self.normalize_ab(input_ab), mask), dim=1)
        c1 = self.model1(x)
        c2 = self.model2(c1[:,:,::2,::2])
        c3 = self.model3(c2[:,:,::2,::2])
        c4 = self.model4(c3[:,:,::2,::2])
        c5 = self.model5(c4)
        c6 = self.model6(c5)
        c7 = self.model7(c6)

        u8 = self.model8up(c7) + self.model3short8(c3)
        c8 = self.model8(u8)
        u9 = self.model9up(c8) + self.model2short9(c2)
        c9 = self.model9(u9)
        u10 = self.model10up(c9) + self.model1short10(c1)
        c10 = self.model10(u10)
        return self.unnormalize_ab(self.model_out(c10))

# ---------------- 4. HELPER FUNCTIONS (REVISED) ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(img_path, size=256):
    """Loads and processes an image for the model, preserving aspect ratio."""
    img = Image.open(img_path).convert("RGB")

    # Store original image for later
    orig_img_display = np.array(img)

    # --- Resize with aspect ratio preservation ---
    # Find the largest dimension and resize it to 'size'
    img.thumbnail((size, size), Image.BICUBIC)

    # Create a new 256x256 image with a black background
    new_img = Image.new("RGB", (size, size), (0, 0, 0))

    # Paste the resized image into the center
    left = (size - img.width) // 2
    top = (size - img.height) // 2
    new_img.paste(img, (left, top))
    # --- End of resize logic ---

    rgb = np.array(new_img) # This is now the 256x256 padded image
    lab = color.rgb2lab(rgb).astype("float32")
    L = lab[..., 0]
    ab = lab[..., 1:]

    # Return the padded L channel AND the original image
    return L, ab, orig_img_display

# ---------------- 5. LOAD MODELS ----------------
# --- Load ECCV16 ---
model_e_pt = ECCVGenerator().to(DEVICE)
ckpt_e = torch.load("/content/best_eccv16_pretrained.pt", map_location=DEVICE)
model_e_pt.load_state_dict(ckpt_e["model_state_dict"] if "model_state_dict" in ckpt_e else ckpt_e)
print("Loaded pretrained ECCV16 model")

# --- Load SIGGRAPH17 ---
model_s_pt = SIGGRAPHGenerator().to(DEVICE)
ckpt_s = torch.load("/content/best_sig17_pretrained.pt", map_location=DEVICE)
model_s_pt.load_state_dict(ckpt_s["model_state_dict"] if "model_state_dict" in ckpt_s else ckpt_s)
print("Loaded pretrained SIGGRAPH17 model")

# ---------------- 6. ENSEMBLE FUNCTION (FIXED) ----------------
def colorize_image_ensemble(model1, model2, img_path, size=256, device=DEVICE):
    # 1. Load image: L is 256x256 padded, orig_img is original HxW
    L, _, orig_img = load_image(img_path, size=size)
    orig_h, orig_w = orig_img.shape[:2]

    L_t = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).to(device) # Shape: [1, 1, 256, 256]

    model1.eval()
    model2.eval()

    with torch.no_grad():
        pred_ab_1 = model1(L_t)
        pred_ab_2 = model2(L_t)
        pred_ab = 0.4 * pred_ab_1 + 0.6 * pred_ab_2

        # 2. Combine with padded L tensor
        out_lab_padded = torch.cat((L_t, pred_ab), dim=1)[0].cpu().numpy().transpose(1,2,0)
        out_rgb_padded = np.clip(color.lab2rgb(out_lab_padded), 0, 1) # This is 256x256

    # 3. Find dimensions to crop out padding
    # We need to know how big the image was *before* padding
    thumb = Image.open(img_path).convert("RGB")
    thumb.thumbnail((size, size), Image.BICUBIC)
    thumb_w, thumb_h = thumb.width, thumb.height

    left = (size - thumb_w) // 2
    top = (size - thumb_h) // 2
    right = left + thumb_w
    bottom = top + thumb_h

    # 4. Crop the 256x256 output to remove padding
    out_rgb_cropped = out_rgb_padded[top:bottom, left:right]

    # 5. Resize this cropped image to the *original* image size
    # Convert 0-1 float to 0-255 uint8 for cv2
    out_rgb_uint8 = (out_rgb_cropped * 255).astype(np.uint8)

    # Use cv2.resize to stretch back to original dimensions
    final_rgb = cv2.resize(out_rgb_uint8, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    return final_rgb, orig_img

# ---------------- 7. RUN INFERENCE ----------------
img_path = "/content/000000330369-copy.jpg" # Make sure this file exists
colored, orig = colorize_image_ensemble(model_e_pt, model_s_pt, img_path, size=256)

# Save output
out_path = "/content/output_colored_ensemble_10.png"
Image.fromarray(colored).save(out_path)
print(f"Saved ensemble colorized image at {out_path}")

# --- Display side by side ---
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
# 'orig' is now the original, full-sized image
plt.imshow(color.rgb2gray(orig), cmap='gray')
plt.title("Original (Grayscale Input)")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(colored)
plt.title("Ensemble Colorized Output (ECCV16 + SIGGRAPH17)")
plt.axis("off")

plt.show()
