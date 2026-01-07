# ensemble vs stock model accuracies
# ensemble for photos

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

# ---------------- 4. HELPER FUNCTIONS ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(img_path, size=256):
    """Loads and processes an image for the model."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    rgb = np.array(img)
    lab = color.rgb2lab(rgb).astype("float32")
    L = lab[..., 0]
    ab = lab[..., 1:]
    return rgb, L, ab

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
    rgb, L, _ = load_image(img_path, size=size)
    L_t = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).to(device) # Shape: [1, 1, 256, 256]

    model1.eval()
    model2.eval()

    with torch.no_grad():
        pred_ab_1 = model1(L_t) # Outputs [1, 2, 256, 256]
        pred_ab_2 = model2(L_t) # Outputs [1, 2, 256, 256]

        # --- ensemble of ab outputs ---
        # Your weighted average:
        pred_ab = 0.4 * pred_ab_1 + 0.6 * pred_ab_2

        # --- BUG FIX: Remove unnecessary interpolate ---
        # Both models already output 256x256, so pred_ab is [1, 2, 256, 256]
        # pred_ab_up = torch.nn.functional.interpolate(pred_ab, size=(L.shape[0], L.shape[1]), mode='bilinear') # <-- REMOVED

        # Concatenate original L tensor with the final predicted ab tensor
        out_lab = torch.cat((L_t, pred_ab), dim=1)[0].cpu().numpy().transpose(1,2,0)
        out_rgb = np.clip(color.lab2rgb(out_lab), 0, 1)
        return (out_rgb * 255).astype(np.uint8), (rgb)


# ---------------- 8. VALIDATION DATASET SETUP ----------------

# --- Helper function from your training script ---
def rgb_to_lab_8u(img_rgb_uint8):
    lab = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    L = L.astype(np.float32) / 2.55  # scale back to [0,100]
    A = A.astype(np.float32) - 128.0
    B = B.astype(np.float32) - 128.0
    return L, A, B

# --- Dataset class from your training script ---
class ColorizationImageDataset(Dataset):
    def __init__(self, files, img_size=256):
        self.files = files
        self.size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((self.size, self.size), Image.BICUBIC)
            rgb = np.array(img, dtype=np.uint8)
            L, A, B = rgb_to_lab_8u(rgb)
            L_t = torch.from_numpy(L).unsqueeze(0).float()
            ab_t = torch.from_numpy(np.stack([A, B], 0)).float()
            return L_t, ab_t
        except Exception as e:
            print(f"Warning: Skipping file {path} due to error: {e}")
            return None # Will be filtered by collate_fn

# --- Custom collate function to filter Nones ---
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

# --- Setup DataLoader (assuming same path as training) ---
VAL_ROOT = "/content/images/val/val2017"
IMG_EXTS = {".jpg", ".jpeg", ".png"}

try:
    val_files = [str(p) for p in Path(VAL_ROOT).rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not val_files:
        print(f"Warning: No validation files found at {VAL_ROOT}. Skipping validation.")
        val_loader = None
    else:
        val_ds = ColorizationImageDataset(val_files, img_size=256)
        # Use batch_size 16-32 for validation
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)
        print(f"Loaded {len(val_files)} validation files.")
except FileNotFoundError:
    print(f"Error: Validation directory not found at {VAL_ROOT}. Skipping validation.")
    val_loader = None


# ---------------- 9. FULL VALIDATION SCRIPT ----------------

if val_loader:
    print("\nStarting validation on the full dataset...")

    # Set models to evaluation mode
    model_e_pt.eval()
    model_s_pt.eval()

    # Trackers for all 3 models
    total_loss_e, total_loss_s, total_loss_ens = 0.0, 0.0, 0.0
    total_acc_e, total_acc_s, total_acc_ens = 0.0, 0.0, 0.0

    with torch.no_grad():
        for L, ab in tqdm(val_loader, desc="Validating"):
            # Filter out empty batches from collate_fn
            if L is None or ab is None:
                continue

            L, ab = L.to(DEVICE), ab.to(DEVICE)

            # 1. Get predictions from both models
            pred_ab_e = model_e_pt(L)
            pred_ab_s = model_s_pt(L)

            # 2. Get ensemble prediction
            pred_ab_ens = 0.4 * pred_ab_e + 0.6 * pred_ab_s

            # 3. Calculate Loss (using L2/MSE as in your training)
            # .item() gets the scalar value from the tensor
            total_loss_e += F.mse_loss(pred_ab_e, ab).item()
            total_loss_s += F.mse_loss(pred_ab_s, ab).item()
            total_loss_ens += F.mse_loss(pred_ab_ens, ab).item()

            # 4. Calculate Accuracy (using your <10 unit metric)
            total_acc_e += torch.mean((torch.abs(pred_ab_e - ab) < 10.0).float()).item()
            total_acc_s += torch.mean((torch.abs(pred_ab_s - ab) < 10.0).float()).item()
            total_acc_ens += torch.mean((torch.abs(pred_ab_ens - ab) < 10.0).float()).item()

    # 5. Average results
    num_batches = len(val_loader)
    avg_loss_e = total_loss_e / num_batches
    avg_loss_s = total_loss_s / num_batches
    avg_loss_ens = total_loss_ens / num_batches

    avg_acc_e = (total_acc_e / num_batches) * 100
    avg_acc_s = (total_acc_s / num_batches) * 100
    avg_acc_ens = (total_acc_ens / num_batches) * 100

    # 6. Print the final report
    print("\n" + "="*30)
    print("  Full Validation Report")
    print("="*30)
    print(f"  Validation Set Size: {len(val_ds)} images")
    print(f"  Batch Size: 16, Batches: {num_batches}")
    print("\n--- Average L2 Loss (MSE) ---")
    print(f"  ECCV16:     {avg_loss_e:.4f}")
    print(f"  SIGGRAPH17: {avg_loss_s:.4f}")
    print(f"  Ensemble:   {avg_loss_ens:.4f}")
    print("\n--- Average Accuracy (<10 LAB units) ---")
    print(f"  ECCV16:     {avg_acc_e:.2f}%")
    print(f"  SIGGRAPH17: {avg_acc_s:.2f}%")
    print(f"  Ensemble:   {avg_acc_ens:.2f}%")
    print("="*30)
else:
    print("Validation loader not initialized. Cannot run full validation.")
