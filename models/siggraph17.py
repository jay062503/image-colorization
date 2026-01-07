# siggraph with pretrained weights

import torch
import torch.nn as nn

class BaseColor(nn.Module):
    def __init__(self):
        super(BaseColor, self).__init__()
        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.

    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm


class SIGGRAPHGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d, classes=529):
        super(SIGGRAPHGenerator, self).__init__()

        def C(in_c, out_c, k=3, s=1, p=1, d=1):
            return nn.Conv2d(in_c, out_c, k, s, p, dilation=d, bias=True)

        self.model1 = nn.Sequential(
            C(4,64), nn.ReLU(True),
            C(64,64), nn.ReLU(True),
            norm_layer(64)
        )
        self.model2 = nn.Sequential(
            C(64,128), nn.ReLU(True),
            C(128,128), nn.ReLU(True),
            norm_layer(128)
        )
        self.model3 = nn.Sequential(
            C(128,256), nn.ReLU(True),
            C(256,256), nn.ReLU(True),
            C(256,256), nn.ReLU(True),
            norm_layer(256)
        )
        self.model4 = nn.Sequential(
            C(256,512), nn.ReLU(True),
            C(512,512), nn.ReLU(True),
            C(512,512), nn.ReLU(True),
            norm_layer(512)
        )
        self.model5 = nn.Sequential(
            C(512,512,3,1,2,2), nn.ReLU(True),
            C(512,512,3,1,2,2), nn.ReLU(True),
            C(512,512,3,1,2,2), nn.ReLU(True),
            norm_layer(512)
        )
        self.model6 = nn.Sequential(
            C(512,512,3,1,2,2), nn.ReLU(True),
            C(512,512,3,1,2,2), nn.ReLU(True),
            C(512,512,3,1,2,2), nn.ReLU(True),
            norm_layer(512)
        )
        self.model7 = nn.Sequential(
            C(512,512), nn.ReLU(True),
            C(512,512), nn.ReLU(True),
            C(512,512), nn.ReLU(True),
            norm_layer(512)
        )

        # Decoder
        self.model8up = nn.Sequential(nn.ConvTranspose2d(512,256,4,2,1))
        self.model3short8 = nn.Sequential(C(256,256))
        self.model8 = nn.Sequential(
            nn.ReLU(True),
            C(256,256), nn.ReLU(True),
            C(256,256), nn.ReLU(True),
            norm_layer(256)
        )

        self.model9up = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1))
        self.model2short9 = nn.Sequential(C(128,128))
        self.model9 = nn.Sequential(
            nn.ReLU(True),
            C(128,128), nn.ReLU(True),
            norm_layer(128)
        )

        self.model10up = nn.Sequential(nn.ConvTranspose2d(128,128,4,2,1))
        self.model1short10 = nn.Sequential(C(64,128))
        self.model10 = nn.Sequential(
            nn.ReLU(True),
            C(128,128), nn.LeakyReLU(0.2, True)
        )

        self.model_out = nn.Sequential(
            C(128,2,1,1,0), nn.Tanh()
        )

    def forward(self, input_L, input_ab=None, mask=None):
        if input_ab is None:
            input_ab = torch.cat((input_L*0, input_L*0), dim=1)
        if mask is None:
            mask = input_L*0

        x = torch.cat((self.normalize_l(input_L), input_ab, mask), dim=1)
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



from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import torch
import cv2
from pathlib import Path

# --- your helper functions ---
def rgb_to_lab_8u(img_rgb_uint8):
    lab = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    L = L.astype(np.float32) / 2.55  # scale back to [0,100]
    A = A.astype(np.float32) - 128.0
    B = B.astype(np.float32) - 128.0
    return L, A, B


def norm_LAB_for_net(L_uint8, A_uint8, B_uint8):
    # L = (L_uint8.astype(np.float32) / 255.0)
    # a = ((A_uint8.astype(np.float32) - 128.0) / 128.0)
    # b = ((B_uint8.astype(np.float32) - 128.0) / 128.0)
    # return L, a, b
    return L_uint8, A_uint8, B_uint8

# --- dataset ---
class ColorizationImageDataset(Dataset):
    def __init__(self, files, img_size=256):
        self.files = files
        self.size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.size, self.size), Image.BICUBIC)
        rgb = np.array(img, dtype=np.uint8)
        L, A, B = rgb_to_lab_8u(rgb)
        Lf, af, bf = norm_LAB_for_net(L, A, B)
        L_t = torch.from_numpy(Lf).unsqueeze(0).float()
        ab_t = torch.from_numpy(np.stack([af, bf], 0)).float()
        return L_t, ab_t, path

# --- load files ---
from sklearn.model_selection import train_test_split
from pathlib import Path
# IMG_EXTS = {".jpg", ".jpeg", ".png"}
# DATA_ROOT = "/kaggle/input/coco-validation-2017/val2017"
# all_files = [str(p) for p in Path(DATA_ROOT).rglob("*") if p.suffix.lower() in IMG_EXTS]

# train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=42)
# train_ds = ColorizationImageDataset(train_files, img_size=256)
# val_ds   = ColorizationImageDataset(val_files,   img_size=256)

# train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
# val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

TRAIN_ROOT = "/content/images/train/test2017"
VAL_ROOT   = "/content/images/val/val2017"

IMG_EXTS = {".jpg", ".jpeg", ".png"}

train_files = [str(p) for p in Path(TRAIN_ROOT).rglob("*") if p.suffix.lower() in IMG_EXTS]
val_files   = [str(p) for p in Path(VAL_ROOT).rglob("*") if p.suffix.lower() in IMG_EXTS]

train_ds = ColorizationImageDataset(train_files, img_size=256)
val_ds   = ColorizationImageDataset(val_files,   img_size=256)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2, pin_memory=True)



import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------ Training ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
CKPT_DIR = "/content"
os.makedirs(CKPT_DIR, exist_ok=True)

model_s_pt = SIGGRAPHGenerator().to(DEVICE)

# ---- Load pretrained SIGGraph17 weights ----
weights = torch.hub.load_state_dict_from_url(
    'https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth',
    map_location='cpu', check_hash=True
)
model_s_pt.load_state_dict(weights, strict=False)
print("Loaded pretrained SIGGraph17 weights.")

# ---- Optimizer ----
opt = torch.optim.AdamW(model_s_pt.parameters(), lr=1e-5, weight_decay=1e-4)
# scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
scaler = torch.amp.GradScaler("cuda", enabled=False)
best_val_loss = float("inf")

train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(1, EPOCHS + 1):
    model_s_pt.train()
    train_loss, train_acc = 0.0, 0.0
    for L, ab, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
        L, ab = L.to(DEVICE), ab.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred_ab = model_s_pt(L)
            loss = F.mse_loss(pred_ab, ab)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        train_loss += loss.item() * L.size(0)
        train_acc += torch.mean((torch.abs(pred_ab - ab) < 10.0).float()).item()
    train_loss /= len(train_ds)
    train_acc /= len(train_loader)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ---------- Validation ----------
    model_s_pt.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for L, ab, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
            L, ab = L.to(DEVICE), ab.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                pred_ab = model_s_pt(L)
                loss = F.mse_loss(pred_ab, ab)
            val_loss += loss.item() * L.size(0)
            val_acc += torch.mean((torch.abs(pred_ab - ab) < 10.0).float()).item()
    val_loss /= len(val_ds)
    val_acc /= len(val_loader)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"\nEpoch {epoch}: Train L2 = {train_loss:.4f} | Train Acc = {train_acc*100:.2f}% | Val L2 = {val_loss:.4f} | Val Acc = {val_acc*100:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_s_pt.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "val_loss": val_loss,
        }, f"{CKPT_DIR}/best_sig17_pretrained.pt")
        print(f"Saved new best model at epoch {epoch} (Val L2 = {val_loss:.4f} | Val Acc = {val_acc*100:.2f}%)")

print("\nTraining complete.")
print(f"Lowest validation loss: {best_val_loss:.4f}")

# ---------- Plot ----------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(range(1,EPOCHS+1), train_losses, label="Train L2")
plt.plot(range(1,EPOCHS+1), val_losses, label="Val L2")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch"); plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1,EPOCHS+1), train_accs, 'b-', label="Train Accuracy")
plt.plot(range(1,EPOCHS+1), val_accs, 'g-', label="Validation Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy vs Epoch"); plt.legend()

plt.tight_layout()
plt.savefig(f"{CKPT_DIR}/sig17_pretrained_curves.png")
print(f"Saved training graphs to {CKPT_DIR}/sig17_pretrained_curves.png")



import torch, cv2, numpy as np
from PIL import Image
from skimage import color
import torch.nn.functional as F

# --- Load image (grayscale or color) ---
def load_image(img_path, size=256):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    rgb = np.array(img)
    lab = color.rgb2lab(rgb).astype("float32")
    L = lab[..., 0]
    ab = lab[..., 1:]
    return rgb, L, ab

# --- Convert LAB → RGB uint8 ---
def lab_to_rgb_uint8(L, ab):
  # L: (1,1,H,W) in 0..100 ; ab: (1,2,H,W) in original units
  L_ = L.squeeze().cpu().numpy()             # (H,W)
  ab_ = ab.squeeze().cpu().numpy()           # (2,H,W)
  lab = np.stack([L_, ab_[0], ab_[1]], axis=-1).astype(np.float32)  # (H,W,3)
  rgb = np.clip(color.lab2rgb(lab), 0, 1)
  return (rgb * 255).astype(np.uint8)

# --- Colorize with pretrained model ---
def colorize_image(model, img_path, size=256, device="cuda" if torch.cuda.is_available() else "cpu"):
    rgb, L, _ = load_image(img_path, size=size)
    L_t = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_ab = model(L_t)                       # (1,2,H,W)
        pred_ab_up = F.interpolate(pred_ab, size=(L.shape[0], L.shape[1]), mode='bilinear')
        out_lab = torch.cat((L_t, pred_ab_up), dim=1)[0].cpu().numpy().transpose(1,2,0)
        out_rgb = np.clip(color.lab2rgb(out_lab), 0, 1)
        return (out_rgb * 255).astype(np.uint8)


# from skimage import color
# import numpy as np



# pred_ab = run_siggraph_forward(model, "/kaggle/input/coco-validation-2017/val2017/000000001268.jpg", 256)
# l_input, _ = prepare_siggraph_inputs("/kaggle/input/coco-validation-2017/val2017/000000001268.jpg", size=256, device=next(model.parameters()).device)
# rgb_uint8 = lab_to_rgb_uint8(l_input, pred_ab)

# Image.fromarray(rgb_uint8)




img_path = "/content/images/train/test2017/000000001286.jpg"
colored = colorize_image(model_s_pt, img_path, size=256)
Image.fromarray(colored).save("/content/output_colored_s_pt.png")
Image.fromarray(colored)
