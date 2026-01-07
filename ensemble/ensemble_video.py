# ensemble for videos

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from tqdm import tqdm
import os
import cv2 # <-- ADD THIS IMPORT

# ---------------- 8. NEW FUNCTIONS FOR VIDEO ----------------

def process_frame(rgb_frame, size=256):
    """Loads and processes an in-memory BGR frame for the model."""
    # Convert BGR (OpenCV default) to RGB
    rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

    # Resize using OpenCV, matching PIL's BICUBIC
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)

    # Process like load_image
    lab = color.rgb2lab(rgb).astype("float32")
    L = lab[..., 0]
    return rgb, L

def colorize_frame_ensemble(model1, model2, bgr_frame, size=256, device=DEVICE):
    """Runs the ensemble on a single BGR video frame."""

    # 1. Process the frame (resize, convert to LAB, get L channel)
    rgb, L = process_frame(bgr_frame, size=size)
    L_t = torch.from_numpy(L).unsqueeze(0).unsqueeze(0).to(device) # Shape: [1, 1, 256, 256]

    # Models are already in eval mode from file loading, but good practice:
    model1.eval()
    model2.eval()

    with torch.no_grad():
        pred_ab_1 = model1(L_t) # Outputs [1, 2, 256, 256]
        pred_ab_2 = model2(L_t) # Outputs [1, 2, 256, 256]

        # 2. Ensemble the 'ab' outputs
        pred_ab = 0.4 * pred_ab_1 + 0.6 * pred_ab_2

        # 3. Combine with original L tensor
        out_lab = torch.cat((L_t, pred_ab), dim=1)[0].cpu().numpy().transpose(1,2,0)

        # 4. Convert back to RGB
        out_rgb_float = np.clip(color.lab2rgb(out_lab), 0, 1)
        out_rgb_uint8 = (out_rgb_float * 255).astype(np.uint8)

        # 5. Convert back to BGR for OpenCV
        out_bgr = cv2.cvtColor(out_rgb_uint8, cv2.COLOR_RGB2BGR)

        return out_bgr

# ---------------- 9. RUN VIDEO COLORIZATION (FIXED) ----------------

# --- Define your video paths ---
VIDEO_IN_PATH = "/content/Untitled video - Made with Clipchamp (2).mp4"
VIDEO_OUT_PATH = "/content/output_video_colorized_3.mp4"

# --- Open the input video ---
cap = cv2.VideoCapture(VIDEO_IN_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_IN_PATH}")
else:
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- CHANGE 1: Define output size using original dimensions ---
    output_size = (frame_width, frame_height)

    # --- CHANGE 2: Use original output_size for the writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, output_size)

    # We still process at 256x256, but the output video will be the original size
    processing_size = 256
    print(f"Processing video: {VIDEO_IN_PATH}")
    print(f"Original Res: {frame_width}x{frame_height}, Processing Res: {processing_size}x{processing_size}")

    # Use tqdm for a progress bar
    for _ in tqdm(range(frame_count), desc="Colorizing frames"):
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Colorize the frame at 256x256
        # colorized_frame is (256, 256)
        colorized_frame = colorize_frame_ensemble(model_e_pt, model_s_pt, frame, size=processing_size)

        # --- CHANGE 3: Resize the (256, 256) output back to the original video size ---
        final_frame = cv2.resize(colorized_frame, output_size, interpolation=cv2.INTER_CUBIC)

        # --- CHANGE 4: Write the resized final_frame ---
        out.write(final_frame)

    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nVideo processing complete. Saved to: {VIDEO_OUT_PATH}")
