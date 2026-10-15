"""Real-ESRGAN の推論（旧 RealESRGAN/model.py の整理版）。"""
import os

import numpy as np
import torch
from PIL import Image

from .rrdbnet_arch import RRDBNet
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, unpad_image

HF_MODELS = {
    2: dict(repo_id="sberbank-ai/Real-ESRGAN", filename="RealESRGAN_x2.pth"),
    4: dict(repo_id="sberbank-ai/Real-ESRGAN", filename="RealESRGAN_x4.pth"),
    8: dict(repo_id="sberbank-ai/Real-ESRGAN", filename="RealESRGAN_x8.pth"),
}


def _download(scale, model_path):
    """重みが無いときだけ Hugging Face から取得する。"""
    from huggingface_hub import hf_hub_download
    cfg = HF_MODELS[scale]
    folder = os.path.dirname(model_path)
    got = hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["filename"], local_dir=folder)
    if os.path.abspath(got) != os.path.abspath(model_path):
        os.replace(got, model_path)


class RealESRGAN:
    def __init__(self, device, scale=4):
        self.device = device
        self.scale = scale
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                             num_block=23, num_grow_ch=32, scale=scale)

    def load_weights(self, model_path, download=True):
        if not os.path.exists(model_path):
            if not download or self.scale not in HF_MODELS:
                raise FileNotFoundError(model_path)
            _download(self.scale, model_path)
        loadnet = torch.load(model_path, map_location="cpu")
        for key in ("params", "params_ema"):
            if key in loadnet:
                loadnet = loadnet[key]
                break
        self.model.load_state_dict(loadnet, strict=True)
        self.model.eval()
        self.model.to(self.device)

    def predict(self, lr_image, batch_size=4, patches_size=192, padding=24, pad_size=15, progress=None):
        """lr_image: PIL.Image または RGB の numpy 配列。progress(done, total) を渡すと進み具合を通知する。"""
        scale, device = self.scale, self.device
        lr = pad_reflect(np.array(lr_image), pad_size)
        patches, p_shape = split_image_into_overlapping_patches(lr, patch_size=patches_size, padding_size=padding)
        img = torch.FloatTensor(patches / 255).permute((0, 3, 1, 2)).to(device).detach()
        total = img.shape[0]
        outs = []
        use_amp = torch.device(device).type == "cuda"
        with torch.no_grad():
            for i in range(0, total, batch_size):
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    outs.append(self.model(img[i:i + batch_size]).float())
                if progress:
                    progress(min(i + batch_size, total), total)
        res = torch.cat(outs, 0)
        sr = res.permute((0, 2, 3, 1)).clamp_(0, 1).cpu().numpy()
        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr.shape[0:2], scale)) + (3,)
        sr = stich_together(sr, padded_image_shape=padded_size_scaled,
                            target_shape=scaled_image_shape, padding_size=padding * scale)
        sr = (sr * 255).astype(np.uint8)
        sr = unpad_image(sr, pad_size * scale)
        return Image.fromarray(sr)
