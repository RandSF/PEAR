from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PearConfig:
    config_path: Path
    checkpoint_path: Path
    assets_path: Path
    device: str = "cuda"
    crop_size: int = 256
    crop_scale: float = 1.25


@dataclass
class PearOutput:
    body_params: dict[str, np.ndarray]
    flame_params: dict[str, np.ndarray]
    camera: np.ndarray  # [4, 4]
    vertices: np.ndarray  # [V, 3]
    joints: np.ndarray  # [J, 3]
    bbox_xyxy: np.ndarray  # [4]


def enable_chumpy_compatibility() -> None:
    '''Provide removed Python and NumPy aliases required while unpickling body models.'''
    import inspect

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    aliases = {
        "bool": bool, "int": int, "float": float, "complex": complex,
        "object": object, "unicode": str, "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def crop_person(image: np.ndarray, bbox_xyxy: np.ndarray, size: int, scale: float) -> np.ndarray:
    '''Crop RGB person image to [size, size, 3].'''
    x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    src = np.asarray([
        [cx - side / 2, cy - side / 2],
        [cx + side / 2, cy - side / 2],
        [cx - side / 2, cy + side / 2],
    ], dtype=np.float32)
    dst = np.asarray([[0, 0], [size - 1, 0], [0, size - 1]], dtype=np.float32)
    transform = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(
        image, transform, (size, size), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


class PearPredictor:
    def __init__(self, cfg: PearConfig):
        enable_chumpy_compatibility()
        import torch
        import torch.nn as nn
        import yaml
        from omegaconf import OmegaConf

        from models.backbones import ViT
        from models.modules.ehm import EHM_v2
        from models.smplx.smplx_head import SMPLXTransformerDecoderHead

        class Regressor(nn.Module):
            def __init__(self, model_cfg, mean_params_path):
                super().__init__()
                self.backbone = ViT(**model_cfg.BACKBONE)
                self.head = SMPLXTransformerDecoderHead(
                    model_cfg.HEAD, 1, mean_params_path=mean_params_path
                )
                self.register_buffer(
                    "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
                )
                self.register_buffer(
                    "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
                )

            def forward(self, x):
                x = (x - self.mean) / self.std
                return self.head(self.backbone(x[:, :, :, 32:-32]))

        self.cfg = cfg
        self.device = torch.device(cfg.device)
        model_cfg = OmegaConf.create(yaml.safe_load(cfg.config_path.read_text()))
        mean_params_path = cfg.assets_path / "SMPLX" / "smpl_mean_params.npz"
        self.regressor = Regressor(model_cfg, mean_params_path)
        state = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
        self.regressor.backbone.load_state_dict(state["backbone"], strict=False)
        self.regressor.head.load_state_dict(state["head"], strict=False)
        self.body_model = EHM_v2(
            str(cfg.assets_path / "FLAME"), str(cfg.assets_path / "SMPLX")
        )
        self.regressor.to(self.device).eval()
        self.body_model.to(self.device).eval()

    def __call__(self, image: np.ndarray, boxes_xyxy: np.ndarray) -> list[PearOutput]:
        '''Infer one RGB image with boxes [N, 4].'''
        import torch

        if len(boxes_xyxy) == 0:
            return []
        crops = np.stack([
            crop_person(image, box, self.cfg.crop_size, self.cfg.crop_scale)
            for box in boxes_xyxy
        ])
        x = torch.from_numpy(crops).permute(0, 3, 1, 2).float().div(255).to(self.device)
        outputs = []
        with torch.inference_mode():
            for i, box in enumerate(boxes_xyxy):
                prediction = self.regressor(x[i:i + 1])
                mesh = self.body_model(
                    prediction["body_param"], prediction["flame_param"], pose_type="aa"
                )
                outputs.append(PearOutput(
                    body_params=self._numpy_dict(prediction["body_param"]),
                    flame_params=self._numpy_dict(prediction["flame_param"]),
                    camera=prediction["pd_cam"][0].float().cpu().numpy(),
                    vertices=mesh["vertices"][0].float().cpu().numpy(),
                    joints=mesh["joints"][0].float().cpu().numpy(),
                    bbox_xyxy=np.asarray(box, dtype=np.float32),
                ))
        return outputs

    @staticmethod
    def _numpy_dict(data):
        return {
            key: value[0].float().cpu().numpy()
            for key, value in data.items()
            if value is not None
        }
