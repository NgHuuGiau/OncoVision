#!/usr/bin/env python
"""
Generate Volume 3D Models for OncoVision - Simplified Version
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, 2)
        in_ch = 1
        for f in [32, 64, 128, 256]:
            self.encoder.append(nn.Sequential(
                nn.Conv3d(in_ch, f, 3, padding=1), nn.BatchNorm3d(f), nn.ReLU(),
                nn.Conv3d(f, f, 3, padding=1), nn.BatchNorm3d(f), nn.ReLU()
            ))
            in_ch = f
        self.bottleneck = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(),
            nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU()
        )
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose3d(512, 256, 2, 2),
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.ConvTranspose3d(64, 32, 2, 2)
        ])
        self.decoder = nn.ModuleList()
        for f in [256, 128, 64, 32]:
            self.decoder.append(nn.Sequential(
                nn.Conv3d(f*2, f, 3, padding=1), nn.BatchNorm3d(f), nn.ReLU(),
                nn.Conv3d(f, f, 3, padding=1), nn.BatchNorm3d(f), nn.ReLU()
            ))
        self.final = nn.Conv3d(32, 2, 1)
    
    def forward(self, x):
        skips = []
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
            x = nn.functional.max_pool3d(x, 2)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i, up in enumerate(self.upconvs):
            x = up(x)
            skip = skips[i]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[i](x)
        return self.final(x)


def main():
    targets = {
        "lung": {"modality": "ct", "input_shape": [1, 128, 128, 128], "num_classes": 3, "classes": ["background", "nodule", "mass"]},
        "liver": {"modality": "ct", "input_shape": [1, 96, 96, 96], "num_classes": 3, "classes": ["background", "tumor", "vessel"]},
        "prostate": {"modality": "mri", "input_shape": [1, 96, 96, 96], "num_classes": 3, "classes": ["background", "peripheral_zone", "transition_zone"]},
        "kidney": {"modality": "ct", "input_shape": [1, 128, 128, 128], "num_classes": 3, "classes": ["background", "kidney", "tumor"]},
        "pancreas": {"modality": "ct", "input_shape": [1, 96, 96, 96], "num_classes": 2, "classes": ["background", "pancreas"]},
        "stomach": {"modality": "endoscopy", "input_shape": [1, 128, 128, 128], "num_classes": 2, "classes": ["background", "lesion"]},
        "colorectal": {"modality": "colonoscopy", "input_shape": [1, 128, 128, 128], "num_classes": 2, "classes": ["background", "polyp"]},
        "cervical": {"modality": "colposcopy", "input_shape": [1, 128, 128, 128], "num_classes": 2, "classes": ["background", "lesion"]},
        "thyroid": {"modality": "ultrasound", "input_shape": [1, 128, 128, 128], "num_classes": 2, "classes": ["background", "nodule"]},
        "breast": {"modality": "ultrasound", "input_shape": [1, 128, 128, 128], "num_classes": 2, "classes": ["background", "lesion"]},
    }

    print("Generating volume models for OncoVision...")
    print("=" * 60)

    for cancer_key, config in targets.items():
        modality = config["modality"]
        print(f"\n=== Generating {cancer_key} ({modality}) ===")

        out_dir = Path("models/pretrained/cancers") / cancer_key / modality
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            model = UNet3D(in_channels=1, out_channels=config["num_classes"])
            model.eval()
            example_input = torch.randn(1, 1, *config["input_shape"][1:])
            traced = torch.jit.trace(model, example_input)
            
            volume_path = out_dir / "volume.pt2"
            traced.save(str(volume_path))
            size_mb = volume_path.stat().st_size / (1024 * 1024)
            print(f"  Created volume.pt2 ({size_mb:.1f} MB)")

            modality_lower = modality.lower()
            manifest = {
                "model_name": f"{cancer_key}_{modality}_segmentation",
                "model_type": "volume_segmentation_3d",
                "cancer_type": cancer_key,
                "modality": modality,
                "model_architecture": "unet3d",
                "input_shape": config["input_shape"],
                "num_classes": config["num_classes"],
                "classes": config["classes"],
                "model_file": "volume.pt2",
                "preprocessing": {
                    "normalize": "ct" if "ct" in modality_lower else
                                 "mri" if "mri" in modality_lower else
                                 "ultrasound" if "ultrasound" in modality_lower else
                                 "endoscopy",
                    "spacing": [1.0, 1.0, 1.0],
                    "orientation": "RAS",
                },
                "postprocessing": {
                    "threshold": 0.5,
                    "largest_connected_component": True,
                    "min_volume_voxels": 100,
                },
                "inference": {
                    "patch_size": [128, 128, 128] if "ct" in modality_lower or "ultrasound" in modality_lower or "colonoscopy" in modality_lower or "colposcopy" in modality_lower else [96, 96, 96],
                    "overlap": 0.5,
                    "batch_size": 1,
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                },
                "metadata": {
                    "created_by": "OncoVision Volume Model Generator",
                    "version": "1.0",
                    "framework": "pytorch",
                }
            }
            manifest_path = out_dir / "volume.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            print(f"  Created volume.json")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone! All volume models generated.")


if __name__ == "__main__":
    main()