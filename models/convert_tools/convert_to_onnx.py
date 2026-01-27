import os
import torch
import torch.nn as nn
from pathlib import Path
from sam2.build_sam import build_sam2

def export_medsam2_split(checkpoint_path, output_dir="models"):
    print("\n" + "=" * 60)
    print("STEP 4: Exporting Split ONNX Components (Encoder & Decoder)")
    print("=" * 60)
    
    # 1. Modell laden
    config = "sam2_hiera_t.yaml" # MedSAM2 nutzt meist Tiny
    model = build_sam2(config, ckpt_path=None, device="cpu")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- TEIL A: IMAGE ENCODER ---
    # Dieser Teil wandelt das 1024x1024 Bild in Embeddings um
    print("\n--- Exporting Image Encoder ---")
    image_encoder = model.image_encoder
    dummy_img = torch.randn(1, 3, 1024, 1024)
    
    encoder_path = output_path / "medsam2_encoder.onnx"
    torch.onnx.export(
        image_encoder,
        dummy_img,
        str(encoder_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['image_embeddings', 'backbone_fpn_0', 'backbone_fpn_1'],
        dynamic_axes={'image': {0: 'batch_size'}}
    )
    print(f"✓ Encoder exported: {encoder_path}")

    # --- TEIL B: MASK DECODER (Interaktiver Teil) ---
    # Dieser Teil nimmt die Embeddings + deine Box/Punkte und gibt die Maske aus
    print("\n--- Exporting Mask Decoder ---")
    
    # Wir erstellen einen Wrapper, da der Decoder im Original tief verschachtelt ist
    class Sam2DecoderWrapper(nn.Module):
        def __init__(self, sam2_model):
            super().__init__()
            self.decoder = sam2_model.sam_mask_decoder
            self.prompt_encoder = sam2_model.sam_prompt_encoder

        def forward(self, image_embeddings, point_coords, point_labels):
            # Berechnet Prompt-Embeddings (Box/Punkte)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            # Berechnet die eigentliche Maske
            low_res_masks, iou_predictions, _, _ = self.decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False
            )
            return low_res_masks, iou_predictions

    decoder_model = Sam2DecoderWrapper(model)
    
    # Dummy Inputs für Decoder (1 Punkt/Box-Ecke)
    dummy_embeds = torch.randn(1, 256, 64, 64)
    dummy_coords = torch.randn(1, 2, 2) # [batch, points, xy]
    dummy_labels = torch.ones(1, 2)     # [batch, labels]

    decoder_path = output_path / "medsam2_decoder.onnx"
    torch.onnx.export(
        decoder_model,
        (dummy_embeds, dummy_coords, dummy_labels),
        str(decoder_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['image_embeddings', 'point_coords', 'point_labels'],
        output_names=['masks', 'iou_predictions'],
        dynamic_axes={
            'point_coords': {1: 'num_points'},
            'point_labels': {1: 'num_points'}
        }
    )
    print(f"✓ Decoder exported: {decoder_path}")

# In deinem main() einfach aufrufen:
# export_medsam2_split("models/MedSAM2_latest.pt")
