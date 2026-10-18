import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Minimal Grad-CAM implementation for CNN feature layers."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inputs, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(image_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam[0, 0].detach().cpu().numpy(), logits.detach().cpu()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def create_content_mask(image_path, target_size=224):
    """Create the non-padding mask produced by aspect-preserving resize+pad."""
    with Image.open(image_path) as img:
        width, height = img.size

    scale = target_size / max(width, height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2

    mask = np.zeros((target_size, target_size), dtype=np.float32)
    mask[top : top + new_height, left : left + new_width] = 1.0
    return mask
