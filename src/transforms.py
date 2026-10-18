from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class AspectPreservingResizePad:
    """Resize an image while preserving aspect ratio, then pad to a square."""

    def __init__(self, target_size=224, fill=0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        scale = self.target_size / max(width, height)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        image = TF.resize(
            image,
            [new_height, new_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        pad_width = self.target_size - new_width
        pad_height = self.target_size - new_height
        left = pad_width // 2
        right = pad_width - left
        top = pad_height // 2
        bottom = pad_height - top

        return TF.pad(image, [left, top, right, bottom], fill=self.fill)
