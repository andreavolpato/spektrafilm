from skimage.transform import resize

def resize_for_preview(image, max_size):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale_factor = max_size / max(h, w)
        return resize(image, (int(h * scale_factor), int(w * scale_factor)),
                      preserve_range=True, anti_aliasing=True, order=1).astype(image.dtype)
    else:
        return image