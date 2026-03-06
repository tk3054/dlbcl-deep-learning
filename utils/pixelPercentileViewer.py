import numpy as np
import tifffile as tiff
import matplotlib
from matplotlib.widgets import Slider
from pathlib import Path


def _ensure_interactive_backend():
    """Ensure slider widgets are draggable by using a GUI backend."""
    backend = matplotlib.get_backend().lower()
    if "inline" not in backend:
        return

    for candidate in ("MacOSX", "TkAgg", "QtAgg"):
        try:
            matplotlib.use(candidate, force=True)
            return
        except Exception:
            continue

    raise RuntimeError(
        "Detected inline Matplotlib backend, but no GUI backend is available. "
        "Install a GUI backend (Tk/Qt) or run with an interactive backend."
    )


_ensure_interactive_backend()
import matplotlib.pyplot as plt


def percentile_viewer(tif_path):
    # --- load image ---
    img = tiff.imread(tif_path).astype(np.float32)

    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {img.shape}")

    non_zero_mask = img > 0
    non_zero_pixels = img[non_zero_mask]

    if non_zero_pixels.size == 0:
        raise ValueError("Image has no nonzero pixels.")

    # initial slider values
    init_lower = 1.0
    init_upper = 99.0

    def compute_views(lower_pct, upper_pct):
        p_low = np.percentile(non_zero_pixels, lower_pct)
        p_high = np.percentile(non_zero_pixels, upper_pct)

        low_clip_mask = np.zeros_like(img, dtype=bool)
        high_clip_mask = np.zeros_like(img, dtype=bool)
        included_mask = np.zeros_like(img, dtype=bool)
        low_clip_mask[non_zero_mask] = non_zero_pixels < p_low
        high_clip_mask[non_zero_mask] = non_zero_pixels > p_high
        included_mask[non_zero_mask] = (
            (non_zero_pixels >= p_low) & (non_zero_pixels <= p_high)
        )

        normalized = np.zeros_like(img, dtype=np.float32)
        if p_high > p_low:
            vals = (non_zero_pixels - p_low) / (p_high - p_low)
            normalized[non_zero_mask] = np.clip(vals, 0, 1)
        else:
            normalized[non_zero_mask] = 0.5

        return p_low, p_high, low_clip_mask, high_clip_mask, included_mask, normalized

    p_low, p_high, low_clip_mask, high_clip_mask, included_mask, normalized = compute_views(
        init_lower, init_upper
    )

    # --- figure layout ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    plt.subplots_adjust(bottom=0.22)

    ax_orig, ax_low, ax_high, ax_norm = axes

    im0 = ax_orig.imshow(img, cmap="gray")
    ax_orig.set_title("Original image")
    ax_orig.axis("off")

    im1 = ax_low.imshow(low_clip_mask, cmap="gray", vmin=0, vmax=1)
    ax_low.set_title("Clipped low (< lower %)")
    ax_low.axis("off")

    im2 = ax_high.imshow(high_clip_mask, cmap="gray", vmin=0, vmax=1)
    ax_high.set_title("Clipped high (> upper %)")
    ax_high.axis("off")

    im3 = ax_norm.imshow(normalized, cmap="gray", vmin=0, vmax=1)
    ax_norm.set_title("Normalized image")
    ax_norm.axis("off")

    non_zero_total = int(non_zero_mask.sum())
    low_count = int(low_clip_mask.sum())
    high_count = int(high_clip_mask.sum())
    in_count = int(included_mask.sum())

    # text readout
    info_text = fig.text(
        0.5,
        0.08,
        (
            f"lower={init_lower:.1f}%, upper={init_upper:.1f}% | "
            f"P_low={p_low:.3f}, P_high={p_high:.3f} | "
            f"nonzero={non_zero_total} | "
            f"low_clipped={low_count} ({100.0 * low_count / non_zero_total:.2f}%) | "
            f"high_clipped={high_count} ({100.0 * high_count / non_zero_total:.2f}%) | "
            f"included={in_count} ({100.0 * in_count / non_zero_total:.2f}%)"
        ),
        ha="center",
        fontsize=10,
    )

    # --- sliders ---
    ax_lower = plt.axes([0.2, 0.12, 0.6, 0.03])
    ax_upper = plt.axes([0.2, 0.06, 0.6, 0.03])

    s_lower = Slider(ax_lower, "Lower %", 0.0, 99.9, valinit=init_lower, valstep=0.1)
    s_upper = Slider(ax_upper, "Upper %", 0.1, 100.0, valinit=init_upper, valstep=0.1)

    def update(_):
        lower = s_lower.val
        upper = s_upper.val

        if lower >= upper:
            return

        p_low, p_high, low_clip_mask, high_clip_mask, included_mask, normalized = compute_views(
            lower, upper
        )
        low_count = int(low_clip_mask.sum())
        high_count = int(high_clip_mask.sum())
        in_count = int(included_mask.sum())

        im1.set_data(low_clip_mask)
        im2.set_data(high_clip_mask)
        im3.set_data(normalized)

        info_text.set_text(
            f"lower={lower:.1f}%, upper={upper:.1f}% | "
            f"P_low={p_low:.3f}, P_high={p_high:.3f} | "
            f"nonzero={non_zero_total} | "
            f"low_clipped={low_count} ({100.0 * low_count / non_zero_total:.2f}%) | "
            f"high_clipped={high_count} ({100.0 * high_count / non_zero_total:.2f}%) | "
            f"included={in_count} ({100.0 * in_count / non_zero_total:.2f}%)"
        )

        fig.canvas.draw_idle()

    s_lower.on_changed(update)
    s_upper.on_changed(update)

    plt.show()


if __name__ == "__main__":
    tif_path = '/Users/taeeonkong/Desktop/DL Project/non-responder/01-03-2026 DLBCL 109241/sample1/1/padded_cells/cell_02_padded.tif'
    percentile_viewer(tif_path)
