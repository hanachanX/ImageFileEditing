"""画像処理：旧版の計算式との一致、境界の条件、旧版で落ちていたケース。"""
import numpy as np
import pytest

from imagetool.imageops import adjust, ascii_art, effects, filters, geometry
from imagetool.imageops.common import block_means, expand_blocks
from . import legacy_ref as L


def close(a, b, tol=1):
    assert a.shape == b.shape
    assert np.abs(a.astype(int) - b.astype(int)).max() <= tol


# ---------------------------------------------------------------- 色・明るさ
def test_gamma_matches_legacy(rgb):
    close(adjust.gamma(rgb, 2.2), L.gamma(rgb, 2.2))


def test_gamma_rejects_nonpositive(rgb):
    with pytest.raises(ValueError):
        adjust.gamma(rgb, 0)


def test_contrast_identical(rgb):
    assert np.array_equal(adjust.contrast(rgb, 1.3), L.contrast(rgb, 1.3))


def test_sepia_matches_legacy(rgb):
    close(adjust.sepia(rgb), L.sepia(rgb))


def test_brightness_clips(rgb):
    out = adjust.brightness(rgb, 100)
    assert out.max() == 255 and out.dtype == np.uint8
    assert np.array_equal(adjust.brightness(rgb, 0), rgb)


def test_rgb_gain_identity_and_channels(rgb):
    assert np.array_equal(adjust.rgb_gain(rgb, 1, 1, 1), rgb)
    out = adjust.rgb_gain(rgb, 0, 1, 1)
    assert out[..., 0].max() == 0 and np.array_equal(out[..., 1:], rgb[..., 1:])


def test_hsv_identity(rgb):
    close(adjust.hsv_adjust(rgb, 0, 1.0, 0), rgb, tol=1)


def test_hsv_hue_shift_red_to_cyan():
    red = np.zeros((4, 4, 3), np.uint8)
    red[..., 0] = 255
    out = adjust.hsv_adjust(red, hue_shift=180)
    assert tuple(out[0, 0]) == (0, 255, 255)


def test_hsv_value_shift_does_not_change_saturation():
    """旧版は明度の加減が彩度にも掛かっていた。灰色に明度だけ足しても色は付かないはず。"""
    grey = np.full((4, 4, 3), 100, np.uint8)
    out = adjust.hsv_adjust(grey, 0, 1.0, 50)
    assert (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all()


def test_posterize_levels(rgb):
    for n in (2, 3, 4):
        assert len(np.unique(adjust.posterize(rgb, n))) <= n


def test_grayscale_has_equal_channels(rgb):
    g = adjust.grayscale(rgb)
    assert g.shape == rgb.shape and (g[..., 0] == g[..., 1]).all() and (g[..., 1] == g[..., 2]).all()


# ---------------------------------------------------------------- フィルタ
def test_clahe_identical(rgb_aligned):
    assert np.array_equal(filters.clahe(rgb_aligned), L.clahe(rgb_aligned))


def test_sharpen_matches_legacy(rgb):
    close(filters.sharpen(rgb), L.sharp(rgb))


def test_pencil_matches_legacy_on_grey_input():
    """旧版は R と B の重みが逆（RGB配列にBGR変換を使っていた）。灰色なら重みの差が出ないので一致する。"""
    g = np.random.default_rng(3).integers(0, 256, (30, 40), dtype=np.uint8)
    img = np.stack([g, g, g], -1)
    out = filters.pencil_sketch(img)
    assert np.array_equal(out[..., 0], L.pencil(img))


def test_pencil_uses_correct_luma_for_colour():
    red = np.zeros((30, 30, 3), np.uint8)
    red[..., 0] = 255
    blue = np.zeros_like(red)
    blue[..., 2] = 255
    # 赤は青より明るい（輝度 0.299 vs 0.114）。旧版は逆だった
    assert filters.pencil_sketch(red).mean() >= filters.pencil_sketch(blue).mean()
    assert cv_gray(red) > cv_gray(blue)


def cv_gray(img):
    import cv2
    return int(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean())


def test_blur_even_kernel_is_fixed_up(rgb):
    out = filters.gaussian_blur(rgb, 6, 1.0)
    assert out.shape == rgb.shape


def test_dilate_erode_order(rgb):
    assert filters.dilate(rgb).sum() >= rgb.sum() >= filters.erode(rgb).sum()


# ---------------------------------------------------------------- 効果
def test_mosaic_matches_legacy_within_rounding(rgb):
    for block in (1, 7, 10, 60):
        close(effects.mosaic(rgb, block), L.mosaic_whole(rgb, block))


def test_mosaic_region_only_changes_region(rgb):
    out = effects.mosaic(rgb, 8, region=(10, 5, 30, 25))
    mask = np.ones(rgb.shape[:2], bool)
    mask[5:25, 10:30] = False
    assert np.array_equal(out[mask], rgb[mask])
    assert not np.array_equal(out[~mask], rgb[~mask])


def test_mosaic_region_at_edge_not_multiple_of_block(rgb):
    """旧版は、範囲の端がブロックの大きさで割り切れないと形が合わず落ちた。"""
    out = effects.mosaic(rgb, 10, region=(3, 3, 50, 35))
    assert out.shape == rgb.shape


def test_mosaic_region_reversed_drag_and_outside(rgb):
    a = effects.mosaic(rgb, 8, region=(30, 25, 10, 5))
    b = effects.mosaic(rgb, 8, region=(10, 5, 30, 25))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        effects.mosaic(rgb, 8, region=(10, 10, 10, 10))


def test_dot_art_identical(rgb):
    assert np.array_equal(effects.dot_art(rgb, 10), L.dot(rgb, 10))


def test_painterly_is_reproducible_with_seed(rgb):
    a = effects.painterly(rgb, seed=5)
    b = effects.painterly(rgb, seed=5)
    c = effects.painterly(rgb, seed=6)
    assert np.array_equal(a, b) and not np.array_equal(a, c) and a.shape == rgb.shape


def test_halftone_bayer_matches_legacy(rgb_aligned):
    for m in (1, 2):
        assert np.array_equal(effects.halftone(rgb_aligned, m)[..., 0], L.bayer(rgb_aligned, m))


@pytest.mark.parametrize("m,od,bs", [(3, effects._ULICHNEY_3, 4), (4, effects._ULICHNEY_4, 8), (5, effects._ULICHNEY_5, 14)])
def test_halftone_ulichney_matches_legacy(rgb_aligned, m, od, bs):
    assert np.array_equal(effects.halftone(rgb_aligned, m)[..., 0], L.ulichney(rgb_aligned, od, bs))


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
def test_halftone_any_size_and_two_levels(rgb, m):
    """旧版は大きさによって落ちたり端が欠けた。新版はどんな大きさでも同じ大きさで返る。"""
    out = effects.halftone(rgb, m)
    assert out.shape == rgb.shape and set(np.unique(out)) <= {0, 255}


def test_halftone_bad_method(rgb):
    with pytest.raises(ValueError):
        effects.halftone(rgb, 9)


# ---------------------------------------------------------------- 変形
def test_normalize_box():
    assert geometry.normalize_box((30, 20, 10, 5), 100, 100) == (10, 5, 30, 20)
    assert geometry.normalize_box((-5, -5, 500, 500), 100, 80) == (0, 0, 100, 80)
    assert geometry.normalize_box((10, 10, 10, 50), 100, 100) is None


def test_crop_shape_and_content(rgb):
    out = geometry.crop(rgb, (30, 20, 2, 5))
    assert out.shape == (15, 28, 3) and np.array_equal(out, rgb[5:20, 2:30])


def test_resize_and_limit(rgb):
    assert geometry.resize(rgb, 2.0).shape == (74, 106, 3)
    with pytest.raises(ValueError):
        geometry.resize(rgb, 0)
    with pytest.raises(ValueError):
        geometry.resize(rgb, 1e6)


def test_flip_rotate(rgb):
    assert np.array_equal(geometry.flip_horizontal(geometry.flip_horizontal(rgb)), rgb)
    assert np.array_equal(geometry.rotate_left(geometry.rotate_right(rgb)), rgb)
    assert geometry.rotate_right(rgb).shape == (53, 37, 3)


# ---------------------------------------------------------------- 共通・アスキー
def test_block_means_edges_use_actual_pixels():
    a = np.arange(5 * 5 * 3, dtype=np.uint8).reshape(5, 5, 3)
    m = block_means(a, 2)
    assert m.shape == (3, 3, 3)
    assert np.allclose(m[2, 2], a[4:5, 4:5].reshape(-1, 3).mean(0))
    assert expand_blocks(m, 2, 5, 5).shape == (5, 5, 3)


def test_ascii_dimensions_and_range(rgb):
    txt = ascii_art.to_ascii(rgb, 40)
    lines = txt.splitlines()
    assert len(lines) > 0 and all(set(l) <= set(ascii_art.CHARSET) for l in lines)
    with pytest.raises(ValueError):
        ascii_art.to_ascii(rgb, 10)
