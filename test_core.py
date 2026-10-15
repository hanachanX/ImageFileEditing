"""画面に依存しない部分：文書、設定、メタデータ、ファイル操作、プロンプト、処理の登録表。"""
import json
import os
import time
import zipfile

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from imagetool import files, metadata, operations, promptgen
from imagetool.document import Document
from imagetool.settings import Settings, load_settings


# ---------------------------------------------------------------- Document
def test_document_undo_redo(rgb):
    d = Document(5)
    d.load_array(rgb)
    assert not d.can_undo
    d.apply(rgb + 1)
    d.apply(rgb + 2)
    assert d.undo() and np.array_equal(d.image, rgb + 1)
    assert d.undo() and np.array_equal(d.image, rgb)
    assert not d.undo()
    assert d.redo() and d.redo() and not d.redo()
    assert np.array_equal(d.image, rgb + 2)


def test_document_new_edit_clears_redo(rgb):
    d = Document()
    d.load_array(rgb)
    d.apply(rgb + 1)
    d.undo()
    d.apply(rgb + 5)
    assert not d.can_redo


def test_document_undo_limit(rgb):
    d = Document(3)
    d.load_array(rgb)
    for i in range(10):
        d.apply(rgb + i)
    n = 0
    while d.undo():
        n += 1
    assert n == 3


def test_document_apply_without_image_raises(rgb):
    with pytest.raises(ValueError):
        Document().apply(rgb)


def test_document_apply_normalizes_dtype(rgb):
    d = Document()
    d.load_array(rgb)
    d.apply(rgb[..., 0].astype(np.uint8))          # グレー1チャンネルでも受け付ける
    assert d.image.shape == rgb.shape and d.image.dtype == np.uint8


def test_document_load_file_and_save_formats(tmp_path, rgb):
    src = tmp_path / "a.png"
    Image.fromarray(rgb).save(src)
    d = Document()
    d.load_file(src)
    assert d.path == src and d.size == (53, 37) and np.array_equal(d.image, rgb)
    for ext in (".png", ".jpg", ".bmp"):
        d.save(tmp_path / f"out{ext}")
        assert Image.open(tmp_path / f"out{ext}").size == (53, 37)
    with pytest.raises(ValueError):
        d.save(tmp_path / "out.gif")


def test_document_load_rgba_and_palette(tmp_path):
    Image.new("RGBA", (5, 4), (1, 2, 3, 128)).save(tmp_path / "a.png")
    Image.new("P", (5, 4)).save(tmp_path / "p.png")
    d = Document()
    for n in ("a.png", "p.png"):
        d.load_file(tmp_path / n)
        assert d.image.shape == (4, 5, 3)


def test_document_load_missing_file(tmp_path):
    with pytest.raises(OSError):
        Document().load_file(tmp_path / "nope.png")


# ---------------------------------------------------------------- Settings
def test_settings_roundtrip(tmp_path):
    p = tmp_path / "settings.json"
    s = Settings(source_dir="C:/a", dest_dir="C:/b", confirm_move=False)
    s.save(p)
    t = load_settings(p)
    assert (t.source_dir, t.dest_dir, t.confirm_move) == ("C:/a", "C:/b", False)
    assert not (tmp_path / "settings.json.tmp").exists()


def test_settings_imports_legacy_ini(tmp_path):
    ini = tmp_path / "dir_config.ini"
    ini.write_text("D:\\src\nD:\\dst\n", encoding="utf-8")
    s = load_settings(tmp_path / "settings.json", ini)
    assert (s.source_dir, s.dest_dir) == ("D:\\src", "D:\\dst")


def test_settings_json_wins_over_legacy(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"source_dir": "X"}), encoding="utf-8")
    ini = tmp_path / "dir_config.ini"
    ini.write_text("A\nB\n", encoding="utf-8")
    assert load_settings(tmp_path / "settings.json", ini).source_dir == "X"


def test_settings_corrupt_or_wrong_types_fall_back(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{broken", encoding="utf-8")
    assert load_settings(p) == Settings()
    p.write_text(json.dumps({"confirm_move": "yes", "undo_limit": "many", "unknown": 1, "source_dir": "ok"}), encoding="utf-8")
    s = load_settings(p)
    assert s.confirm_move is True and s.undo_limit == 20 and s.source_dir == "ok"


# ---------------------------------------------------------------- メタデータ
A1111 = "masterpiece, 1girl, (smile:1.2)\nNegative prompt: lowres, bad hands\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 123"


def test_parse_parameters_full():
    g = metadata.parse_parameters(A1111)
    assert g.prompt == "masterpiece, 1girl, (smile:1.2)"
    assert g.negative == "lowres, bad hands"
    assert g.other.startswith("Steps: 20") and g.other.endswith("Seed: 123")


def test_parse_parameters_no_negative():
    g = metadata.parse_parameters("a cat\nSteps: 5, Seed: 1")
    assert (g.prompt, g.negative, g.other) == ("a cat", "", "Steps: 5, Seed: 1")


def test_parse_parameters_only_steps_line():
    g = metadata.parse_parameters("Steps: 5, Seed: 1")
    assert g.prompt == "" and g.other == "Steps: 5, Seed: 1"


def test_parse_parameters_multiline_prompt_and_japanese():
    """旧版は日本語（非ASCII）を全部消し、複数行のプロンプトも崩していた。"""
    g = metadata.parse_parameters("青い空,\n白い雲\nNegative prompt: 低品質,\nぼやけ\nSteps: 20, Seed: 9")
    assert g.prompt == "青い空,\n白い雲" and g.negative == "低品質,\nぼやけ" and g.other == "Steps: 20, Seed: 9"


def test_parse_parameters_word_negative_inside_prompt_not_confused():
    g = metadata.parse_parameters("a Negative prompt: x\nSteps: 1")
    assert g.negative == "" and "Negative prompt" in g.prompt


def test_read_gen_info_png_roundtrip(tmp_path, rgb):
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", A1111)
    Image.fromarray(rgb).save(tmp_path / "a.png", pnginfo=meta)
    assert metadata.read_gen_info(tmp_path / "a.png").negative == "lowres, bad hands"


def test_read_gen_info_utf8_text_chunk(tmp_path, rgb):
    meta = PngImagePlugin.PngInfo()
    meta.add_itxt("parameters", "猫\nNegative prompt: 犬\nSteps: 1")
    Image.fromarray(rgb).save(tmp_path / "a.png", pnginfo=meta)
    g = metadata.read_gen_info(tmp_path / "a.png")
    assert g.prompt == "猫" and g.negative == "犬"


def test_read_gen_info_utf8_bytes_stored_in_text_chunk(tmp_path, rgb):
    """UTF-8 の生バイトが tEXt に入っている画像（文字化けしやすい）も読める。"""
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "猫\nNegative prompt: 犬\nSteps: 1".encode("utf-8").decode("latin-1"))
    Image.fromarray(rgb).save(tmp_path / "a.png", pnginfo=meta)
    g = metadata.read_gen_info(tmp_path / "a.png")
    assert g.prompt == "猫" and g.negative == "犬"


def test_read_gen_info_none_cases(tmp_path, rgb):
    Image.fromarray(rgb).save(tmp_path / "plain.png")
    Image.fromarray(rgb).save(tmp_path / "a.jpg")
    (tmp_path / "bad.png").write_bytes(b"not a png")
    for n in ("plain.png", "a.jpg", "bad.png", "missing.png"):
        assert not metadata.read_gen_info(tmp_path / n)


# ---------------------------------------------------------------- ファイル
def _imgs(folder, names):
    for n in names:
        Image.new("RGB", (4, 4)).save(folder / n)


def test_list_images_filters_and_sorts(tmp_path):
    _imgs(tmp_path, ["b.png", "A.jpg", "c.JPEG"])
    (tmp_path / "x.txt").write_text("x")
    (tmp_path / "sub.png").mkdir()               # フォルダは含めない
    assert files.list_images(tmp_path) == ["A.jpg", "b.png", "c.JPEG"]
    assert files.list_images(tmp_path / "none") == []


def test_list_images_by_time(tmp_path):
    _imgs(tmp_path, ["a.png", "b.png", "c.png"])
    now = time.time()
    for i, n in enumerate(["c.png", "a.png", "b.png"]):
        os.utime(tmp_path / n, (now + i, now + i))
    assert files.list_images(tmp_path, "time") == ["c.png", "a.png", "b.png"]


def test_move_file(tmp_path):
    s, d = tmp_path / "s", tmp_path / "d"
    s.mkdir(), d.mkdir()
    _imgs(s, ["a.png"])
    files.move_file(s, "a.png", d)
    assert (d / "a.png").exists() and not (s / "a.png").exists()
    _imgs(s, ["a.png"])
    with pytest.raises(FileExistsError):
        files.move_file(s, "a.png", d)
    assert (s / "a.png").exists()                # 失敗しても元は残る


def test_rename_file(tmp_path):
    _imgs(tmp_path, ["a.png", "b.png"])
    assert files.rename_file(tmp_path, "a.png", "z").name == "z.png"      # 拡張子を消しても保つ
    with pytest.raises(FileExistsError):
        files.rename_file(tmp_path, "z.png", "b.png")
    for bad in ("", "  ", "a/b.png", "a:b.png", ".."):
        with pytest.raises(ValueError):
            files.rename_file(tmp_path, "z.png", bad)


def test_archive_images(tmp_path):
    _imgs(tmp_path, ["a.png", "b.jpg"])
    out = files.archive_images(tmp_path)
    assert out.name == tmp_path.name + ".zip"
    assert sorted(zipfile.ZipFile(out).namelist()) == ["a.png", "b.jpg"]
    assert (tmp_path / "a.png").exists()                                  # 元は消さない
    with pytest.raises(ValueError):
        files.archive_images(tmp_path / "empty_none", [])


def test_delete_to_trash_never_hard_deletes_off_windows(tmp_path):
    _imgs(tmp_path, ["a.png"])
    if os.name != "nt":
        with pytest.raises(OSError):
            files.delete_to_trash(tmp_path / "a.png")
        assert (tmp_path / "a.png").exists()
    with pytest.raises(FileNotFoundError):
        files.delete_to_trash(tmp_path / "missing.png")


# ---------------------------------------------------------------- プロンプト
def test_format_term():
    assert promptgen.format_term("a") == "a"
    assert promptgen.format_term("a", 2) == "((a))"
    assert promptgen.format_term("a", 0, 1.2) == "a:1.2"
    assert promptgen.format_term("a", 1, 1.2) == "(a:1.2)"
    with pytest.raises(ValueError):
        promptgen.format_term("a", 5)


def test_prompt_builder_undo_clear():
    b = promptgen.PromptBuilder()
    b.add("a")
    b.add("b", 1)
    assert b.text == "a, (b)"
    assert b.undo() == "a"
    b.undo()
    assert b.undo() == ""                         # 空で戻しても落ちない
    b.add("x")
    b.clear()
    assert b.text == ""


def test_load_prompt_words(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("\ufeffMasterpiece\n\n  HDR  \r\n", encoding="utf-8")
    assert promptgen.load_prompt_words(p) == ["Masterpiece", "HDR"]
    assert promptgen.load_prompt_words(tmp_path / "none.txt") == []


# ---------------------------------------------------------------- 処理の登録表
def test_every_operation_runs_and_keeps_format(rgb):
    for op in operations.OPERATIONS:
        out = operations.run(op, rgb)
        assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3, op.id
        assert out is not rgb


def test_operation_ids_unique_and_defaults_in_range():
    ids = [o.id for o in operations.OPERATIONS]
    assert len(ids) == len(set(ids))
    for op in operations.OPERATIONS:
        for p in op.params:
            if p.kind != "choice":
                assert p.lo <= p.default <= p.hi, (op.id, p.key)


def test_run_does_not_modify_input(rgb):
    before = rgb.copy()
    for op in operations.OPERATIONS:
        operations.run(op, rgb)
    assert np.array_equal(rgb, before)


def test_selection_is_passed_only_to_ops_that_use_it(rgb):
    out = operations.run(operations.BY_ID["mosaic"], rgb, {"block": 8}, region=(0, 0, 16, 16))
    assert np.array_equal(out[20:], rgb[20:])
    operations.run(operations.BY_ID["gray"], rgb, region=(0, 0, 16, 16))      # 無視されるだけで落ちない
