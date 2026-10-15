# IMAGE FILE TOOL  (Version 4.0)

Stable Diffusion (AUTOMATIC1111) で大量に作った画像を整理するためのツールです。簡単な画像編集もできます。**Windows 専用**。

## 使い方

```
pip install -r requirements.txt
pythonw.exe Image.pyw        # または Image.pyw をダブルクリック
```

AUTOMATIC1111 の `venv` の `pythonw.exe` で `Image.pyw` を開くようにしておくと、追加のインストールなしで動きます。
デスクトップにショートカットを置くと便利です。起動に失敗したときは `image_tool_error.txt` に理由が書かれます。

超解像（CARN / Real-ESRGAN）を使うときだけ、追加で `pip install -r requirements-upscale.txt` が必要です。
使わない場合は PyTorch が無くても起動します。

## できること

- **ファイル一覧**（左＝送り元／右＝送り先）：↑↓で画像を切り替え、`Delete` でゴミ箱へ、`→` で右のフォルダへ移動。
  フォルダの中身は自動で一覧に反映されます（エクスプローラーでの削除・追加も）。
- **生成情報の表示**：PNG に埋め込まれたプロンプト / ネガティブプロンプト / 設定を表示し、コピーできます。
- **編集**：色・明るさ、ぼかしなどのフィルタ、モザイク（ドラッグで範囲を選べます）、ハーフトーン、ドットアート、
  絵画風、拡大縮小、反転・回転、切り抜き。**元に戻す／やり直し** は最大20回（設定で変更可）。
- **超解像**：CARN_V2（軽い）／ Real-ESRGAN（きれい）。
- **アスキーアート**、**プロンプト作成**（`prompt.txt` の言葉を選んでつなげる）、**ZIPにまとめる**。

操作のコツ：画像をクリックすると「画面に合わせる」と「原寸」が切り替わります。画像の上をドラッグすると範囲選択
（モザイクと切り抜きに使われます）。`Esc` で選択解除。右クリックで編集メニュー。

## ファイル構成

```
Image.pyw            起動用
imagetool/
  imageops/          画像処理（画面に依存しない・テスト済み）
  operations.py      処理の一覧表（メニューと設定ダイアログはここから自動で作られる）
  document.py        編集中の画像と「元に戻す」
  settings.py  files.py  metadata.py  promptgen.py  clipboard.py
  upscale/           超解像（PyTorch は使うときまで読み込まない）
  ui/                画面
tests/               自動テスト
prompt.txt           プロンプト作成用の言葉（自由に編集してください）
CARN_model.pt        CARN の重み
weights/             Real-ESRGAN の重み（無ければ初回に自動ダウンロード）
```

新しい画像処理を足したいときは、`imagetool/imageops/` に関数を書き、`imagetool/operations.py` の `OPERATIONS` に
1行足すだけで、メニューとパラメータ調整ダイアログが自動でできます。

## テスト

```
pip install -r requirements-dev.txt
python -m pytest tests
```

## 変更履歴

### 4.0
内部を全面的に作り直しました（見た目と操作はほぼそのままです）。2,400行あった `Image.pyw` を役割ごとのファイルに分け、
画像処理に自動テストを付けました。あわせて、旧版にあった不具合を修正しています。

- 設定は `settings.json` に保存されます（旧版の `dir_config.ini` は初回だけ自動で取り込みます）。
- 削除はゴミ箱へ送ります。
- 重い処理は別スレッドで実行するので、実行中も画面が固まりません。
- 修正した不具合の一覧は `CHANGES-4.0.md` を参照してください。

### 3.1 以前
3.1 明るさ・シャープ・膨張・収縮・CLAHE ／ 3.0 ハーフトーンに Ulichney 法を追加（参考: https://github.com/SankarSrin/Digital-Halftoning）／
2.9 右一覧の名前順・日時順 ／ 2.8 ドットアート・絵画風 ／ 2.7 プロンプト作成 ／ 2.6 ZIP圧縮 ／ 2.5 ハーフトーン ／
2.4 アスキーアート ／ 2.3 コントラスト ／ 以前は旧 README を参照。
