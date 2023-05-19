The author is a beginner in programming!!!
# Image File Editing and Image Processing
Version 1.5
    -Added support for stable diffusion(Automatic1111). Display prompt and image generation information for PNG files. Added clipboard copying functionality.

Version 1.4
    -Add cropping feature

Version 1.3
    -Adding sepia tone conversion functionality.
-Adding flip feature.

Version 1.2
    -convert Gray_scale feature addition
    
Version 1.1
    -Gamma correction feature addition

Version 1.0 (learning of tkinter for me)
    -Version 1.0 is a tool that allows you to draw and delete a large number of image files. It comes with features such as hue conversion, blur, mosaic, resize, and pencil sketch. 
    You can use the up and down buttons to select and draw files. Press the Delete key to delete files. 
    It also has real-time monitoring of image files that are generated automatically by AI and continue to accumulate.
    
大量の画像ファイル（JPGないしPNGのみ対応）の内容確認と削除を実行するツール。リストボックスにファイル一覧が表示されますので上下キーで画像を確認し削除したい場合はDeleteキーで
削除が実行されます。このリストボックスはディレクトリをリアルタイムで監視しているのでAI画像自動生成などをしてファイルが追加されたらリアルタイムでリストボックスに反映されます。
リストボックスで右クリックすれば画像の名前を編集できます。
簡単な加工ツールがあります（ガウシアンぼかし、モザイク、拡大縮小、鉛筆画、RGB調整）

バージョン1.1　- ガンマ補正機能追加

バージョン1.2　- グレースケール変換機能追加

バージョン1.3　- セピア調変換、ミラー変換機能追加

バージョン1.4　- トリミング機能の追加（トリミングの選択で画像左クリック＆ドラッグが有効化）

バージョン1.5　- Stable Diffusion(Automatic1111)に対応。PNGファイルからプロンプト、生成情報を抽出、表示、クリップボードへ送る機能を追加。
メインウインドウ以外を非表示にする機能を追加。

追加ライブラリのインストール：python.exe -m pip install -r requirements.txt
