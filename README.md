# Function Description #
This app is a tool for managing files created in large quantities by StableDiffusion (AUTOMATIC1111). It also has simple image editing functions. It does not work on Mac. It extracts image generation information embedded in PNG files. It is a fast and efficient image viewer. Unwanted works can be deleted with the Delete key. You can send image files to your favorite folder using the right arrow key. You can switch images using the up and down arrow keys. The information of the source folder and destination folder, once loaded, is recorded in a file and will be automatically loaded in subsequent sessions.

Known Issue: The deleted files in the Explorer are not reflected in the list box. Please just reload App.

* Version 1.8
I have added an experimental feature to upscale images using super-resolution. Please note that this process can be very slow without a GPU and there may be some noise in the output. This feature is strictly experimental. Please make sure to place Common.py, Models.py, and CARN_model.pt in the same directory as Image.py before running it. Also note that the program has not been converted to an executable file yet.    

* Version 1.7
I have merged the four windows into two. I have fixed a critical bug where file selection was incorrect. I have also updated the UI to be entirely in English.


* Version 1.6
I have created a destination window for file transfer. Now, you can send files using the "→" button. I have also implemented the functionality to open the currently displayed image file in the File Explorer. Additionally, I have enabled the main window's image to be zoomed in by 1.05 times when left-clicked. Various bugs have been fixed as well.

* Version 1.5
    Added support for stable diffusion(Automatic1111). Display prompt and image generation information for PNG files. Added clipboard copying functionality.

* Version 1.4
    Add cropping feature

* Version 1.3
    Adding sepia tone conversion functionality.
    Adding flip feature.

* Version 1.2
    convert Gray_scale feature addition
    
* Version 1.1
    Gamma correction feature addition

* Version 1.0
    Version 1.0 is a tool that allows you to draw and delete a large number of image files. It comes with features such as hue conversion, blur, mosaic, resize, and pencil sketch. 
    You can use the up and down buttons to select and draw files. Press the Delete key to delete files. 
    It also has real-time monitoring of image files that are generated automatically by AI and continue to accumulate.

***
StableDiffusion(AUTOMATIC1111)で生成された大量の画像ファイル（JPGないしPNGのみ対応）の内容確認と削除を実行を目的としたツール。
リストボックスにファイル一覧が表示されますので上下キーで画像を確認し削除したい場合はDeleteキーで削除が実行されます。
このリストボックスはディレクトリをリアルタイムで監視しているのでAI画像自動生成などをしてファイルが追加されたらリアルタイムでリストボックスに反映されます。
簡単な画像加工ツールがあります

* バージョン1.1　- ガンマ補正機能追加

* バージョン1.2　- グレースケール変換機能追加

* バージョン1.3　- セピア調変換、ミラー変換機能追加

* バージョン1.4　- トリミング機能の追加（トリミングの選択で画像左クリック＆ドラッグが有効化）

* バージョン1.5　- Stable Diffusion(Automatic1111)に対応。PNGファイルからプロンプト、生成情報を抽出、表示、クリップボードへ送る機能を追加。
メインウインドウ以外を非表示にする機能を追加。

* バージョン1.6 - ファイルの転送先のウインドウを作成しました。→ボタンで送ることができるようになりました。
現在表示中の画像のファイルをエクスプローラーで開くことができるようにしました。
プロンプト、ネガティブプロンプト情報をクリップボードに送るボタンを設置しました。
メインウインドウの画像を左クリックすると1.05倍づつ拡大するようにしました。リストボックスと送り先フォルダ（お気に入り？傑作？）の情報は"dir_config.ini"というファイルに記録されるので
次回の起動時にいちいちフォルダを指定する必要がありません。その他バグを修正しました。

* バージョン1.7 - 4つあったウインドウを統合し2つにしました。ファイルの選択がずれている重大なバグを修正しました。UIを全て英語表記にしました。

* バージョン1.8　- 実験的に超解像度にアップスケールする機能を追加しました。GPUがないとすごく時間がかかるのでお勧めしません。ノイズが混じることがありあくまでも実験的なものです。Exe化はしていません。Common.py , Models.py , CARN_model.ptをImage.pyと同一フォルダに配置して実行してください。google driveの実行ファイルはバージョン1.7のものです。

* 必要な追加ライブラリのインストール：python.exe -m pip install -r requirements.txt
* 実行ファイル:https://drive.google.com/file/d/1PQ4PhxMu0VRu1GH2LUvx2tiS7RqotvnA/view?usp=sharing
* 実行ファイルをGoogle Driveに置いておきますがセキュリティの問題があるのでできればご自身でEXE化してください。

ビルド

pip install pyinstaller

pyinstaller --noconsole --windowed --name Image Image.Py

アプリの画面　いたってシンプルです。
![起動画面](image/view.png)
