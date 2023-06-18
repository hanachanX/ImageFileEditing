from Models import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter.font import Font
import os
from PIL import Image , ImageTk , ImageGrab , ImageDraw
from PIL.Image import Resampling
import cv2
import numpy as np
import io
import win32clipboard
import logging
from tkinter import messagebox
from tkinter import simpledialog
from collections import deque
import threading
import time
import signal
import re
import zipfile
import subprocess
import random
import torch
import concurrent.futures
from torchvision.transforms.functional import to_tensor
from RealESRGAN import RealESRGAN

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s' , level=logging.WARNING , encoding='utf-8')
logger = logging.getLogger(__name__)


def check_text_chunk(filepath):
    try:
        with open(filepath, 'rb') as file:
            signature = file.read(8)
            if signature != b'\x89PNG\r\n\x1a\n': # 8バイトシグネチャがPNG形式でない。
                return False
            data = file.read(0x100)
            if b'\x69\x54\x58\x74' in data or b'\x74\x45\x58\x74' in data:
                logger.debug('exist tEXt chunk')
                return True
                # file.seek(length + 4, 1)  # チャンクデータとCRC4バイトはスキップする。
            logger.debug('not exist tEXt chunk')
            return False
    except IOError:
        messagebox.showerror("Error", "Failed to read the file.")
        return False

def sepia(image):
    # NumPy配列に変換
    img_np = np.array(image)

    # セピアカラーに変換するための行列計算
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])

    # 行列計算を行い、セピア変換を適用
    sepia_image = np.dot(img_np, sepia_matrix.T)

    # 値を0から255の範囲に制限する
    sepia_image = np.clip(sepia_image, 0, 255)

    # NumPy配列からcv2の画像形式に変換
    sepia_image = sepia_image.astype(np.uint8)

    return sepia_image

def dodgeV2(x,y):
    return cv2.divide(x, 255 - y, scale=256)

def pencilsketch(inp_img):
    img_gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smoothing = cv2.GaussianBlur(img_invert, (21,21), sigmaX=0, sigmaY=0)
    final_img = dodgeV2(img_gray, img_smoothing)
    return final_img

def send_to_clipboard(clip_type, data):
    # クリップボードをクリアして、データをセットする
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(clip_type, data)
    win32clipboard.CloseClipboard()

def tk_to_cv2(tk_image):
    # ImageTkオブジェクトをPIL.Imageオブジェクトに変換
    pil_image = ImageTk.getimage(tk_image)

    # PIL.ImageオブジェクトをNumPy配列に変換
    cv2_rgb_image = np.array(pil_image)
    
    # RGB -> BGRによりCV2画像オブジェクトに変換
    cv2_image = cv2.cvtColor(cv2_rgb_image, cv2.COLOR_RGB2BGR)

    return cv2_image

def cv2_to_pil(image):

    ret = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    return Image.fromarray(ret)

def tk_to_pil(image):
    ret = tk_to_cv2(image)
    return cv2_to_pil(ret)

def tk_to_tensor(image):
    return torch.from_numpy(np.array(ImageTk.getimage(image)).transpose((2,0,1)))

def adjust_contrast(img, alpha=1.0, beta=0.0):
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

def bayers(img,met):
    
    # Bayers HalfToning Algorithm https://imageprocessing-sankarsrin.blogspot.com/2018/05/bayers-digital-halftoning-dispersed-and.html
    
    img = np.array(ImageTk.getimage(img)).astype(np.float32)/255
    img = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
    s1, s2 = img.shape
    s1 = s1 // 8 * 8
    s2 = s2 // 8 * 8
    img = img[:s1 , :s2]
    
    if met == 1:
        # Bayers Dispersed Dot
        DA = np.array([[0,48,12,60,3,51,15,63],
                       [32,16,44,28,35,19,47,31],
                       [8,56,4,52,11,59,7,55],
                       [40,24,36,20,43,27,39,23],
                       [2,50,14,62,1,49,13,61],
                       [34,18,46,30,33,17,45,29],
                       [10,58,6,54,9,57,5,53],
                       [42,26,38,22,41,25,37,21]] , dtype=np.float32)/64.0


    else:
        # Bayers Clustered Dot
        DA = np.array([[24,10,12,26,35,47,49,37],
                       [8,0,2,14,45,59,61,51],
                       [22,6,4,16,43,57,63,53],
                       [30,20,18,28,33,41,55,39],
                       [34,46,48,36,25,11,13,27],
                       [44,58,60,50,9,1,3,15],
                       [42,56,62,52,23,7,5,17],
                       [32,40,54,38,31,21,19,29]] , dtype=np.float32)/64.0


    mask = np.tile(DA,(s1//8, s2//8))
    HOD = img < mask
    ret = HOD.astype(np.uint8)*255
    ret = cv2.bitwise_not(ret)

    return ret

def sort_files_by_timestamp(files , dir):
    """
    ファイル名のリストを、最終更新時刻の昇順にソートする関数
    """
    # ファイル名と最終更新時刻を格納する辞書オブジェクトを作成
    file_timestamps = {}
    for file in files:
        path = os.path.join(dir, file)
        file_timestamps[file] = os.path.getmtime(path)

    # 最終更新時刻を昇順にソートしたリストを生成
    sorted_files = [file for file, _ in sorted(file_timestamps.items(), key=lambda x: x[1])]
    return sorted_files

class AAEngine():
    def __init__(self , width , image):
        self.window = None
        self.user_width = width
        self.image = image
        self.block_size = self.image.width() // self.user_width
        self.mapping = None
        self.obj = None
        
    def create(self):
        self.window = tk.Toplevel()
        self.window.title('ASCII ART CANVAS')
        self.window.resizable(False,False)
        width = self.image.width()
        height = self.image.height()
        img = np.array(ImageTk.getimage(self.image))
        self.var = tk.BooleanVar()
        self.var.set(True)
        self.menu = tk.Menu(self.window)
        self.menu.add_command(label='invert' , command=self.on_invert)
        self.window.config(menu=self.menu)
        self.label = None
        # color_set = 'ＭＷ８６５３＜；・．　'[::-1]
        color_set = 'ＭＷＮ＄＠％＃＆Ｂ８９ＥＧＡ６ｍＫ５ＨＲｋｂＹＴ４３Ｖ０ＪＬ７ｇｐａｓｅｙｘｚｎｏｃｖ？ｊＩｆｔｒ１ｌｉ＊＝－～＾｀’：；，．　'[::-1]
        # color_set = '＠Ｗ％ＱＭＢ＆ＮｍＤＲＧＯ８ｇＨＥｗＳＫＡ＄６９０ｄｂＺｑＵｐ５ＰＸ＃ａＣｈ４Ｖ２３ｅｋＦｏｕｎｙＴＹｓｚｘ７１＊ＬＪＩｆｊｔｖｃ＜［］｛＞｝＝？＋／（＼）ｒｌ～！ｉ｜＂＿；－：，＾＇｀．　'[::-1]
        # color_set = 'Ｂ＃ＥＲ＝ＤＭＷ％Ｚ＄ＰＧＱ＆２５ＨＦ＠Ｓ８Ｎ４Ｏ＋Ｋ９６ｍｚＣｇＡ０ｅｗ３ＴｄｂＵＬＸｐｑ［］Ｖ７ｈｋ－ａｔ＜＞＊ｆｏｓｙ？ＩＹｕＪｎ｝｛ｃｘ～ｊｖ｜１／＼ｒｌ！）（ｉ＿＾：；＂｀＇．，　'[::-1]
        # color_set = 'ＢＥ＝Ｍ％＄Ｇ＆５ＦＳＮＯＫ６ｚｇ０ｗＴｂＬｐ［Ｖｈ－ｔ＞ｆｓ？ＹＪ｝ｃ～ｖ１＼ｌ）ｉ＾；｀．　'[::-1]
        num_colors = len(color_set)
        self.mapping = np.zeros((height // self.block_size+1, width // self.block_size+1), dtype=int)
        for y in range(0, height, self.block_size):
            for x in range(0, width, self.block_size):
                roi = img[y:y+self.block_size, x:x+self.block_size]

                # ROIの範囲が画像をはみ出す場合の処理
                if roi.shape != (self.block_size, self.block_size):
                    roi = img[y:min(y+self.block_size, height), x:min(x+self.block_size, width)]

                mean = np.mean(roi)
                index = int(mean / 255 * (num_colors - 1))
                index = max(0, min(index, num_colors - 1))
                self.mapping[y // self.block_size, x // self.block_size] = index
                
        with io.BytesIO() as stream:
            for y in range(self.mapping.shape[0]):
                for x in range(self.mapping.shape[1]):
                    index = self.mapping[y, x]
                    character = color_set[index]
                    stream.write(character.encode('utf-8'))

                stream.write(b"\n")
            self.obj = stream.getvalue()
    
    def on_invert(self):
        if self.var.get():
            self.var.set(False)
        else:
            self.var.set(True)
        if self.obj:
            self.drawing_AA()
    
    def drawing_AA(self):
        if self.obj:
            obj_text = self.obj.decode('utf-8')
            if self.var.get():
                background_color = 'black'
                foreground_color = 'white'
            else:
                background_color = 'white'
                foreground_color = 'black'
            if self.label:
                self.label.pack_forget()
            self.label = ttk.Label(self.window, text=obj_text, font=('Courier New' , 4), background=background_color, foreground=foreground_color)
            self.label.pack(side=tk.LEFT)
        else:
            messagebox.showerror('Error' , 'Can not create AA Image')
            self.window.destroy()
            self.window = None

class DirectoryWatcher(threading.Thread):
    def __init__(self , directory , filelist):
        super().__init__()
        self.directory = directory
        self.filelist1 = filelist
        self.file_set = set()
        self.running = True

    def run(self):
        while self.running:
            files = os.listdir(self.directory)
            new_files = [file for file in files if file not in self.file_set and os.path.splitext(file)[1].lower() in ('.jpeg' , '.jpg' , '.png')]
            for file in new_files:
                self.filelist1.insert(tk.END , file)
                self.file_set.add(file)

            time.sleep(1)

    def stop(self):
        self.running = False
class PlaceHolder(tk.Entry):
    def __init__(self, master=None, placeholder="", color='grey', **kwargs):
        super().__init__(master, **kwargs)
        
        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg_color = self['fg']
        
        self.bind('<FocusIn>', self.focus_in)
        self.bind('<FocusOut>', self.focus_out)
        self.bind('<Key>', self.clear_placeholder) 
        
        if not self.get():
            self.insert(0, self.placeholder)
            self.configure(fg=self.placeholder_color)


    def put_placeholder(self):
        self.insert(0, self.placeholder)
        self.configure(foreground=self.placeholder_color)

    def remove_placeholder(self):
        self.delete(0, 'end')
        self.configure(foreground=self.default_fg_color)

    def focus_in(self, event):
        if self.get() == self.placeholder:
            self.delete('0', tk.END)
        self.configure(fg=self.default_fg_color)

    def focus_out(self, event):
        if not self.get():
            self.insert(0, self.placeholder)
            self.configure(fg=self.placeholder_color)
            
    def clear_placeholder(self, event):  # 追加
        if self['fg'] == self.placeholder_color:
            self.delete('0', 'end')
            self.configure(fg=self.default_fg_color)

class App(tk.Tk):
    def __init__(self , **kwargs):
        super().__init__(**kwargs)
        # メインウインドウ設定
        self.title('IIMAGE FILE TOOL')
        self.geometry('512x768+100+100')
        self.resizable(False , False)
        self.update()

        # セカンドパネルの作成
        self.win = tk.Toplevel()
        self.win.title('File Info')
        self.win.geometry(f'600x970+{self.winfo_x()+self.winfo_width()}+{self.winfo_y()}')
        self.win.protocol('WM_DELETE_WINDOW' , self.close_Handler)
        self.win.resizable(False , False)

        self.frame1 = ttk.Frame(self.win)
        self.frame1.grid(row=0,column=0 , pady=10)
        self.frame2 = ttk.Frame(self.win)
        self.frame2.grid(row=0 , column=1 , pady=10) 
        self.entry1 = PlaceHolder(self.frame1 , placeholder='Open Source Folder' , color='Gray' , width=25 , justify='right')
        self.entry1.config(state='disabled')
        self.dir_button1 = ttk.Button(self.frame1 , text='Open' , command=self.on_open_dir)
        self.entry1.grid(row=0 , column=0 , padx=5, pady=10 , sticky='w')
        self.dir_button1.grid(row=0 , column=1 , padx=5 , pady=10 , sticky='w')
        
        # リストボックス1の作成

        self.filelist1 = tk.Listbox(self.frame1 , width=40 , height=30 )
        self.filelist1.grid(row=1, column=0, columnspan=2 , padx=10, pady=10, sticky='ns')
        self.scroll1 = ttk.Scrollbar(self.frame1 , command=self.filelist1.yview)
        self.scroll1.grid(row=1, column=2, sticky='ns')
        self.filelist1.config(yscrollcommand=self.scroll1.set)

        self.entry2 = PlaceHolder(self.frame2 , placeholder='Open Destination Folder' , color='Gray' , width=25 , justify='right')
        self.entry2.config(state='disabled')
        self.dir_button2 = ttk.Button(self.frame2 , text='Open' , command=self.on_open_send_dir)
        self.entry2.grid(row=0 , column=0 , padx=5, pady=10 , sticky='w')
        self.dir_button2.grid(row=0 , column=1 , padx=5 , pady=10 , sticky='w')
        # リストボックス2の作成
        self.filelist2 = tk.Listbox(self.frame2 , width=40 , height=30)
        self.filelist2.grid(row=1, column=0, columnspan=2 , padx=10, pady=10, sticky='ns')
        self.scroll2 = ttk.Scrollbar(self.frame2  , command=self.filelist2.yview)
        self.scroll2.grid(row=1, column=2, sticky='ns')
        self.filelist2.config(yscrollcommand=self.scroll2.set)

        # ラベルスタイルの設定
        style = ttk.Style()
        style.configure('Bold.TLabel' , font=('Helvetica' , 12 , 'bold'))

        # 画像生成情報表示用テキストボックスの作成
        self.label1 = ttk.Label(self.win , text='Prompt' , width=32 , justify='left' , style='Bold.TLabel')
        self.label2 = ttk.Label(self.win , text='Negative Prompt' , width=32, justify='left' , style='Bold.TLabel')
        self.label3 = ttk.Label(self.win , text='Generated Info.' , width=32, justify='left' , style='Bold.TLabel')
        self.text1 = tk.Text(self.win , width=40 , height=5 , state='disabled')
        self.text2 = tk.Text(self.win , width=40 , height=5 , state='disabled')
        self.text3 = tk.Text(self.win , width=40 , height=5 , state='disabled')
        self.button1 = ttk.Button(self.win , text='Copy' , command=self.on_copy_prompt)
        self.button2 = ttk.Button(self.win , text='Copy' , command=self.on_copy_negative)
        self.label1.grid(row=1 , column=0 , padx=10 , pady=10 )
        self.label2.grid(row=3 , column=0 , padx=10 , pady=10 )
        self.label3.grid(row=5 , column=0 , padx=10 , pady=10 )
        self.text1.grid(row=2 , column=0, columnspan=2, padx=10 , pady=10, sticky='nsew')
        self.text2.grid(row=4 , column=0, columnspan=2, padx=10 , pady=10, sticky='nsew')
        self.text3.grid(row=6 , column=0, columnspan=2, padx=10 , pady=10, sticky='nsew')
        self.button1.grid(row=1 , column=1 , padx=10 , pady=10 , sticky='e')
        self.button2.grid(row=3 , column=1 , padx=10 , pady=10 , sticky='e')

        # メインメニューバーの作成
        self.menubar = tk.Menu(self)
        self.filemenu = tk.Menu(self.menubar , tearoff=0)
        self.filemenu.add_command(label="Open" , command=self.on_open_file)
        self.filemenu.add_command(label="Save" , command=self.on_save_file)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit" , command=self.quit)
        self.menubar.add_cascade(label='File' , menu=self.filemenu)

        self.processmenu = tk.Menu(self.menubar , tearoff=0)
        self.processmenu.add_command(label='Gaussian Blur', command=self.on_gaussian_blur)
        self.processmenu.add_command(label='Mosaic' , command=self.on_mosaic)
        self.processmenu.add_command(label='Pencil' , command=self.on_pencil)
        self.processmenu.add_command(label='Resize' , command=self.on_resize)
        self.processmenu.add_command(label='Color conv.' , command=self.on_change)
        self.processmenu.add_command(label='HSV conv.' , command=self.on_hsv_panel)
        self.processmenu.add_command(label='Contrast adj.' , command=self.on_contrast_panel)
        self.processmenu.add_command(label='Gamma corr.', command=self.on_gamma)
        self.processmenu.add_command(label='Gray conv.' , command=self.on_gray_scale)
        self.processmenu.add_command(label='Sepia conv.' , command=self.on_sepia)
        self.processmenu.add_command(label='Half-Tone1' , command=lambda : self.on_half(1))
        self.processmenu.add_command(label='Half-Tone2' , command=lambda : self.on_half(0))
        self.processmenu.add_command(label='Dot-Art' , command=self.on_dot)
        self.processmenu.add_command(label='Painterly style' ,  command=self.on_paint)
        self.processmenu.add_command(label='Posterization' , command=self.on_poster_panel)
        self.processmenu.add_command(label='Mirror' , command=self.on_mirror)
        self.processmenu.add_command(label='Cropping' , command=self.on_trim)
        self.menubar.add_cascade(label='Conv' , menu=self.processmenu)
        
        # 編集メニュー
        self.editmenu = tk.Menu(self.menubar , tearoff=0)
        self.editmenu.add_command(label='Paste' , command=self.on_paste)
        self.editmenu.add_command(label='Copy', command=self.copy_to_clipboard)
        self.editmenu.add_command(label='Undo' , command=self.undo) 
        self.menubar.add_cascade(label='Edit' , menu=self.editmenu)

        # アップスケール
        self.upscalemenu = tk.Menu(self.menubar , tearoff=False)
        self.upscalemenu.add_command(label='CARNv2' , command=self.on_carn_panel)
        self.upscalemenu.add_command(label='R-ESRGAN' , command=self.on_esrgan_panel)
        self.menubar.add_cascade(label='ULTRA-Resolution' , menu=self.upscalemenu)
        
        # ASCIIアート
        self.menubar.add_command(label='Create AA', command=self.on_aa_panel)
        
        # Promptジェネレータ
        self.menubar.add_command(label='Prompt Gen.' , command=self.on_gen_panel)
        self.ppt = None
        self.que = deque()
        
        # ステータスバー
        self.frame_status = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        self.frame_status.pack(side=tk.BOTTOM , fill=tk.X)
        self.status_bar = tk.Label(self.frame_status , text='' , justify=tk.RIGHT, anchor=tk.E)
        self.status_bar.pack(fill=tk.X , padx=10)

        # UPScale WINDOW
        self.carn = None
        self.esrgan = None
        self.carn_var = tk.IntVar()
        self.esrgan_var = tk.IntVar()
        
        # AA パネル
        self.aa = None
        self.aa_var = tk.IntVar()
        self.show_aa = None
        
        # posterization用パネル
        self.poster = None
        self.poster_var = tk.IntVar()
        self.poster_var.set(2)
        self.image_poster = None

        #　設定メニュー
        # self.configmenu = tk.Menu(self.menubar , tearoff=0)
        self.menubar.add_command(label='Settings' , command=self.on_config_panel)
        # self.menubar.add_cascade(label='Settings' , menu=self.configmenu)
        self.config(menu=self.menubar)

        # パネル2にメニュー追加
        self.filemenubar = tk.Menu(self.win)
        self.filemenu2 = tk.Menu(self.filemenubar , tearoff=False)
        self.filemenu2.add_command(label='Open in Explorer(Left)' , command=lambda : self.on_open_explorer(1))
        self.filemenu2.add_command(label='Open in Explorer(Right)' , command=lambda : self.on_open_explorer(2))
        self.filemenu2.add_command(label='Rename(Left)' , command= lambda : self.on_rename(1))
        self.filemenu2.add_command(label='Rename(Right)' , command= lambda : self.on_rename(2))
        self.filemenubar.add_cascade(label='Command' , menu=self.filemenu2)
        self.filemenubar.add_command(label='Archive' , command=self.on_archive)
        self.filemenubar.add_command(label='By Name' , command=self.on_sort_by_name)
        self.filemenubar.add_command(label='By Time' , command=self.on_sort_by_time)
        self.sendmenu = tk.Menu(self.filemenubar , tearoff=False)
        self.win.config(menu=self.filemenubar)

        # Stable Diffusion情報
        self.target_text1 = ''
        self.target_text2 = ''
        self.target_text3 = ''

        # トリミングのためのCanvasの作成
        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 汎用変数
        self.image_path = ''
        self.directory = None # Sourse Directory
        self.senddir = None # Destination Directory
        self.prev_dest_dir = None
        self.prev_src_dir = None
 
        # ImageTk オブジェクト 
        self.image = None # PhotoImage
        self.original = None # Undo用
        self.popup = None # ポップアップウインドウ
        self.width = None # self.imageの幅
        self.height = None # self.imageの高さ

        # GaussianBlur関数用変数
        self.gaus = None
        self.image_arr = None # CV2 image
        self.image_tmp = None
        self.kernel_value = None
        self.sigma_value = None

        # クリックポップアップウィンドウ
        self.bind('<Button-1>' , self.on_resize_opt) # リサイズ
        self.bind('<Button-3>' , self.show_menu) # 編集メニュー
        

        # クリッピング、モザイク用
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

        x = 0
        # パネル2のコールバック関数の設定
        self.filelist1.bind('<Double-Button-1>' , self.on_draw)
        self.filelist2.bind('<Double-Button-1>' , self.on_draw)
        self.filelist1.bind('<Button-3>' , self.popup_menu)
        self.filelist2.bind('<Button-3>' , self.popup_menu)
        self.filelist1.bind('<Down>' , self.on_down)
        self.filelist2.bind('<Down>' , self.on_down)
        self.filelist1.bind('<Up>' , self.on_up)
        self.filelist2.bind('<Up>' , self.on_up)
        self.filelist1.bind('<Right>' , self.on_send)
        self.filelist1.bind('<Delete>' , lambda x: self.on_delete(1))
        self.filelist2.bind('<Delete>' , lambda x: self.on_delete(2))

        # サブスレッドのためのシグナルオブジェクト
        self.watcher = None
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # ガンマ補正のサブウインドウ
        self.gamma = None

        # モザイクパネル
        self.mosaic_window = None

        # HSV パネル
        self.hsv = None
        
        # コントラストパネル
        self.cont = None
        
        # 保険の変数
        self.image_cont = None
        self.image_half = None
        self.image_tmp = None

        # 設定画面
        self.config_panel = None
        self.chk_var1 = tk.BooleanVar()
        self.chk_var2 = tk.BooleanVar()
        self.chk_var3 = tk.BooleanVar()
        self.chk_var1.set(True)
        self.chk_var2.set(True)
        self.chk_var3.set(True)

        # 送り元、送り先ディレクトリ情報
        self.source_dir = ''
        self.dest_dir = ''

        # # ディレクトリ情報ファイル問い合わせ
        dirfile_dir = r'./dir_config.ini'
        if os.path.isfile(dirfile_dir):
            with open(dirfile_dir , 'r' , encoding='utf-8') as f:
                lines = f.read().splitlines()
                if len(lines) == 2:
                    self.source_dir = self.directory = lines[0]
                    self.dest_dir = self.senddir = lines[1]
                    self.entry1.config(state='normal')
                    self.entry1.delete('0' , tk.END)
                    self.entry1.insert(tk.END,self.source_dir)
                    self.entry1.xview_moveto(1)
                    self.entry1.config(state='disabled')
                    self.entry2.config(state='normal')
                    self.entry2.delete('0' , tk.END)
                    self.entry2.insert(tk.END,self.dest_dir)
                    self.entry2.xview_moveto(1)
                    self.entry2.config(state='disabled')
                    self.dir_watcher(event=None)
        else:
            with open(dirfile_dir , 'w') as f:
                pass

        if self.source_dir or self.directory:
            file_list = os.listdir(self.source_dir) 
            for filename in file_list:
                if os.path.splitext(filename)[1].lower() not in ('.jpeg' , '.jpg' , '.png'):
                    continue
                self.filelist1.insert(tk.END , filename)

        if self.dest_dir or self.senddir:
            file_list = os.listdir(self.dest_dir)
            for filename in file_list:
                if os.path.splitext(filename)[1].lower() not in ('.jpeg' , '.jpg' , '.png'):
                    continue
                self.filelist2.insert(tk.END , filename)
                
    def on_sort_by_name(self):
        items = list(self.filelist2.get(0 , tk.END))
        items.sort()
        self.filelist2.delete(0 , tk.END)
        for item in items:
            self.filelist2.insert(tk.END , item)
    
    def on_sort_by_time(self):
        files = os.listdir(self.dest_dir)
        sorted_file = sort_files_by_timestamp(files , self.dest_dir)
        self.filelist2.delete(0 , tk.END)
        for file in sorted_file:
            self.filelist2.insert(tk.END , file)
                
    def on_poster_panel(self , event=None):
        if not self.poster or (not self.poster.winfo_exists()):
            self.poster = tk.Toplevel()
            self.poster.title('Posterization')
            self.poster.geometry('350x100')
            self.poster.protocol('WM_DELETE_WINDOW' , self.on_poster_destroy)
            self.poster.withdraw()
            self.poster.update()
            self.poster.geometry(f'+{self.winfo_x()+20}+{self.winfo_y()+20}')
            self.poster.deiconify()
            self.poster_scale = ttk.Scale(self.poster,
                                          from_=2,
                                          to=4,
                                          length=200,
                                          orient=tk.HORIZONTAL,
                                          variable=self.poster_var,
                                          command=self.on_update_poster)
            self.poster_label = ttk.Label(self.poster , text='')
            self.poster_scale.grid(row=0 , column=0 , padx=10 , pady=10)
            self.poster_label.grid(row=0 , column=1 , padx=10 , pady=10)
            self.on_update_poster()
            self.poster.bind("<MouseWheel>" , self.on_wheel_poster)
            
    def on_poster_destroy(self):
        self.image = self.image_poster
        self.image_poster = None
        self.poster.destroy()
        self.poster = None
        self.poster_var.set(2)

    def on_wheel_poster(self , event=None):
        if event.delta > 0 and self.poster_scale.get() < self.poster_scale.cget('to'):
            self.poster_scale.set(self.poster_scale.get() + 1)
        elif event.delta < 0 and self.poster_scale.get() > self.poster_scale.cget('from'):
            self.poster_scale.set(self.poster_scale.get() - 1)
        
    def on_update_poster(self , event=None):
        if self.image:
            self.poster_label.config(text=f'quantize:{self.poster_var.get()}')
            sz = 256 / self.poster_var.get()
            buf = sz / 2
            self.image_arr = np.array(ImageTk.getimage(self.image))
            self.image_arr = np.uint8(self.image_arr / sz)
            self.image_arr = np.uint8(self.image_arr * sz + buf)
            self.image_poster = ImageTk.PhotoImage(Image.fromarray(self.image_arr))
            self.canvas.delete('image')
            self.canvas.create_image(0,0,image=self.image_poster , anchor=tk.NW, tag="image")
        else:
            messagebox.showerror('Error' , 'Display Image')
                
    def on_paint(self , event=None):
        if self.image:
            pil_img = ImageTk.getimage(self.image)
            img_w , img_h = self.image.width() , self.image.height()
            out_image = Image.new('RGB' , (img_w , img_h))
            draw = ImageDraw.Draw(out_image)
            for i in range(img_w*img_h):
                x = random.randint(0 , img_w-1)
                y = random.randint(0 , img_h-1)
                rgb = pil_img.getpixel((x , y))
                r = random.randint(1 , 8)
                draw.ellipse((x , y , x+r , y+r) , fill=rgb , outline=None)
            self.image = ImageTk.PhotoImage(out_image)
            self.canvas.create_image(0,0,image=self.image , anchor=tk.NW, tag="image")
        else:
            messagebox.showerror('Error' , 'Display Image')
    
                
    def on_dot(self , event=None):
        if self.image:
            tensor = tk_to_tensor(self.image)
            sz = 10
            height , width = tensor.shape[1:]
            dst_img = Image.new('RGB' , (width , height))
            draw = ImageDraw.Draw(dst_img)
            for y in range( 0 , height , sz):
                for x in range(0 , width , sz):
                    rgb = tensor[: , y:y+sz , x:x+sz]
                    r = torch.mean(rgb[0,:,:].float()).to(torch.uint8)
                    g = torch.mean(rgb[1,:,:].float()).to(torch.uint8)
                    b = torch.mean(rgb[2,:,:].float()).to(torch.uint8)
                    draw.ellipse((x+1 , y+1 , x+sz-1 , y+sz-1) , fill=(r , g , b) , outline=None)
            self.image = ImageTk.PhotoImage(dst_img)
            self.canvas.create_image(0,0, image=self.image , anchor=tk.NW, tag="image")
        else:
            messagebox.showerror('Error' , 'Display Image')
                    
                
    def on_gen_panel(self , event=None):
        if not self.ppt or ( not self.ppt.winfo_exists()):
            self.ppt = tk.Toplevel()
            self.ppt.title('Prompt Generator')
            self.ppt.geometry('500x830')
            self.ppt.withdraw()
            self.ppt.update()
            self.ppt.geometry('+%d+%d' % (self.winfo_rootx() + 40, self.winfo_rooty() + 40))
            self.ppt.deiconify()
            self.ppt.protocol('WM_DELETE_WINDOW' , self.on_ppt_destroy)
            self.ppt_list = tk.Listbox(self.ppt , width=60 , height=30)
            self.ppt_scroll = ttk.Scrollbar(self.ppt  , command=self.ppt_list.yview)
            self.ppt_scroll.grid(row=0, column=4, sticky='ns')
            self.ppt_list.config(yscrollcommand=self.ppt_scroll.set)
            
            self.ppt_enh1 = tk.IntVar()
            self.ppt_enh1.set(0)
            self.ppt_r1 = ttk.Radiobutton(self.ppt , text='None' , variable=self.ppt_enh1 , value=0)
            self.ppt_r2 = ttk.Radiobutton(self.ppt , text='x1' , variable=self.ppt_enh1 , value=1)
            self.ppt_r3 = ttk.Radiobutton(self.ppt , text='x2' , variable=self.ppt_enh1 , value=2)
            self.ppt_r4 = ttk.Radiobutton(self.ppt , text='x3' , variable=self.ppt_enh1 , value=3)
            
            self.ppt_enh2 = tk.DoubleVar()
            self.ppt_enh2.set(1.0)
            self.ppt_r5 = ttk.Radiobutton(self.ppt , text='0.8' , variable=self.ppt_enh2 , value=0.8)
            self.ppt_r6 = ttk.Radiobutton(self.ppt , text='0.9', variable=self.ppt_enh2 , value=0.9)
            self.ppt_r7 = ttk.Radiobutton(self.ppt , text='1.0', variable=self.ppt_enh2 , value=1.0)
            self.ppt_r8 = ttk.Radiobutton(self.ppt , text='1.1', variable=self.ppt_enh2 , value=1.1)
            self.ppt_r9 = ttk.Radiobutton(self.ppt , text='1.2', variable=self.ppt_enh2 , value=1.2)
            self.ppt_r10 = ttk.Radiobutton(self.ppt , text='1.3', variable=self.ppt_enh2 , value=1.3)
            self.ppt_r11 = ttk.Radiobutton(self.ppt , text='1.4', variable=self.ppt_enh2 , value=1.4)
            self.ppt_r12 = ttk.Radiobutton(self.ppt , text='1.5', variable=self.ppt_enh2 , value=1.5)
            
            self.ppt_copy = ttk.Button(self.ppt , text='Copy' , command=self.on_to_clipboard)
            self.ppt_undo = ttk.Button(self.ppt , text='Undo' , command=self.on_undo)
            ppt_font = Font(family='Helvetica' , size=10)
            self.ppt_text = tk.Text(self.ppt , width=50 , height=10 , font=ppt_font)
            self.ppt_sep = ttk.Separator(self.ppt)
            
            self.ppt_list.grid(row=0 , column=0 , columnspan=4 , padx=10 , pady=10 , sticky=tk.EW)
            
            self.ppt_r1.grid(row=1 , column=0 , padx=10 , pady=10)
            self.ppt_r2.grid(row=1 , column=1 , padx=10 , pady=10)
            self.ppt_r3.grid(row=1 , column=2 , padx=10 , pady=10)
            self.ppt_r4.grid(row=1 , column=3 , padx=10 , pady=10)
            
            self.ppt_sep.grid(row=2 , column=0 , columnspan=4 , sticky=tk.EW)
            
            self.ppt_r5.grid(row=3 , column=0 , padx=10 , pady=10)
            self.ppt_r6.grid(row=3 , column=1 , padx=10 , pady=10)
            self.ppt_r7.grid(row=3 , column=2 , padx=10 , pady=10)
            self.ppt_r8.grid(row=3 , column=3 , padx=10 , pady=10)
            self.ppt_r9.grid(row=4 , column=0 , padx=10 , pady=10)
            self.ppt_r10.grid(row=4 , column=1 , padx=10 , pady=10)
            self.ppt_r11.grid(row=4 , column=2 , padx=10 , pady=10)
            self.ppt_r12.grid(row=4 , column=3 , padx=10 , pady=10)
            self.ppt_text.grid(row=5 , column=0 , columnspan=3 , rowspan=2 ,  padx=10 , pady=10)
            self.ppt_copy.grid(row=5 , column=3 , padx=10 , pady=10)
            self.ppt_undo.grid(row=6 , column=3 , padx=10 , pady=10)
            
            self.ppt_list.bind('<Double-Button-1>' , self.on_put_prompt)
            
            with open('prompt.txt' , 'r' , encoding='utf-8') as f:
                prompts = f.readlines()
            lst = []
            for i in prompts:
                s = i.replace('\n' , '')
                lst.append(s)
                self.ppt_list.insert(tk.END , s)
                
    def on_ppt_destroy(self):
        self.ppt_enh1.set(0)
        self.ppt_enh2.set(1.0)
        self.que.clear()
        self.ppt.destroy()
        self.ppt = None
                
    def on_undo(self , enent=None):
        if self.que:
            if len(self.que) > 1:
                self.que.pop()
                put_data = self.que[-1]
                self.ppt_text.delete('1.0' , 'end-1c')
                self.ppt_text.insert(tk.END , put_data)
            else:
                self.ppt_text.delete('1.0' , 'end-1c')
                self.que.clear()
                
    def on_put_prompt(self , event=None):
        enh = self.ppt_enh1.get()
        sth = self.ppt_enh2.get()
        if sth != 1.0:
            index = self.ppt_list.curselection()[0]
            put_data = self.ppt_list.get(index)
            put_data += ':' + str(sth)
        else:
            index = self.ppt_list.curselection()[0]
            put_data = self.ppt_list.get(index)
        if enh == 1:
            put_data = '(' + put_data + ')'
        elif enh == 2:
            put_data = '((' + put_data + '))'
        elif enh == 3:
            put_data = '(((' + put_data + ')))'
        if self.que:
            put_data = ', ' + put_data 
        self.ppt_text.insert(tk.END , put_data)
        history = self.ppt_text.get('1.0' , 'end-1c')
        self.que.append(history) 
            
    def on_to_clipboard(self , event=None):
        text = self.ppt_text.get('1.0' , 'end-1c')
        self.clipboard_clear()
        self.clipboard_append(text)
                
    def on_archive(self , event=None):
        total_file_size = 0
        archive_file_size = 0
        result = messagebox.askyesno('Confirm' , 'Compress the image files in the left listbox to ZIP. Are you sure?')
        if result:
            file_list = [f.lower() for f in  os.listdir(self.directory) if f.lower().endswith(('.jpg' , '.png'))]
            output_path = os.path.join(self.directory, os.path.basename(self.directory) + '.zip')
            print(output_path)
            with zipfile.ZipFile(output_path, mode='w') as zfile:
                for file in file_list:
                    archived_file = (os.path.join(self.directory, file))
                    zfile.write(archived_file, arcname=file)
            archive_file_size = os.path.getsize(output_path)
            archive_file_size /= 1024**2
            delete_ = messagebox.askyesno('Confirm' , f'Compression successful. Compressed to {archive_file_size:.2f} MB. Do you want to delete the original data?')
            if delete_:
                for file in file_list:
                    os.remove(self.directory + '/' + file)
            else:
                return
        else:
            return
                    
    def update_status(self):
        self.width = self.image.width()
        self.height = self.image.height()
        self.status_bar.config(text=f'({self.width}x{self.height})')

    def set_AI_info(self , image_path):
        with open(image_path , 'rb') as f:
            data = f.read()
        # テキストチャンクの先頭インデックスを取得
        index = data.find(b'\x74\x45\x58\x74')
        if index == -1:
            index = data.find(b'\x69\x54\x58\x74')
        logger.debug('index:%s',index)
        # テキスト長を取得
        length = int.from_bytes(data[index-4:index], byteorder='big')
        logger.debug('length:%s' , length)
        # テキストデータを取得
        text_data = data[index+4:index+4+length]
        # keyword(parameters)を読み飛ばしテキストを取得する
        if b'\x00\x00\x00\x00\x00' in text_data:
            txt = text_data.split(b'\x00\x00\x00\x00\x00')
        else:
            txt = text_data.split(b'\x00')
        text = txt[1].decode('shift-jis' , errors='ignore')
        text = re.sub(r'[\x01-\x09\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
        text = re.sub(r'[^\x00-\x7f]+' , ' ' , text)
        # text = text.replace('?' , ' ')
        
        # Prompt
        pattern1 = re.compile('^(.+)\n.*Negative.+$' , re.DOTALL)
        self.target_text1 = re.sub(pattern1 , r'\1' , text)
        # Negative Prompt
        pattern2 = re.compile('(?<=prompt:\s)(.+)(?=Steps:)' , re.S)
        self.target_text2 = '\n'.join(re.findall(pattern2, text)).strip()
        # その他の情報
        pattern3 = re.compile('^.+(.{1}?Steps.+)$' , re.DOTALL)
        self.target_text3 = re.sub(pattern3 , r'\1' , text)

        # 各ウィジェットに情報の挿入
        self.text1.config(state='normal')
        self.text1.delete('1.0' , tk.END)
        self.text1.insert(tk.END , self.target_text1)
        self.text1.config(state='disabled')

        self.text2.config(state='normal')
        self.text2.delete('1.0' , tk.END)
        self.text2.insert(tk.END , self.target_text2)
        self.text2.config(state='disabled')

        self.text3.config(state='normal')
        self.text3.delete('1.0' , tk.END)
        self.text3.insert(tk.END , self.target_text3)
        self.text3.config(state='disabled')

    def on_carn_panel(self):
        if not self.carn or not self.carn.winfo_exists():
            self.carn = tk.Toplevel()
            self.carn.title('CARN Upscaler')
            self.carn.geometry(f'+{self.winfo_x()+10}+{self.winfo_y()+10}')
            self.carn_var.set(20)
            self.carn_scale = ttk.Scale(self.carn , 
                                        from_=11 , 
                                        to=20 , 
                                        length=200 , 
                                        variable=self.carn_var , 
                                        command=self.update_carn_scale)
            self.carn_label = ttk.Label(self.carn , text='')
            self.carn_button = ttk.Button(self.carn , text='Exec' , command=self.on_exec_carn)
            self.update_carn_scale(self.carn_var)
            self.carn_scale.grid(row=0 , column=0 , padx=10 , pady=10)
            self.carn_label.grid(row=0 , column=1 , padx=10 , pady=10)
            self.carn_button.grid(row=1 , column=1 , padx=10 , pady=10 , sticky='e')
    
    def update_carn_scale(self , value=None):
        value = self.carn_var.get()/10
        self.carn_label.config(text=f'{value:.1f}x')

    def on_exec_carn(self):
        mag = self.carn_var.get()/10
        model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=torch.nn.Conv2d,
                                single_conv_size=3, single_conv_group=1,
                                scale=2, activation=torch.nn.LeakyReLU(0.1),
                                SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))
        model_cran_v2 = network_to_half(model_cran_v2)
        checkpoint = r'./CARN_model.pt'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_cran_v2.load_state_dict(torch.load(checkpoint , map_location=device))
        if device.type == 'cpu':
            model_cran_v2 = model_cran_v2.float()
        if device.type == 'cuda':
            model_cran_v2 = model_cran_v2.to(device)
        if self.image:
            img = ImageTk.getimage(self.image)
            img = img.convert('RGB')
            img_upscale = img.resize((img.size[0], img.size[1]), resample=Resampling.BICUBIC)
            img_t = to_tensor(img_upscale).unsqueeze(0)
            if device.type == 'cuda':
                img_t = img_t.to(device)
            with torch.no_grad():
                out = model_cran_v2(img_t.to(device).float())
            if device.type == 'cuda':
                out = out.cpu()
            out_np = out.float().squeeze(0).cpu().numpy()
            out_np = np.clip(out_np, 0, 1)
            out_np = out_np.transpose((1, 2, 0))
            out_np = cv2.resize(out_np, None, fx=mag/2.0, fy=mag/2.0, interpolation=cv2.INTER_LANCZOS4)
            out_np = (out_np * 255).astype(np.uint8)
            h, w, _ = out_np.shape
            self.image = ImageTk.PhotoImage(Image.fromarray(out_np))
            self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW, tag="image")
            self.wm_geometry(f'{w}x{h}')
            self.update_status()
            self.carn.destroy()
            self.carn = None
        else:
            messagebox.showerror('Error' , 'Display Image')
            self.carn.destroy()
            self.carn = None
    
    def on_esrgan_panel(self):
        if not self.esrgan or not self.esrgan.winfo_exists():
            self.esrgan = tk.Toplevel()
            self.esrgan.title('R-ESRGAN Upscaler')
            self.esrgan.geometry(f'+{self.winfo_x()+10}+{self.winfo_y()+10}')
            self.esrgan_var.set(20)
            self.esrgan_scale = ttk.Scale(self.esrgan , 
                                        from_=11 , 
                                        to=40 , 
                                        length=200 , 
                                        variable=self.esrgan_var , 
                                        command=self.update_esrgan_scale)
            self.esrgan_label = ttk.Label(self.esrgan, text='')
            self.esrgan_radio_var = tk.BooleanVar()
            self.esrgan_radio_var.set(False)
            self.esrgan_label2 = ttk.Label(self.esrgan , text='Which R-ESRGAN Use?')
            self.esrgan_radio1 = ttk.Radiobutton(self.esrgan , text='x4' , variable=self.esrgan_radio_var , value=True)
            self.esrgan_radio2 = ttk.Radiobutton(self.esrgan , text='x2' , variable=self.esrgan_radio_var , value=False)
            self.esrgan_button = ttk.Button(self.esrgan , text='Exec' , command=self.on_exec_esrgan)
            self.update_esrgan_scale()
            self.esrgan_scale.grid(row=0 , column=0 , padx=10 , pady=10)
            self.esrgan_label.grid(row=0 , column=1 , padx=10 , pady=10)
            self.esrgan_label2.grid(row=1 , column=0 , padx=10 , pady=10)
            self.esrgan_button.grid(row=3 , column=1 , padx=10 , pady=10 , sticky='e')
            self.esrgan_radio1.grid(row=2 , column=0 , padx=10 , pady=10 , sticky='w')
            self.esrgan_radio2.grid(row=3 , column=0 , padx=10 , pady=10 , sticky='w')

    def update_esrgan_scale(self , value=None):
        value = self.esrgan_var.get()/10
        self.esrgan_label.config(text=f'{value:.1f}x')

    def on_exec_esrgan(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_filex2 = r'./weights/RealESRGAN_x2.pth'
        model_filex4 = r'./weights/RealESRGAN_x4.pth'
        download1 = True
        if self.esrgan_radio_var:
            if os.path.exists(model_filex2):
                download = False
            model = RealESRGAN(device , scale=2)
            model.load_weights(model_filex2, download=download1)
        else:
            if os.path.exists(model_filex4):
                download = False
            model = RealESRGAN(device , scale=4)
            model.load_weights(model_filex4, download=download)
        if self.image:
            img = ImageTk.getimage(self.image)
            img = img.convert('RGB')
            out_img = model.predict(img)
            mag = self.esrgan_var.get()/10
            self.image_arr = np.array(out_img)
            if self.esrgan_radio_var: # x2
                self.image_arr = cv2.resize(self.image_arr , dsize=None , fx=mag/2.0 , fy=mag/2.0 , interpolation=cv2.INTER_LANCZOS4 )
            else: # x4
                self.image_arr = cv2.resize(self.image_arr , dsize=None , fx=mag/4.0 , fy=mag/4.0 , interpolation=cv2.INTER_LANCZOS4 )
            w , h , _ = self.image_arr.shape
            self.image = ImageTk.PhotoImage(Image.fromarray(self.image_arr))
            self.canvas.create_image(0,0,image=self.image , anchor=tk.NW, tag="image")
            self.wm_geometry(f'{self.image.width()}x{self.image.height()}')
            self.update_status()
            self.esrgan.destroy()
            self.esrgan = None
        else:
            messagebox.showerror('Error' , 'Display Image')
            self.esrgan = None
            
    def on_half(self , pattern):
        if self.image:
            self.image_arr = bayers(self.image , pattern)
            self.image = ImageTk.PhotoImage(Image.fromarray(self.image_arr))
            self.canvas.create_image(0,0,image=self.image, anchor=tk.NW, tag="image")
        else:
            messagebox.showerror('Error' , 'Display Image')
            
    def on_aa_panel(self):
        if self.image:
            if not self.aa or ( not self.aa.winfo_exists()):
                self.aa = tk.Toplevel()
                self.aa.title('ASCII ART Config.')
                self.aa.withdraw()
                self.aa.update()
                self.aa.geometry('+%d+%d' % (self.winfo_rootx() - 40, self.winfo_rooty() - 40))
                self.aa.deiconify()
                
                # ウィジェットの作成と配置
                self.entry_aa = ttk.Entry(self.aa , width=5 , justify=tk.RIGHT)
                self.entry_aa.insert(tk.END , '140')
                self.button_aa = ttk.Button(self.aa , text='Exec.' , command=self.on_exec_aa)
                self.label_aa = ttk.Label(self.aa , text='Width:(40~200)')
                self.label_aa.grid(row=0 , column=0 , padx=10 , pady=10)
                self.entry_aa.grid(row=1 , column=0 , padx=10 , pady=10)
                self.button_aa.grid(row=1, column=1 , padx=10 , pady=10)
        else:
            messagebox.showerror('Error' , 'Display Image.')
            
    def on_exec_aa(self):
        try:
            width = int(self.entry_aa.get())
        except TypeError:
            messagebox.showerror('Error' , 'Input Integer')
            self.aa.destroy()
            self.aa = None
            return
        else:
            if width < 40 or width > 200:
                messagebox.showerror('Error' , 'Please input a value between 40 and 200 for the width.')
                self.aa.destroy()
                self.aa = None 
                return
            aa = AAEngine(width , self.image)
            aa.create()
            aa.drawing_AA()
            self.aa.destroy()
            self.aa = None
            
    def on_contrast_panel(self):
        if self.image:
            if not self.cont or ( not self.cont.winfo_exists()):
                self.cont = tk.Toplevel()
                self.cont.title('Contrast adjustment')
                self.cont.withdraw()
                self.cont.update()
                self.cont.geometry('+%d+%d' % (self.winfo_rootx() + 10, self.winfo_rooty() + 10))
                self.cont.deiconify()
                
                # ウィジェットの作成
                self.cont_label = ttk.Label(self.cont , text='' , width=20)
                self.cont_var = tk.IntVar()
                self.cont_var.set(110)
                style = ttk.Style()
                style.configure("Custom.Horizontal.TScale", troughcolor="white", sliderlength=30, borderwidth=0)
                self.cont_scale = ttk.Scale(self.cont,
                                            from_=40 ,
                                            to=160,
                                            length=200,
                                            style='Custom.Horizontal.TScale',
                                            orient=tk.HORIZONTAL,
                                            variable=self.cont_var,
                                            command=self.on_exec_cont)
                self.cont_scale.pack(side=tk.LEFT)
                self.cont_label.pack(side=tk.RIGHT)
                self.cont.bind("<MouseWheel>" , self.on_wheel_cont)
                self.cont.bind('<FocusOut>' , self.on_focus_out_cont)
        else:
            messagebox.showerror('Error' , 'Display Image.')
            
    def on_wheel_cont(self , event=None):
        if event.delta > 0 and self.cont_scale.get() < self.cont_scale.cget('to'):
            self.cont_scale.set(self.cont_scale.get() + 1)
        elif event.delta < 0 and self.cont_scale.get() > self.cont_scale.cget('from'):
            self.cont_scale.set(self.cont_scale.get() - 1)
            
    def on_focus_out_cont(self , enent=None):
        if self.cont:
            self.cont.destroy()
            self.cont = None
                
    def on_exec_cont(self , value=None):
        
        alpha = self.cont_var.get()/100
        self.cont_label.config(text=f'value:{alpha:.2f}')
        
        self.image_arr = np.array(ImageTk.getimage(self.image))
        self.image_arr = adjust_contrast(self.image_arr , alpha , beta=0.0)
        
        self.image_cont = ImageTk.PhotoImage(Image.fromarray(self.image_arr))
        self.canvas.create_image(0,0,image=self.image_cont , anchor=tk.NW, tag="image")
                        
    def on_hsv_panel(self):
        if self.image:
            if not self.hsv or (not self.hsv.winfo_exists()):
                self.hsv = tk.Toplevel()
                self.hsv.title('HSV color space')
                self.hsv.withdraw()  
                self.hsv.update_idletasks()
                self.hsv.geometry('+%d+%d' % (self.winfo_rootx() + 10, self.winfo_rooty() + 10))
                self.hsv.deiconify() 
                self.image_arr = np.array(ImageTk.getimage(self.image))

                self.hsv_var1 = tk.IntVar()
                self.hsv_var2 = tk.IntVar()
                self.hsv_var3 = tk.IntVar()
                self.hsv_var1.set(0)
                self.hsv_var2.set(100)
                self.hsv_var3.set(0)
                self.hsv_frame = tk.Frame(self.hsv)
                self.hsv_frame.pack()
                style = ttk.Style()
                style.configure("Custom.Horizontal.TScale", troughcolor="white", sliderlength=30, borderwidth=0)
                self.hsv_scale1 = ttk.Scale(self.hsv_frame , 
                                            from_ = 0 , 
                                            to=360 , 
                                            length=200 , 
                                            style="Custom.Horizontal.TScale",
                                            variable=self.hsv_var1 , 
                                            command=self.update_hsv)
                self.hsv_scale2 = ttk.Scale(self.hsv_frame , 
                                            from_ =0 , 
                                            to=100 , 
                                            length=200 , 
                                            style="Custom.Horizontal.TScale",
                                            variable=self.hsv_var2 , 
                                            command=self.update_hsv)
                self.hsv_scale3 = ttk.Scale(self.hsv_frame , 
                                            from_ =-255 , 
                                            to=255 , 
                                            length=200 , 
                                            style="Custom.Horizontal.TScale",
                                            variable=self.hsv_var3 , 
                                            command=self.update_hsv) 
                self.hsv_label1 = ttk.Label(self.hsv_frame , text='')
                self.hsv_label2 = ttk.Label(self.hsv_frame , text='')
                self.hsv_label3 = ttk.Label(self.hsv_frame , text='')

                self.hsv_scale1.grid(row=0 , column=0 , padx=10 , pady=10)
                self.hsv_scale2.grid(row=1 , column=0 , padx=10 , pady=10)
                self.hsv_scale3.grid(row=2 , column=0 , padx=10 , pady=10)
                self.hsv_label1.grid(row=0 , column=1 , padx=10 , pady=10)
                self.hsv_label2.grid(row=1 , column=1 , padx=10 , pady=10)
                self.hsv_label3.grid(row=2 , column=1 , padx=10 , pady=10)

                self.hsv_scale1.bind("<MouseWheel>" , self.on_wheel_1)
                self.hsv_scale2.bind("<MouseWheel>" , self.on_wheel_2)
                self.hsv_scale3.bind("<MouseWheel>" , self.on_wheel_3)

                self.hsv.bind('<FocusOut>' , self.focus_out)            
                self.update_hsv(self.hsv_var1)
                self.update_hsv(self.hsv_var2)
                self.update_hsv(self.hsv_var3)
                self.hsv.attributes('-topmost' , True)
                
        else:
            messagebox.showerror('Error' , 'Display Image')
            self.hsv.destroy()
            self.hsv = None

    def focus_out(self , enent=None):
        if self.hsv:
            self.hsv.destroy()
            self.hsv = None

    def on_wheel_1(self , event=None):
        if event.delta > 0 and self.hsv_scale1.get() < self.hsv_scale1.cget('to'):
            self.hsv_scale1.set(self.hsv_scale1.get() + 1)
        elif event.delta < 0 and self.hsv_scale1.get() > self.hsv_scale1.cget('from'):
            self.hsv_scale1.set(self.hsv_scale1.get() - 1)

    def on_wheel_2(self , event=None):
        if event.delta > 0 and self.hsv_scale2.get() < self.hsv_scale2.cget('to'):
            self.hsv_scale2.set(self.hsv_scale2.get() + 1)
        elif event.delta < 0 and self.hsv_scale2.get() > self.hsv_scale2.cget('from'):
            self.hsv_scale2.set(self.hsv_scale2.get() - 1)

    def on_wheel_3(self , event=None):
        if event.delta > 0 and self.hsv_scale3.get() < self.hsv_scale3.cget('to'):
            self.hsv_scale3.set(self.hsv_scale3.get() + 1)
        elif event.delta < 0 and self.hsv_scale3.get() > self.hsv_scale3.cget('from'):
            self.hsv_scale3.set(self.hsv_scale3.get() - 1)


    def change_hsv(self, bgr_img, shift_h=0, scale_s=1.0, shift_v=0):
        # BGR画像からHSV画像に変換
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV_FULL)

        # Hue成分（色相）を加算して、値域を[0, 360)に収める
        hue_shifted = (hsv[..., 0].astype(np.int32) + shift_h) % 360
        hsv[..., 0] = np.clip(hue_shifted, 0, 359)

        # 彩度と明度を乗算・加算して、値域を[0, 255]に収める
        hsvf = hsv.astype(np.float32)
        hsvf[..., 1] = np.clip(hsvf[..., 1] * scale_s + shift_v, 0, 255)
        hsvf[..., 2] = np.clip(hsvf[..., 2] + shift_v, 0, 255)

        # HSV画像からBGR画像に変換
        return cv2.cvtColor(hsvf.astype(np.uint8), cv2.COLOR_HSV2BGR_FULL)

    def update_hsv(self , value=None):
        h_val = self.hsv_var1.get()
        s_val = self.hsv_var2.get()/100
        v_val = self.hsv_var3.get()
        self.hsv_label1.config(text=f'Hue:{h_val}')
        self.hsv_label2.config(text=f'Sat.:{s_val}')
        self.hsv_label3.config(text=f'Val.:{v_val}')
        rgb = self.change_hsv(self.image_arr , h_val , s_val , v_val)
        rgb = np.clip(rgb , 0 , 255)
        self.image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(0,0,image=self.image,anchor=tk.NW, tag="image")

    def popup_menu(self , event):
        '''
        パネル2の右クリックポップアップウインドウ
        '''
        self.filemenu2.post(event.x_root , event.y_root)

    def on_open_explorer(self , value):
        switch = True
        if value ==  2:
            switch = False
        if switch:
            if self.directory:
                logger.debug('on_open_explorer: %s' , self.directory)
                if len(self.filelist1.curselection()) == 0:
                    subprocess.run(['start' , self.directory] , shell=True)
                else:
                    index = self.filelist1.curselection()[0]
                    file_name = self.filelist1.get(index)
                    full_path = os.path.join(self.directory , file_name)
                    subprocess.Popen(r'explorer /select,"{}"'.format(os.path.normpath(full_path)))
            else:
                messagebox.showerror('Error' , 'Open Image Folder')
        else:
            if self.senddir:
                if len(self.filelist2.curselection()) == 0:
                    subprocess.run(['start' , self.senddir] , shell=True)
                else:
                    index = self.filelist2.curselection()[0]
                    file_name = self.filelist2.get(index)
                    full_path = os.path.join(self.senddir , file_name)
                    subprocess.Popen(r'explorer /select,"{}"'.format(os.path.normpath(full_path)))
            else:
                messagebox.showerror('Error' , 'Open Image Folder')

    def on_open_send_dir(self , event=None):
        self.prev_dest_dir = self.senddir
        self.senddir = filedialog.askdirectory()
        if self.senddir:
            file_list = os.listdir(self.senddir)
            self.filelist2.delete(0 , tk.END)
            for filename in file_list:
                if os.path.splitext(filename)[1].lower() not in ('.jpeg' , '.jpg' , '.png'):
                    continue
                self.filelist2.insert(tk.END , filename)
            self.dest_dir = self.senddir
            if not self.dest_dir or self.senddir != self.prev_dest_dir:
                self.entry2.config(state='normal')
                self.entry2.delete('0' , tk.END)
                self.entry2.insert(tk.END , self.dest_dir)
                self.entry2.xview_moveto(1) 
                self.entry2.config(state='disabled')
                self.prev_dest_dir = self.dest_dir
            if self.source_dir:
                written_data = [self.source_dir + '\n', self.dest_dir + '\n']
                with open(r'./dir_config.ini' , 'w' , encoding='utf-8') as f: 
                    f.writelines(written_data)
        else:
            self.senddir = self.prev_dest_dir


    def on_open_dir(self , event=None):
        self.prev_src_dir = self.directory 
        self.directory = filedialog.askdirectory()
        if self.directory:
            self.dir_watcher(event=None)
            self.filelist1.delete(0 , tk.END)
            file_list = os.listdir(self.directory)
            for filename in file_list:
                if os.path.splitext(filename)[1].lower() not in ('.jpeg' , '.jpg' , '.png'):
                    continue
                self.filelist1.insert(tk.END , filename)
            self.source_dir=self.directory
            if self.source_dir != self.prev_src_dir:
                self.entry1.config(state='normal')
                self.entry1.delete('0' , tk.END)
                self.entry1.insert(tk.END , self.source_dir)
                self.entry1.xview_moveto(1)
                self.entry1.config(state='disabled')
                self.prev_src_dir = self.source_dir
            if self.dest_dir:
                written_data = [self.source_dir + '\n', self.dest_dir + '\n' ]
                with open(r'./dir_config.ini' , 'w' , encoding='utf-8') as f:
                    f.writelines(written_data)
        else:
            self.directory = self.prev_src_dir

    def on_config_panel(self , event=None):
        if not self.config_panel or not self.config_panel.winfo_exists():
            self.config_panel = tk.Toplevel()
            self.config_panel.title('Settings')
            self.config_panel.geometry(f'+{self.winfo_x()}+{self.winfo_y()}')
            self.check1 = ttk.Checkbutton(self.config_panel, 
                                          text='Display file information window' , 
                                          variable=self.chk_var1 , 
                                          command=self.on_change1
                                          )
            self.check2 = ttk.Checkbutton(self.config_panel, 
                                          text='Enable image zoom on left-click' , 
                                          variable=self.chk_var2 , 
                                          command=self.on_change2
                                          )
            self.check3 = ttk.Checkbutton(self.config_panel, 
                                          text='When moving the file using the right key, display a warning.' , 
                                          variable=self.chk_var3
                                          )
            self.check1.pack(ipadx=10 , ipady=10)
            self.check2.pack(ipadx=10 , ipady=10)
            self.check3.pack(ipadx=10 , ipady=10)
        else:
            self.config_panel.deiconify()

    def on_change1(self , event=None):
        checked = self.chk_var1.get()
        if checked:
            self.win.deiconify()
        else:
            self.win.withdraw()

    def on_change2(self):
        checked = self.chk_var2.get()
        if checked:
            self.bind('<Button-1>' , self.on_resize_opt)
        else:
            self.unbind('<Button-1>')


    def on_copy_prompt(self , event=None):
        text = self.text1.get('1.0' , 'end-1c')
        self.clipboard_clear()
        self.clipboard_append(text)

    def on_copy_negative(self , event=None):
        text = self.text2.get('1.0' , 'end-1c')
        self.clipboard_clear()
        self.clipboard_append(text)

    def on_trim(self , event=None):
        self.unbind('<Button-1>')
        self.canvas.bind('<Button-1>' , self.on_choose_start)
        self.canvas.bind('<B1-Motion>' , self.on_drag)
        self.canvas.bind('<ButtonRelease-1>' , self.on_mouse_release)

    def on_choose_start(self , event=None): 
        logger.info("始点(%s , %s)" , event.x , event.y)
        self.start_x = event.x
        self.start_y = event.y
        
    def on_drag(self , event=None):
        self.canvas.delete('rect')
        self.canvas.create_rectangle(self.start_x , self.start_y , event.x , event.y , outline='white' , tags='rect')
        
    def on_mouse_release(self , event=None):
        logger.info('終点(%s , %s)' , event.x , event.y)
        self.canvas.delete('rect')
        self.canvas.create_rectangle(self.start_x , self.start_y , event.x , event.y , outline='blue' , tags='rect')
        self.end_x = event.x
        self.end_y = event.y
        result = messagebox.askyesno('Confirm' , 'Crop the rectangular region. Are you sure?')
        if result:
            self.canvas.delete('rect')
            PIL_img = ImageTk.getimage(self.image)
            cropped_image = PIL_img.crop((self.start_x , self.start_y , self.end_x , self.end_y))
            self.image = ImageTk.PhotoImage(cropped_image)
            self.canvas.create_image(0,0,image=self.image,anchor=tk.NW, tag="image")
            width = self.end_x - self.start_x
            height = self.end_y - self.start_y
            self.wm_geometry(f'{width}x{height}')
        else:
            self.canvas.delete('rect')
        self.canvas.unbind('<Button-1>')
        self.canvas.unbind('<B1-Motion>')
        self.canvas.unbind('<ButtonRelease-1>')
        self.bind('<Button-1>' , self.on_resize_opt)

    def on_mirror(self , event=None):
        if self.image:
            cv2_img = np.array(ImageTk.getimage(self.image))
            cv2_img = cv2.flip(cv2_img , 1)
            self.image = ImageTk.PhotoImage(Image.fromarray(cv2_img))
            self.canvas.create_image(0,0,image=self.image , anchor=tk.NW, tag="image")
        else:
            messagebox.showinfo('Confirm' , 'Display Image')

    def on_sepia(self , event=None):
        if self.image:
            img_pil = ImageTk.getimage(self.image)
            if img_pil.mode == 'RGBA':
                img_pil_rgb = img_pil.convert('RGB')
                sepia_image = sepia(img_pil_rgb)
            else:
                sepia_image = sepia(img_pil)
            self.image = ImageTk.PhotoImage(Image.fromarray(sepia_image))
            self.canvas.create_image(0,0,image=self.image , anchor=tk.NW, tag="image")
        else:
            messagebox.showinfo('Confirm' , 'Display Image')

    def on_gray_scale(self , event=None):
        if self.image:
            cv2_image = np.array(ImageTk.getimage(self.image))
            cv2_image = cv2.cvtColor(cv2_image , cv2.COLOR_RGB2GRAY)
            self.image = ImageTk.PhotoImage(Image.fromarray(cv2_image))
            self.canvas.create_image(0,0,image=self.image , anchor=tk.NW, tag="image")
        else:
            messagebox.showinfo('Confirm' , 'Display Image')

    def on_gamma(self , event=None):
        if not self.gamma:
            self.gamma = tk.Toplevel()
            self.gamma.title('Gamma Correction')
            self.gamma.geometry(f'+{self.winfo_x()+20}+{self.winfo_y()+20}')
            self.gamma_var = tk.DoubleVar()
            self.gamma_var.set(22)
            self.label_for_gamma = ttk.Label(self.gamma , text='Correction Value')
            self.scale_for_gamma = ttk.Scale(self.gamma , 
                                             from_=1 , 
                                             to=50 , 
                                             length=200 , 
                                             variable=self.gamma_var , 
                                             orient=tk.HORIZONTAL , 
                                             command=self.update_gamma_scale
                                         )
            self.update_gamma_scale(1)
            self.button_for_gamma = ttk.Button(self.gamma , text='＞' , command=self.on_exec_gamma)
            self.scale_for_gamma.grid(row=0 , column=0 , pady=10)
            self.label_for_gamma.grid(row=0 , column=1, padx=10 , pady=10)
            self.button_for_gamma.grid(row=1 , column=1 , padx=10 , pady=10)
    
    def update_gamma_scale(self , value):
        val = self.gamma_var.get()/10
        self.label_for_gamma.config(text=f'Correction Value:{val:.2f}')

    def on_exec_gamma(self , event=None):
        gamma = self.gamma_var.get()/10.0
        logger.debug('execute gamma correction event:%s gamma_var:%s' , event , gamma)
        inv_gamma = 1.0 / gamma
        image_cv2 = np.array(ImageTk.getimage(self.image))
        corrected_image = np.power(image_cv2 / 255.0 , inv_gamma) *255.0
        corrected_image = np.clip(corrected_image , 0 , 255).astype(np.uint8)
        self.image = ImageTk.PhotoImage(Image.fromarray(corrected_image))
        self.canvas.create_image(0,0,image=self.image,anchor=tk.NW, tag="image")

        self.canvas.create_image(0,0,image=self.image , anchor=tk.NW, tag="image")
        self.gamma.destroy()
        self.gamma = False

    def close_Handler(self , event=None):
        messagebox.showinfo('Confirm', 'The close button is disabled') 
        return

    def signal_handler(self , signal , frame):
        if self.watcher:
            self.watcher.stop()
            self.watcher.join()
        exit()

    def on_delete(self, event=None):
        if event == 1:
            index = self.filelist1.curselection()
            if index:
                index = index[0]
                selected_file = self.filelist1.get(index)
                os.remove(os.path.join(self.directory, selected_file))
                self.filelist1.delete(index)
                if index == self.filelist1.size():
                    # 最後の項目を削除した場合、選択を前の項目に変更
                    self.filelist1.selection_set(index - 1)
                    self.filelist1.activate(index - 1)
                else:
                    self.filelist1.selection_set(index)
                    self.filelist1.activate(index)
                event = tk.Event()
                event.widget = self.filelist1
                self.on_draw(event=event)
                self.filelist1.focus_force()
            else:
                messagebox.showerror('Error', 'Failed to delete file')
        elif event == 2:
            index = self.filelist2.curselection()
            if index:
                index = index[0]
                selected_file = self.filelist2.get(index)
                os.remove(os.path.join(self.dest_dir, selected_file))
                self.filelist2.delete(index)
                if index == self.filelist2.size():
                    # 最後の項目を削除した場合、選択を前の項目に変更
                    self.filelist2.selection_set(index - 1)
                    self.filelist2.activate(index - 1)
                else:
                    self.filelist2.selection_set(index)
                    self.filelist2.activate(index)
                event = tk.Event()
                event.widget = self.filelist2
                self.on_draw(event=event)
                self.filelist2.focus_force()
            else:
                messagebox.showerror('Error', 'Failed to delete file')


    def on_up(self, event=None):
        widget = event.widget
        switch = True
        if str(widget) != '.!toplevel.!frame.!listbox':
            switch = False
        if switch:
            selected_index = self.filelist1.curselection()
            if selected_index:
                active_index = int(selected_index[0]) - 1
                if active_index >= 0:
                    self.filelist1.selection_clear(0,tk.END)
                    self.filelist1.selection_set(active_index)
                    self.filelist1.see(active_index)
                    event = tk.Event()
                    event.widget = self.filelist1
                    self.on_draw(event=event)
            else:
                return
        else:
            selected_index = self.filelist2.curselection()
            if selected_index:
                active_index = int(selected_index[0]) - 1
                if active_index >= 0:
                    self.filelist2.selection_clear(0,tk.END)
                    self.filelist2.see(active_index)
                    self.filelist2.selection_set(active_index)
                    event = tk.Event()
                    event.widget = self.filelist2
                    self.on_draw(event=event)
            else:
                return

    def on_down(self, event=None):
        widget = event.widget
        switch = True
        if str(widget) != '.!toplevel.!frame.!listbox':
            switch = False
        if switch:
            selected_index = self.filelist1.curselection()
            if selected_index:
                active_index = int(selected_index[0]) + 1
                if active_index < self.filelist1.size():
                    self.filelist1.selection_clear(selected_index)
                    self.filelist1.selection_set(active_index)
                    event = tk.Event()
                    event.widget = self.filelist1
                    self.on_draw(event=event)
            else:
                return
        else:
            selected_index = self.filelist2.curselection()
            if selected_index:
                active_index = int(selected_index[0]) + 1
                if active_index < self.filelist2.size():
                    self.filelist2.selection_clear(selected_index)
                    self.filelist2.selection_set(active_index)
                    event = tk.Event()
                    event.widget = self.filelist2
                    self.on_draw(event=event)
            else:
                return

    def on_send(self , event=None):
        if self.senddir:
            selected_index = self.filelist1.curselection()
            result = True
            if self.chk_var3:
                result = messagebox.askyesno('confirm' , 'Do you want to transfer the file to the right listbox?')
            if result:
                if selected_index:
                    selected_index = selected_index[0]
                    target_file = self.filelist1.get(selected_index)
                    full_path = os.path.join(self.directory , target_file)
                    file_list = os.listdir(self.senddir)
                    logger.debug('%s' , file_list)
                    if target_file not in file_list:
                        dest = os.path.join(self.senddir , target_file)
                        os.rename(full_path , dest)
                        self.filelist2.insert(tk.END , target_file)
                        self.filelist1.delete(selected_index)
                        if selected_index == self.filelist1.size():
                            self.filelist1.selection_set(selected_index - 1)
                            self.filelist1.activate(selected_index - 1)
                        else:
                            self.filelist1.selection_set(selected_index)
                            self.filelist1.activate(selected_index)
                        event = tk.Event()
                        event.widget = self.filelist1
                        self.on_draw(event=event)
                        self.filelist1.focus_force()

                    else:
                        messagebox.showerror('Error','There is already a file with the same name at the destination')
            else:
                self.filelist1.focus_force()
        else:
            messagebox.showerror('Error' , 'Open the destination folder')

    def dir_watcher(self , event=None):
        if self.watcher:
            self.watcher.stop()
        self.watcher = DirectoryWatcher(directory=self.directory , filelist=self.filelist1)
        self.watcher.daemon = True
        self.watcher.start()

    def on_rename(self , value):
        switch = True
        if value == 2:
            switch = False
        if switch:
            index = self.filelist1.curselection()
        else:
            index = self.filelist2.curselection()
        if index:
            if switch:
                item = self.filelist1.get(index)
                new_name = simpledialog.askstring("Rename" , "Input New Filename:" , initialvalue=item)
                if new_name:
                    try:
                        os.rename(os.path.join(self.directory , item) , os.path.join(self.directory , new_name))
                        self.filelist1.delete(index)
                        self.filelist1.insert(index , new_name)
                        messagebox.showinfo('Confirm','The file name has been changed.')
                    except OSError:
                        messagebox.showerror('error','Failed to change Filename.')
            else:
                item = self.filelist2.get(index)
                new_name = simpledialog.askstring("Rename" , "Input New Filename:" , initialvalue=item)
                if new_name:
                    try:
                        os.rename(os.path.join(self.senddir , item) , os.path.join(self.senddir , new_name))
                        self.filelist2.delete(index)
                        self.filelist2.insert(index , new_name)
                        messagebox.showinfo('Confirm','The file name has been changed.')
                    except OSError:
                        messagebox.showerror('Error','Failed to change Filename.')

    def on_draw(self , event=None):
        widget = event.widget
        switch = True
        if str(widget) == '.!toplevel.!frame.!listbox':
            index = self.filelist1.curselection()
        else:
            index = self.filelist2.curselection()
            switch = False
        if index:
            if switch:
                target_file = self.filelist1.get(index)
                full_path = os.path.join(self.directory , target_file)
                # if os.path.splitext(full_path)[-1].lower() == 'zip':
                #     self.confirm_unzip()
            else:
                target_file = self.filelist2.get(index)
                full_path = os.path.join(self.senddir , target_file)
            self.image = self.original = ImageTk.PhotoImage(Image.open(full_path))
            if self.image:
                self.width , self.height = self.image.width() , self.image.height()
                self.canvas.create_image(0,0,image=self.image,anchor=tk.NW, tag="image")
                self.wm_geometry(f'{self.width}x{self.height}')
                self.update_status()
                logger.debug('%s , %s' ,full_path ,  os.path.splitext(full_path)[1].lower())
                self.attributes("-topmost", True)
                self.after_idle(self.attributes , '-topmost' , False)
                if os.path.splitext(full_path)[1].lower() == '.png':
                    if check_text_chunk(full_path):
                        self.set_AI_info(full_path)
                    else:
                        self.text1.config(state='normal')
                        self.text2.config(state='normal')
                        self.text3.config(state='normal')
                        self.text1.delete('1.0' , tk.END)
                        self.text2.delete('1.0' , tk.END)
                        self.text3.delete('1.0' , tk.END)
                        self.text1.config(state='disabled')
                        self.text2.config(state='disabled')
                        self.text3.config(state='disabled')
                else:
                    self.text1.config(state='normal')
                    self.text2.config(state='normal')
                    self.text3.config(state='normal')
                    self.text1.delete('1.0' , tk.END)
                    self.text2.delete('1.0' , tk.END)
                    self.text3.delete('1.0' , tk.END)
                    self.text1.config(state='disabled')
                    self.text2.config(state='disabled')
                    self.text3.config(state='disabled')
            else:
                messagebox.showerror('Error','Display Image.')

    def on_open_file(self , event=None):
        filetype =[('JPEG files' , '*.jpg;*.jpeg') , ('PNG files' , '*.png')]
        initialdir = os.getcwd()
        self.image_path = filedialog.askopenfilename(filetypes=filetype , initialdir=initialdir)
        if self.image_path:
            self.image = Image.open(self.image_path)
            self.width , self.height = self.image.size
            self.image = self.original = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0,0,image=self.image,anchor=tk.NW, tag="image")
            self.wm_geometry(str(self.width) + 'x' + str(self.height))
            if os.path.splitext(self.image_path)[1].lower() == '.png':
                if check_text_chunk(self.image_path):
                    self.set_AI_info(self.image_path)
                else:
                    self.text1.config(state='normal')
                    self.text2.config(state='normal')
                    self.text3.config(state='normal')
                    self.text1.delete('1.0' , tk.END)
                    self.text2.delete('1.0' , tk.END)
                    self.text3.delete('1.0' , tk.END)
                    self.text1.config(state='disabled')
                    self.text2.config(state='disabled')
                    self.text3.config(state='disabled')
            else:
                self.text1.config(state='normal')
                self.text2.config(state='normal')
                self.text3.config(state='normal')
                self.text1.delete('1.0' , tk.END)
                self.text2.delete('1.0' , tk.END)
                self.text3.delete('1.0' , tk.END)
                self.text1.config(state='disabled')
                self.text2.config(state='disabled')
                self.text3.config(state='disabled')
            self.canvas.create_image(0,0,image=self.image , anchor=tk.NW, tag="image")
            self.wm_geometry(f'{self.width}x{self.height}')
            self.update_status()

    def on_save_file(self , event=None):
        self.image_path = filedialog.asksaveasfilename(defaultextension='.jpg')
        if self.image_path:
            if self.image:
                name = tk_to_pil(self.image)
                name.save(self.image_path)
            else:
                messagebox.showerror('Error','Display Image')

    def on_gaussian_blur(self , event=None):
        if self.image:
            pil_image = tk_to_pil(self.image) 
            self.image_arr = np.array(pil_image)
            self.image_arr = cv2.cvtColor(self.image_arr , cv2.COLOR_RGB2BGR)
            self.popup_gaussian()
        else:
            messagebox.showerror('Error','Display Image')
            
    def on_mosaic(self , enent=None):
        if self.mosaic_window is None or (not hasattr(self.mosaic_window, 'winfo_exists')) or (not self.mosaic_window.winfo_exists()):
            self.mosaic_window = tk.Toplevel()
            self.mosaic_window.title('Settings Mosaic')
            self.mosaic_window.withdraw()
            self.mosaic_window.geometry(f'+{self.winfo_x()+20}+{self.winfo_y()+20}')
            self.mosaic_window.update()
            self.mosaic_window.deiconify()
            self.slidar_var = tk.DoubleVar()
            self.slidar_var.set(10.0)
            self.frame_mosaic = tk.Frame(self.mosaic_window)
            self.frame_mosaic.pack()
            self.mosaic_slidar_label = ttk.Label(self.frame_mosaic , text='')
            self.mosaic_slidar = ttk.Scale(self.frame_mosaic ,
                                            from_=1,
                                            to=30,
                                            length=200,
                                            orient=tk.HORIZONTAL,
                                            variable=self.slidar_var,
                                            command=self.update_mosaic_slidar)
            self.mosaic_exec_button = ttk.Button(self.frame_mosaic , text='Exec',command=self.on_exec_mosaic)
            self.mosaic_label = ttk.Label(self.frame_mosaic , text='Select Mosaic ->' , foreground='blue')
            self.parcial_mosaic_button = ttk.Button(self.frame_mosaic ,text='Specify the range' , command=self.on_exec_parcial_mosaic)
            self.mosaic_slidar.grid(row=0 , column= 0 , pady=10)
            self.mosaic_slidar_label.grid(row=0 , column=1 , pady=10)
            self.mosaic_exec_button.grid(row=0 , column=2 , padx=10 , pady=10)
            self.mosaic_label.grid(row=2 , column=0 , columnspan=2 , padx=10 , pady=10)
            self.parcial_mosaic_button.grid(row=2 , column=2 , padx=10 , pady=10)
            self.update_mosaic_slidar(self.slidar_var)

    def on_pencil(self , event=None):
        if self.image:
            img = ImageTk.getimage(self.image)
            arr_img = pencilsketch(np.array(img))
            arr_img = cv2.cvtColor(arr_img , cv2.COLOR_BGR2RGB)
            img = Image.fromarray(arr_img)
            self.image = ImageTk.PhotoImage(img)
            self.canvas.create_image(0,0,image=self.image,anchor=tk.NW, tag="image")
        else:
            messagebox.showerror('Error','Display Image')

    def undo(self , event=None):
        if not self.original:
            messagebox.showerror('Error','No image data available.')
            return
        image = tk_to_cv2(self.original)
        h , w , _ = image.shape
        image = cv2.resize(image , dsize=(w , h) , interpolation=cv2.INTER_LANCZOS4)
        image = cv2_to_pil(image)
        self.image = ImageTk.PhotoImage(image) 
        self.canvas.create_image(0,0,image=self.image,anchor=tk.NW, tag="image")
        self.wm_geometry(str(w)+'x'+str(h))
        self.update_status()
    
    def update_mosaic_slidar(self , event):
        var = self.slidar_var.get()
        self.mosaic_slidar_label.config(text=f'Sth:{var:.2f}')
        
    def on_exec_mosaic(self,event=None): 
        if not self.image: #PhotoImage
            self.mosaic_slidar_label.config(text='Display Image')
            return
        block_size = int(self.slidar_var.get())
        # varが大きいほど縮小率が上がる
        image = tk_to_cv2(self.image)
        h , w , _ = image.shape
        for y in range(0 , h , block_size):
            for x in range(0 , w , block_size):
                roi = image[y : y+block_size , x:x+block_size] # Region Of Interest
                mean = cv2.mean(roi)[:3]
                image[y:y+block_size , x:x+block_size] = mean
        image = cv2_to_pil(image)
        self.image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0,0,image=self.image,anchor=tk.NW, tag="image")
        self.mosaic_window.destroy()
        self.mosaic_window = None

    def on_exec_parcial_mosaic(self , even=None):
        self.unbind('<Button-1>')
        self.canvas.bind('<Button-1>' , self.on_choose_start)
        self.canvas.bind('<B1-Motion>' , self.on_drag)
        self.canvas.bind('<ButtonRelease-1>' , self.on_mouse_release_for_mosaic)

    def on_mouse_release_for_mosaic(self , event=None):
        if self.image:
            logger.info('終点(%s , %s)' , event.x , event.y)
            self.canvas.delete('rect')
            self.canvas.create_rectangle(self.start_x , self.start_y , event.x , event.y , outline='blue' , tags='rect')
            self.end_x = event.x
            self.end_y = event.y
            result = messagebox.askyesno('Confirm' , 'Mosaic the rectangular region. Are you sure?')
            if result:
                self.canvas.delete('rect')
                img = np.array(ImageTk.getimage(self.image))
                block_size = int(self.slidar_var.get())
                for y in range(self.start_y , self.end_y , block_size ):
                    for x in range(self.start_x , self.end_x , block_size):
                        roi = img[y:y+block_size , x:x+block_size ]
                        mean_color = cv2.mean(roi)[:3]
                        if img.shape[2] == 4:
                            rgba_array = np.zeros((block_size , block_size ,4 ) , dtype=np.uint8)
                            rgba_array[..., :3] = mean_color[:3]
                            alpha_channnel_slice = img[y:y+block_size , x:x+block_size , 3:4]
                            if len(alpha_channnel_slice) > 0:
                                rgba_array[...,3] = alpha_channnel_slice[:,:,0]
                            else:
                                rgba_array[...,3] = 255
                            img[y:y+block_size , x:x+block_size] = rgba_array
                        else:
                            color_array = np.ones((block_size, block_size, 3), dtype=np.uint8) * mean_color
                            img[y:y+block_size , x:x+block_size] = color_array
                self.image = ImageTk.PhotoImage(Image.fromarray(img))
                self.canvas.create_image(0,0,image=self.image,anchor=tk.NW, tag="image")
            else:
                self.canvas.delete('rect')
            self.canvas.unbind('<Button-1>')
            self.canvas.unbind('<B1-Motion>')
            self.canvas.unbind('<ButtonRelease-1>')
            self.bind('<Button-1>' , self.on_resize_opt)
            self.mosaic_window.destroy()
            self.mosaic_window = False
        else:
            messagebox.showerror('Error' , 'Display Image')
            self.mosaic_window.destroy()
            self.mosaic_window = False

    def popup_gaussian(self):
        if not self.gaus or ( not self.gaus.winfo_exists() ):
            self.gaus = tk.Toplevel(self)
            self.gaus.title('Gaussian')
            self.gaus.withdraw()
            self.gaus.geometry(f'+{self.winfo_x()+20}+{self.winfo_y()+20}')
            self.gaus.update()
            self.gaus.deiconify()
            self.gaus.protocol('WM_DELETE_WINDOW' , self.on_destroy_gaus)
            # self.label1_g = ttk.Label(self.gaus , text='Kernel size')
            self.label_kernel = ttk.Label(self.gaus ,text=None)
            self.kernel_value = tk.IntVar(value=5)
            val1 , val2 = 0 , 0
            self.scale1 = ttk.Scale(
                self.gaus , 
                from_=1 , 
                to=31 , 
                length=200 , 
                orient=tk.HORIZONTAL , 
                variable=self.kernel_value,
                command=lambda val1: self.on_update_gaus(val1))
            # self.label2_g = ttk.Label(self.gaus , text='Sigma_X')
            self.label_sigma = ttk.Label(self.gaus , text=None)
            self.sigma_value = tk.DoubleVar(value=13.0)

            self.scale2 = ttk.Scale(
                self.gaus , 
                from_=0 , 
                to=100 , 
                length=200 , 
                orient=tk.HORIZONTAL , 
                variable=self.sigma_value,
                command=lambda val2: self.on_update_gaus(val2))
            self.scale1.grid(row=1 , column = 0 , padx=10)
            self.label_kernel.grid(row=1 , column=1 , padx=10)
            self.scale2.grid(row=6 , column=0 , padx=10)
            self.label_sigma.grid(row=6 , column=1 , padx=10)
            self.scale1.bind("<MouseWheel>" , self.on_wheel_kernel)
            self.scale2.bind("<MouseWheel>" , self.on_wheel_sigma)
            self.on_update_gaus(None)
            
    def on_wheel_kernel(self , event=None):
        if event.delta > 0 and self.scale1.get() < self.scale1.cget('to'):
            self.scale1.set(self.scale1.get() + 1)
        elif event.delta < 0 and self.scale1.get() > self.scale1.cget('from'):
            self.scale1.set(self.scale1.get() - 1)
    
    def on_wheel_sigma(self , event=None):
        if event.delta > 0 and self.scale2.get() < self.scale2.cget('to'):
            self.scale2.set(self.scale2.get() + 1)
        elif event.delta < 0 and self.scale2.get() > self.scale2.cget('from'):
            self.scale2.set(self.scale2.get() - 1) 
        
    def on_destroy_gaus(self):
        self.image = self.image_tmp
        self.image_tmp = None
        self.gaus.destroy()
        self.gause = None
        self.kernel_value.set(5)
        self.sigma_value.set(13.0)

    def on_update_gaus(self , event=None):
        if self.image:
            kernel = self.kernel_value.get()
            sigma = self.sigma_value.get()/10
            self.label_kernel.config(text=f'Kernel:{kernel}')
            self.label_sigma.config(text=f'Sigma:{sigma:.2f}')
            if not event:
                return
            else:
                if kernel % 2 == 0:
                    kernel += 1  
            kernel_size = (kernel , kernel)
            self.image_arr = np.array(ImageTk.getimage(self.image))
            self.image_arr = cv2.GaussianBlur(self.image_arr , ksize=kernel_size , sigmaX=sigma)
            self.image_tmp = ImageTk.PhotoImage(Image.fromarray(self.image_arr))
            self.canvas.delete('image')
            self.canvas.create_image(0,0,image=self.image_tmp,anchor=tk.NW, tag="image")
        else:
            messagebox.showerror('Error','Display Image')

    def show_menu(self , event=None):
        self.editmenu.post(event.x_root , event.y_root)

    def on_paste(self , event=None):
        self.image = ImageGrab.grabclipboard()
        self.text1.config(state='normal')
        self.text2.config(state='normal')
        self.text3.config(state='normal')
        self.text1.delete('1.0' , tk.END)
        self.text2.delete('1.0' , tk.END)
        self.text3.delete('1.0' , tk.END)
        self.text1.config(state='disabled')
        self.text2.config(state='disabled')
        self.text3.config(state='disabled')
        if isinstance(self.image , Image.Image):
            self.image = self.image.convert('RGB')
            self.width , self.height = self.image.size
            self.image = ImageTk.PhotoImage(self.image)
            if self.image:
                self.original = self.image
                self.canvas.create_image(0,0,image=self.image,anchor=tk.NW, tag="image")
                self.wm_geometry(str(self.width)+'x'+str(self.height))
                self.update_status()
        elif isinstance(self.image , bytes):
            data = self.image[2:]
            rgb_image = Image.frombytes(
                'RGB' , (self.image[0] , self.image[1]) , data , 'raw' , 'BGRX'
            )
            self.width , self.height = rgb_image.size
            self.image = ImageTk.PhotoImage(rgb_image)
            if self.image:
                self.original = self.image
                self.canvas.create_image(0,0,image=self.image , anchor=tk.NW, tag="image")
                self.wm_geometry(str(self.width) + 'x' + str(self.height))
                self.update_status()
        elif isinstance(self.image , list):
            img = Image.open(self.image[0])
            self.width , self.height = img.size
            self.image = ImageTk.PhotoImage(img)
            if self.image:
                self.original = self.image
                self.canvas.create_image(0,0,image=self.image,anchor=tk.NW, tag="image")
                self.wm_geometry(str(self.width) + 'x' + str(self.height))
                self.update_status()
        else:
            messagebox.showerror('Error','Clipboad is empty')

    def copy_to_clipboard(self , event=None):
        if self.image:
            # メモリストリームにBMP形式で保存してから読み出す
            output = io.BytesIO()
            img_pil = tk_to_pil(self.image)
            img_pil.convert('RGB').save(output, 'BMP')
            # ヘッダー14バイトを除いたものがDevice Independent Bitmap
            data = output.getvalue()[14:] 
            output.close()
            send_to_clipboard(win32clipboard.CF_DIB, data)
        else:
            messagebox.showerror('Error','Display Image')

    def on_resize(self , event=None):
        global top_window
        top_window = tk.Toplevel()
        top_window.title('Resize')
        top_window.geometry(f'+{self.winfo_x()+10}+{self.winfo_y()+10}')
        self.resize_label = ttk.Label(top_window , text='Magnification')
        self.var = tk.DoubleVar()
        self.var.set(11.0)
        self.update_resize_scale(self.var)
        self.resize_slidar = ttk.Scale(top_window , from_=1 , to=30 , length=200 , variable=self.var , command=self.update_resize_scale)
        self.resize_button = ttk.Button(top_window , text='exec.' , command=self.on_exec_resize)
        self.resize_slidar.grid(row=0 , column=0 , pady=10)
        self.resize_label.grid(row=0 , column=1 , padx=10 ,  pady=10)
        self.resize_button.grid(row=1, column=1 , padx=10 , pady=10)
        self.update_status()

    def on_resize_opt(self , event=None):
        if self.image:
            cv2_img = np.array(ImageTk.getimage(self.image))
            cv2_img = cv2.resize(cv2_img , dsize=None , fx=1.05 , fy=1.05 , interpolation=cv2.INTER_LANCZOS4)
            h , w , _ = cv2_img.shape
            self.image = ImageTk.PhotoImage(Image.fromarray(cv2_img))
            self.canvas.create_image(0, 0 , image=self.image , anchor=tk.NW, tag="image")
            self.wm_geometry(f'{w}x{h}')
            self.update_status()

    def update_resize_scale(self , value):
        var = self.var.get()/10
        self.resize_label.config(text=f'{var:.2f}x')

    def on_exec_resize(self , event=None):
        if self.image:
            var = int(self.var.get())
            arr_img = np.array(ImageTk.getimage(self.image))
            rate = var / 10
            arr_img = cv2.resize(arr_img , dsize=None , fx=rate , fy=rate ,interpolation=cv2.INTER_LANCZOS4)
            height , width , _ = arr_img.shape
            self.image = ImageTk.PhotoImage(Image.fromarray(arr_img))
            self.canvas.create_image(0,0,image=self.image,anchor=tk.NW, tag="image")
            self.wm_geometry(f'{width}x{height}')
            self.update_status
            top_window.destroy()
        else:
            messagebox.showerror('Error','Display Image')
            top_window.destroy()
    
    def on_change(self,event=None):
        self.popup = tk.Toplevel()
        self.popup.title('Change Channel Value')
        self.popup.geometry(f'+{self.winfo_x()+20}+{self.winfo_y()-20}')
        self.popup.protocol('WM_DELETE_WINDOW' , self.on_change_image)
        self.label1_rgb = ttk.Label(self.popup,text='B', width= 15)
        self.label2_rgb = ttk.Label(self.popup,text='G', width= 15)
        self.label3_rgb = ttk.Label(self.popup,text='R', width= 15)
        self.var_B = tk.DoubleVar()
        self.var_G = tk.DoubleVar()
        self.var_R = tk.DoubleVar()
        self.var_B.set(10.0)
        self.var_G.set(10.0)
        self.var_R.set(10.0)

        
        self.blue_scale = ttk.Scale(self.popup ,
                                    from_=0 ,
                                    to=30 , 
                                    length=200 , 
                                    orient=tk.HORIZONTAL,
                                    variable=self.var_B,
                                    command=self.update_rgb)
        self.green_scale = ttk.Scale(self.popup ,
                                    from_=0 ,
                                    to=30 , 
                                    length=200 , 
                                    orient=tk.HORIZONTAL,
                                    variable=self.var_G,
                                    command=self.update_rgb)
        self.red_scale = ttk.Scale(self.popup ,
                                    from_=0 ,
                                    to=30 , 
                                    length=200 , 
                                    orient=tk.HORIZONTAL,
                                    variable=self.var_R,
                                    command=self.update_rgb)
        # self.convert_button = ttk.Button(self.popup , text='＞' , command=self.on_exec_change)
        self.blue_scale.grid(row=0 , column=0 , pady=10)
        self.green_scale.grid(row=1 , column=0 , pady=10)
        self.red_scale.grid(row=2 , column=0 , pady=10)
        self.label1_rgb.grid(row=0 , column=1 , padx=10 , pady=10)
        self.label2_rgb.grid(row=1 , column=1 , padx=10 , pady=10)
        self.label3_rgb.grid(row=2 , column=1 , padx=10 , pady=10)
        # self.convert_button.grid(row=3 , column=1 , padx=10 , pady=10)
        self.blue_scale.bind("<MouseWheel>" , self.on_wheel_B)
        self.green_scale.bind("<MouseWheel>" , self.on_wheel_G)
        self.red_scale.bind("<MouseWheel>" , self.on_wheel_R)
        self.update_rgb(None)
        self.update_rgb(None)
        self.update_rgb(None)
        
    def on_change_image(self):
        self.image = self.image_tmp
        self.popup.destroy()
        self.popup = None
        
    def on_wheel_B(self , event=None):
        if event.delta > 0 and self.blue_scale.get() < self.blue_scale.cget('to'):
            self.blue_scale.set(self.blue_scale.get() + 1)
        elif event.delta < 0 and self.blue_scale.get() > self.blue_scale.cget('from'):
            self.blue_scale.set(self.blue_scale.get() - 1)
            
    def on_wheel_G(self , event=None):
        if event.delta > 0 and self.green_scale.get() < self.green_scale.cget('to'):
            self.green_scale.set(self.green_scale.get() + 1)
        elif event.delta < 0 and self.green_scale.get() > self.green_scale.cget('from'):
            self.green_scale.set(self.green_scale.get() - 1)
            
    def on_wheel_R(self , event=None):
        if event.delta > 0 and self.red_scale.get() < self.red_scale.cget('to'):
            self.red_scale.set(self.red_scale.get() + 1)
        elif event.delta < 0 and self.red_scale.get() > self.red_scale.cget('from'):
            self.red_scale.set(self.red_scale.get() - 1)
        
        
    def update_rgb(self , value=None):
        if self.image:
            if self.label1_rgb.winfo_exists():
                val1 = self.var_B.get()/10
                val2 = self.var_G.get()/10
                val3 = self.var_R.get()/10
                self.label1_rgb.config(text=f'B:{val1:.1f}')
                self.label2_rgb.config(text=f'G:{val2:.1f}')
                self.label3_rgb.config(text=f'R:{val3:.1f}')
                self.image_arr = np.array(ImageTk.getimage(self.image)).astype(np.float32)/255.0
                blue = self.image_arr[:,:,2]
                green = self.image_arr[:,:,1]
                red = self.image_arr[:,:,0]
                blue *= val1
                green *= val2
                red *= val3
                self.image_arr= cv2.merge((red,green,blue))
                self.image_arr = np.clip(self.image_arr , 0  , 1)
                arr_uint8 = (self.image_arr*255).astype(np.uint8)
                self.image_tmp = ImageTk.PhotoImage(Image.fromarray(arr_uint8))
                self.canvas.create_image(0,0,image=self.image_tmp,anchor=tk.NW, tag="image")
        else:
            messagebox.showerror('Error' , 'Display Image!!!')
            self.popup.destroy()
            self.popup = None
            
    def quit(self):
        self.destroy()

    def run(self):
        self.mainloop()

app = App()
app.run()