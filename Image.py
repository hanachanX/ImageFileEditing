import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
from PIL import Image , ImageTk , ImageGrab
import cv2
import numpy as np
import io
import win32clipboard
import logging
from tkinter import messagebox
from tkinter import simpledialog
import threading
import time
import signal
import re
import subprocess

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s' , level=logging.WARNING , encoding='utf-8')
logger = logging.getLogger(__name__)


def check_text_chunk(filepath):
    try:
        with open(filepath, 'rb') as file:
            signature = file.read(8)
            if signature != b'\x89PNG\r\n\x1a\n': # 8バイトシグネチャがPNG形式でない。
                return False

            while True:
                length_bytes = file.read(4)
                if len(length_bytes) < 4:
                    break
                length = int.from_bytes(length_bytes, 'big')
                chunk_type = file.read(4)
                if chunk_type == b'tEXt':
                    return True
                file.seek(length + 4, 1)  # チャンクデータとCRC4バイトはスキップする。
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


class App(tk.Tk):
    def __init__(self , **kwargs):
        super().__init__(**kwargs)
        # メインウインドウ設定
        self.title('IIMAGE FILE TOOL')
        self.geometry('512x768+100+100')
        self.resizable(False , False)

        # セカンドパネルの作成
        self.win = tk.Toplevel()
        self.win.title('File Info')
        self.win.geometry('600x930+612+100')
        self.win.protocol('WM_DELETE_WINDOW' , self.close_Handler)
        self.win.geometry(f'+{self.winfo_x()+800}+{self.winfo_y()+500}')

        self.frame1 = ttk.Frame(self.win)
        self.frame1.grid(row=0,column=0 , pady=10) 
        # リストボックス1の作成

        self.filelist1 = tk.Listbox(self.frame1 , width=40 , height=30 )
        self.filelist1.pack(side='left' , fill='y')
        self.scroll1 = ttk.Scrollbar(self.frame1 , command=self.filelist1.yview)
        self.scroll1.pack(side='left' , fill='y')
        self.filelist1.config(yscrollcommand=self.scroll1.set)

        self.frame2 = ttk.Frame(self.win)
        self.frame2.grid(row=0 , column=1 , pady=10)
        # リストボックス2の作成
        self.filelist2 = tk.Listbox(self.frame2 , width=40 , height=30)
        self.filelist2.pack(side='left' , fill='y')
        self.scroll2 = ttk.Scrollbar(self.frame2 , command=self.filelist2.yview)
        self.scroll2.pack(side='left' , fill='y')
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
        self.processmenu.add_command(label='Gamma corr.', command=self.on_gamma)
        self.processmenu.add_command(label='Gray conv.' , command=self.on_gray_scale)
        self.processmenu.add_command(label='Sepia conv.' , command=self.on_sepia)
        self.processmenu.add_command(label='Mirror' , command=self.on_mirror)
        self.processmenu.add_command(label='Trim' , command=self.on_trim)
        self.menubar.add_cascade(label='Conv' , menu=self.processmenu)
        
        # 編集メニュー
        self.editmenu = tk.Menu(self.menubar , tearoff=0)
        self.editmenu.add_command(label='Paste' , command=self.on_paste)
        self.editmenu.add_command(label='Copy', command=self.copy_to_clipboard)
        self.editmenu.add_command(label='Undo' , command=self.undo) 
        self.menubar.add_cascade(label='Edit' , menu=self.editmenu)

        #　設定メニュー
        self.configmenu = tk.Menu(self.menubar , tearoff=0)
        self.configmenu.add_command(label='Settings' , command=self.on_config_panel)
        self.menubar.add_cascade(label='Settings' , menu=self.configmenu)
        self.config(menu=self.menubar)

        # パネル2にメニュー追加
        self.filemenubar = tk.Menu(self.win)
        self.filemenu2 = tk.Menu(self.filemenubar , tearoff=False)
        self.filemenu2.add_command(label='Read Folder1' , command=self.on_open_dir)
        self.filemenu2.add_command(label='Read Folder2' , command=self.on_open_send_dir)
        self.filemenu2.add_command(label='Open in Explorer(Left)' , command=lambda : self.on_open_explorer(1))
        self.filemenu2.add_command(label='Open in Explorer(Right)' , command=lambda : self.on_open_explorer(2))
        self.filemenu2.add_command(label='Rename(Left)' , command= lambda : self.on_rename(1))
        self.filemenu2.add_command(label='Rename(Right)' , command= lambda : self.on_rename(2))
        self.filemenubar.add_cascade(label='Command' , menu=self.filemenu2)
        self.sendmenu = tk.Menu(self.filemenubar , tearoff=False)
        self.win.config(menu=self.filemenubar)

        # Stable Diffusion情報
        self.target_text1 = ''
        self.target_text2 = ''
        self.target_text3 = ''

        

        # トリミングのためのCanvasの作成
        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH , expand=True)

        # 汎用変数
        self.image_path = ''
        self.directory=None # Sourse Directory
        self.senddir=None # Destination Directory
 
        # ImageTk オブジェクト 
        self.image = None # PhotoImage
        self.original = None # Undo用
        self.popup = None # ポップアップウインドウ
        self.width = None # self.imageの幅
        self.height = None # self.imageの高さ

        # GaussianBlur関数用変数
        self.image_arr = None # CV2 image
        self.kernel_value = 5 
        self.sigma_value = 13

        # クリックポップアップウィンドウ
        self.bind('<Button-1>' , self.on_resize_opt) # リサイズ
        self.bind('<Button-3>' , self.show_menu) # 編集メニュー
        
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

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
        self.filelist1.bind('<Delete>' , self.on_delete)

        # サブスレッドのためのシグナルオブジェクト
        self.watcher = None
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # ガンマ補正のサブウインドウ
        self.gamma = None

        # 設定画面
        self.config_panel = None
        self.chk_var1 = tk.BooleanVar()
        self.chk_var2 = tk.BooleanVar()
        self.chk_var1.set(True)
        self.chk_var2.set(True)

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
                    self.dir_watcher(event=None)
        else:
            with open(dirfile_dir , 'w') as f:
                pass

        if self.source_dir:
            file_list = os.listdir(self.source_dir) 
            for filename in file_list:
                if os.path.splitext(filename)[1] not in ('.jpeg' , '.jpg' , '.png'):
                    continue
                self.filelist1.insert(tk.END , filename)

        if self.dest_dir:
            file_list = os.listdir(self.dest_dir)
            for filename in file_list:
                if os.path.splitext(filename)[1] not in ('.jpeg' , '.jpg' , '.png'):
                    continue
                self.filelist2.insert(tk.END , filename)

    def set_AI_info(self , image_path):
        with open(image_path , 'rb') as f:
            data = f.read()
        # テキストチャンクの先頭インデックスを取得
        index = data.find(b'\x74\x45\x58\x74')
        # テキスト長を取得
        length = int.from_bytes(data[index-4:index], byteorder='big')
        # テキストデータを取得
        text_data = data[index+4:index+4+length]
        # keyword(parameters)を読み飛ばしテキストを取得する
        txt = text_data.split(b'\x00')
        text = txt[1].decode('shift-jis')
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
        self.senddir = filedialog.askdirectory()
        if self.senddir:
            file_list = os.listdir(self.senddir)
            self.filelist2.delete(0 , tk.END)
            for filename in file_list:
                if os.path.splitext(filename)[1] not in ('.jpeg' , '.jpg' , '.png'):
                    continue
                self.filelist2.insert(tk.END , filename)
            self.dest_dir = self.senddir
            if self.source_dir:
                written_data = [self.source_dir + '\n', self.dest_dir + '\n']
                with open(r'./dir_config.ini' , 'w' , encoding='utf-8') as f:
                    f.writelines(written_data)

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
            self.check1.pack(ipadx=10 , ipady=10)
            self.check2.pack(ipadx=10 , ipady=10)
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
        self.canvas.bind('<Button-1>' , self.on_trim_start)
        self.canvas.bind('<B1-Motion>' , self.on_drag)
        self.canvas.bind('<ButtonRelease-1>' , self.on_mouse_release)

    def on_trim_start(self , event=None):
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
            self.canvas.create_image(0,0,image=self.image,anchor=tk.NW)
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
            self.canvas.create_image(0,0,image=self.image , anchor=tk.NW)
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
            self.canvas.create_image(0,0,image=self.image , anchor=tk.NW)
        else:
            messagebox.showinfo('Confirm' , 'Display Image')

    def on_gray_scale(self , event=None):
        if self.image:
            cv2_image = np.array(ImageTk.getimage(self.image))
            cv2_image = cv2.cvtColor(cv2_image , cv2.COLOR_RGB2GRAY)
            self.image = ImageTk.PhotoImage(Image.fromarray(cv2_image))
            self.canvas.create_image(0,0,image=self.image , anchor=tk.NW)
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
        self.canvas.create_image(0,0,image=self.image,anchor=tk.NW)

        self.canvas.create_image(0,0,image=self.image , anchor=tk.NW)
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
                else:
                    messagebox.showerror('Error','There is already a file with the same name at the destination')
        else:
            messagebox.showerror('Error' , 'Open the destination folder')

    def dir_watcher(self , event=None):
        if self.watcher:
            self.watcher.stop()
        self.watcher = DirectoryWatcher(directory=self.directory , filelist=self.filelist1)
        self.watcher.daemon = True
        self.watcher.start()

    def on_open_dir(self , event=None):
        self.directory = filedialog.askdirectory()
        if self.directory:
            self.dir_watcher(event=None)
            self.filelist1.delete(0 , tk.END)
            file_list = os.listdir(self.directory)
            for filename in file_list:
                if os.path.splitext(filename)[1] not in ('.jpeg' , '.jpg' , '.png'):
                    continue
                self.filelist1.insert(tk.END , filename)
            self.source_dir=self.directory
            if self.dest_dir:
                written_data = [self.source_dir , self.dest_dir ]
                with open(r'./dir_config.ini' , 'w' , encoding='utf-8') as f:
                    f.writelines(written_data)

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
            else:
                target_file = self.filelist2.get(index)
                full_path = os.path.join(self.senddir , target_file)
            self.image = self.original = ImageTk.PhotoImage(Image.open(full_path))
            if self.image:
                self.width , self.height = self.image.width() , self.image.height()
                self.canvas.create_image(0,0,image=self.image,anchor=tk.NW)
                self.canvas.create_image(0,0,image=self.image , anchor=tk.NW)
                self.wm_geometry(f'{self.width}x{self.height}')
                logger.debug('%s , %s' ,full_path ,  os.path.splitext(full_path)[1].lower())
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
            self.canvas.create_image(0,0,image=self.image,anchor=tk.NW)
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
            self.canvas.create_image(0,0,image=self.image , anchor=tk.NW)
            self.wm_geometry(f'{self.width}x{self.height}')

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
        global top_window
        top_window = tk.Toplevel()
        top_window.title('Settings Mosaic')
        self.slidar_var = tk.DoubleVar()
        self.slidar_var.set(10.0)
        self.frame_mosaic = tk.Frame(top_window)
        self.frame_mosaic.pack()
        self.mosaic_slidar_label = ttk.Label(self.frame_mosaic , text='')
        self.mosaic_slidar = ttk.Scale(self.frame_mosaic ,
                                        from_=1,
                                        to=30,
                                        length=200,
                                        orient=tk.HORIZONTAL,
                                        variable=self.slidar_var,
                                        command=self.update_mosaic_slidar)
        self.mosaic_exec_button = ttk.Button(self.frame_mosaic , text='＞',command=self.on_exec_mosaic)
        self.undo_button = ttk.Button(self.frame_mosaic , text='Undo' , command=self.undo)
        self.mosaic_slidar.grid(row=0 , column= 0 , pady=10)
        self.mosaic_slidar_label.grid(row=0 , column=1 , pady=10)
        self.mosaic_exec_button.grid(row=1 , column=1 , padx=10 , pady=10)
        self.undo_button.grid(row=1 , column=2 , padx=10 , pady=10)

    def on_pencil(self , event=None):
        if self.image:
            img = ImageTk.getimage(self.image)
            arr_img = pencilsketch(np.array(img))
            arr_img = cv2.cvtColor(arr_img , cv2.COLOR_BGR2RGB)
            img = Image.fromarray(arr_img)
            self.image = ImageTk.PhotoImage(img)
            self.canvas.create_image(0,0,image=self.image,anchor=tk.NW)
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
        self.canvas.create_image(0,0,image=self.image,anchor=tk.NW)
        self.wm_geometry(str(w)+'x'+str(h))
    
    def update_mosaic_slidar(self , event):
        var = self.slidar_var.get()
        self.mosaic_slidar_label.config(text=f'Strength{var:.2f}')
        
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
        self.canvas.create_image(0,0,image=self.image,anchor=tk.NW)
        self.canvas.create_image(0,0,image=self.image , anchor=tk.NW)
        top_window.destroy()

    def popup_gaussian(self):
        global top_window
        top_window = tk.Toplevel(self)
        top_window.title('Settings')
        label1 = ttk.Label(top_window , text='Kernel size')
        self.label_kernel = ttk.Label(top_window ,text=None)
        self.kernel_value = tk.DoubleVar(value=5.0)
        scale1 = ttk.Scale(
            top_window , 
            from_=1 , 
            to=31 , 
            length=200 , 
            orient=tk.HORIZONTAL , 
            variable=self.kernel_value,
            command=self.update_label_kernel)
        label2 = ttk.Label(top_window , text='Sigma_X')
        self.label_sigma = ttk.Label(top_window , text=None)
        self.sigma_value = tk.DoubleVar(value=13.0)
        scale2 = ttk.Scale(
            top_window , 
            from_=0 , 
            to=100 , 
            length=200 , 
            orient=tk.HORIZONTAL , 
            variable=self.sigma_value,
            command=self.update_sigma_label)
        label1.grid(row=0 , column = 0 , padx=10)
        scale1.grid(row=1 , column = 0 , padx=10)
        self.label_kernel.grid(row=1 , column=1 , padx=10)
        label2.grid(row=5 , column=0 , padx=10)
        scale2.grid(row=6 , column=0 , padx=10)
        self.label_sigma.grid(row=6 , column=1 , padx=10)
        button = ttk.Button(top_window , text='Convert' , command=self.on_Gaussian)
        button.grid(row=7 , column=1 , padx=10 , pady=10)

    def update_label_kernel(self , value):
        self.label_kernel.config(text=f'{float(value):.1f}')

    def update_sigma_label(self , value):
        self.label_sigma.config(text=f'{float(value)/10:.2f}')

    def get_kernel(self, value):
        return int(value)

    def get_sigma(self, value):
        return float(f"{value:.2f}")

    def on_Gaussian(self , event=None):
        if self.image_arr.any():
            kernel_size = self.get_kernel(self.kernel_value.get())
            if not kernel_size % 2:
                kernel_size += 1
            kernel = (kernel_size , kernel_size)
            sigmaX = self.get_sigma(int(self.sigma_value.get()/10))
            gaus = cv2.GaussianBlur(self.image_arr , ksize=kernel , sigmaX=sigmaX)
            self.image = cv2_to_pil(gaus)
            self.image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0,0,image=self.image,anchor=tk.NW)
            top_window.destroy()
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
                self.canvas.create_image(0,0,image=self.image,anchor=tk.NW)
                self.wm_geometry(str(self.width)+'x'+str(self.height))
        elif isinstance(self.image , bytes):
            data = self.image[2:]
            rgb_image = Image.frombytes(
                'RGB' , (self.image[0] , self.image[1]) , data , 'raw' , 'BGRX'
            )
            self.width , self.height = rgb_image.size
            self.image = ImageTk.PhotoImage(rgb_image)
            if self.image:
                self.original = self.image
                self.canvas.create_image(0,0,image=self.image , anchor=tk.NW)
                self.wm_geometry(str(self.width) + 'x' + str(self.height))
        elif isinstance(self.image , list):
            img = Image.open(self.image[0])
            self.width , self.height = img.size
            self.image = ImageTk.PhotoImage(img)
            if self.image:
                self.original = self.image
                self.canvas.create_image(0,0,image=self.image,anchor=tk.NW)
                self.wm_geometry(str(self.width) + 'x' + str(self.height))
        else:
            messagebox.showerror('Error','Display Image')

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
        self.resize_label = ttk.Label(top_window , text='Magnification')
        self.var = tk.DoubleVar()
        self.var.set(11.0)
        self.resize_slidar = ttk.Scale(top_window , from_=1 , to=30 , length=200 , variable=self.var , command=self.update_resize_scale)
        self.resize_button = ttk.Button(top_window , text='＞' , command=self.on_exec_resize)
        self.resize_slidar.grid(row=0 , column=0 , pady=10)
        self.resize_label.grid(row=0 , column=1 , padx=10 ,  pady=10)
        self.resize_button.grid(row=1, column=1 , padx=10 , pady=10)

    def on_resize_opt(self , event=None):
        if self.image:
            cv2_img = np.array(ImageTk.getimage(self.image))
            cv2_img = cv2.resize(cv2_img , dsize=None , fx=1.05 , fy=1.05)
            h , w , _ = cv2_img.shape
            self.image = ImageTk.PhotoImage(Image.fromarray(cv2_img))
            self.canvas.create_image(0, 0 , image=self.image , anchor=tk.NW)
            self.wm_geometry(f'{w}x{h}')

    def update_resize_scale(self , value):
        var = self.var.get()/10
        self.resize_label.config(text=f'Magnification:{var:.2f}')

    def on_exec_resize(self , event=None):
        if self.image:
            var = int(self.var.get())
            arr_img = np.array(ImageTk.getimage(self.image))
            rate = var / 10
            arr_img = cv2.resize(arr_img , dsize=None , fx=rate , fy=rate ,interpolation=cv2.INTER_LANCZOS4)
            height , width , _ = arr_img.shape
            self.image = ImageTk.PhotoImage(Image.fromarray(arr_img))
            self.canvas.create_image(0,0,image=self.image,anchor=tk.NW)
            self.wm_geometry(str(width) +'x' + str(height))
            top_window.destroy()
        else:
            messagebox.showerror('Error','Display Image')
            top_window.destroy()
    
    def on_change(self,event=None):
        self.popup = tk.Toplevel()
        self.popup.title('Change Channel Value')
        self.popup.geometry(f'+{self.winfo_x()+20}+{self.winfo_y()-20}')
        self.label1 = ttk.Label(self.popup,text='Blue')
        self.label2 = ttk.Label(self.popup,text='Green')
        self.label3 = ttk.Label(self.popup,text='Red')
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
                                    command=self.update_blue)
        self.green_scale = ttk.Scale(self.popup ,
                                    from_=0 ,
                                    to=30 , 
                                    length=200 , 
                                    orient=tk.HORIZONTAL,
                                    variable=self.var_G,
                                    command=self.update_green)
        self.red_scale = ttk.Scale(self.popup ,
                                    from_=0 ,
                                    to=30 , 
                                    length=200 , 
                                    orient=tk.HORIZONTAL,
                                    variable=self.var_R,
                                    command=self.update_red)
        self.convert_button = ttk.Button(self.popup , text='＞' , command=self.on_exec_change)
        self.blue_scale.grid(row=0 , column=0 , pady=10)
        self.green_scale.grid(row=1 , column=0 , pady=10)
        self.red_scale.grid(row=2 , column=0 , pady=10)
        self.label1.grid(row=0 , column=1 , padx=10 , pady=10)
        self.label2.grid(row=1 , column=1 , padx=10 , pady=10)
        self.label3.grid(row=2 , column=1 , padx=10 , pady=10)
        self.convert_button.grid(row=3 , column=1 , padx=10 , pady=10)
        
    def update_blue(self , value):
        val = self.var_B.get()/10
        self.label1.config(text=f'B:{val:.1f}')
        
    def update_green(self , value):
        val = self.var_G.get()/10
        self.label2.config(text=f'G:{val:.1f}')
        
    def update_red(self , value):
        val = self.var_R.get()/10
        self.label3.config(text=f'R:{val:.1f}')
    
    def on_exec_change(self , event=None):
        if self.image:
            B = self.var_B.get()/10
            G = self.var_G.get()/10
            R = self.var_R.get()/10
            arr_image = np.array(ImageTk.getimage(self.image)).astype(np.float32)
            arr_image = cv2.cvtColor(arr_image , cv2.COLOR_RGB2BGR)
            arr_image /= 255.0
            blue_array = arr_image[:,:,0]
            green_array = arr_image[:,:,1]
            red_array = arr_image[:,:,2]
            blue_array *= B
            green_array *= G
            red_array *= R
            img = arr_image.copy()
            img[:,:,0] = blue_array
            img[:,:,1] = green_array
            img[:,:,2] = red_array
            img = np.clip(img ,0 , 1.0)
            arr_uint8 = (img*255).astype(np.uint8)
            arr_uint8 = cv2.cvtColor(arr_uint8 , cv2.COLOR_BGR2RGB)
            self.image = ImageTk.PhotoImage(Image.fromarray(arr_uint8))
            self.canvas.create_image(0,0,image=self.image,anchor=tk.NW)
            self.popup.destroy()
        else:
            messagebox.showerror('Error','Display Image')
            
    def quit(self):
        self.destroy()

    def run(self):
        self.mainloop()

app = App()
app.run()