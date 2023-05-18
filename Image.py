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

logging.basicConfig(filename='tcode01.log' , level=logging.INFO , encoding='utf-8')
logger = logging.getLogger(__name__)

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
        self.filelist = filelist
        self.file_set = set()
        self.running = True

    def run(self):
        while self.running:
            files = os.listdir(self.directory)
            new_files = [file for file in files if file not in self.file_set and os.path.splitext(file)[1].lower() in ('.jpeg' , '.jpg' , '.png')]
            for file in new_files:
                self.filelist.insert(tk.END , file)
                self.file_set.add(file)

            time.sleep(1)

    def stop(self):
        self.running = False


class App(tk.Tk):
    def __init__(self , **kwargs):
        super().__init__(**kwargs)
        # Basic Settings
        self.title('画像加工')
        self.geometry('800x800')
        self.resizable(False , False)

        # Create Secondary Window
        self.win2 = tk.Toplevel()
        self.win2.title('ファイル')
        self.win2.geometry('300x800')
        self.win2.protocol('WM_DELETE_WINDOW' , self.close_Handler)
        self.win2.geometry(f'+{self.winfo_x()+1100}+{self.winfo_y()+250}')

        # List box belong in Win2
        self.filelist = tk.Listbox(self.win2)
        self.filelist.pack(side=tk.LEFT , fill=tk.BOTH , expand=True)

        # ScrollBar for ListBox
        self.scroll = ttk.Scrollbar(self.win2 , orient=tk.VERTICAL , command=self.filelist.yview)
        self.scroll.pack(side=tk.RIGHT , fill=tk.Y)
        self.filelist.config(yscrollcommand=self.scroll.set)

        # Create Menu for Win2
        self.filemenubar = tk.Menu(self.win2)
        self.filemenu = tk.Menu(self.filemenubar , tearoff=False)
        self.filemenu.add_command(label='開く' , command=self.on_open_dir)
        self.filemenubar.add_cascade(label='読み込み' , menu=self.filemenu)
        self.win2.config(menu=self.filemenubar)

        # Create Menubar
        self.menubar = tk.Menu(self)
        # Create Menu
        self.filemenu = tk.Menu(self.menubar , tearoff=0)
        self.filemenu.add_command(label="開く" , command=self.on_open_file)
        self.filemenu.add_command(label="保存" , command=self.on_save_file)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="終了" , command=self.quit)
        self.menubar.add_cascade(label='ファイル' , menu=self.filemenu)

        self.blurmenu = tk.Menu(self.menubar , tearoff=0)
        self.blurmenu.add_command(label='Gaussian Blur', command=self.on_gaussian_blur)
        self.blurmenu.add_command(label='モザイク' , command=self.on_mosaic)
        self.blurmenu.add_command(label='鉛筆画' , command=self.on_pencil)
        self.blurmenu.add_command(label='拡大・縮小' , command=self.on_resize)
        self.blurmenu.add_command(label='色彩変換' , command=self.on_change)
        self.blurmenu.add_command(label='ガンマ補正', command=self.on_gamma)
        self.blurmenu.add_command(label='グレースケール変換' , command=self.on_gray_scale)
        self.blurmenu.add_command(label='セピア色変換' , command=self.on_sepia)
        self.blurmenu.add_command(label='ミラー処理' , command=self.on_mirror)
        self.menubar.add_cascade(label='加工' , menu=self.blurmenu)
        
        
        # Create Edit Menu
        self.editmenu = tk.Menu(self.menubar , tearoff=0)
        self.editmenu.add_command(label='貼り付け' , command=self.on_paste)
        self.editmenu.add_command(label='コピー', command=self.copy_to_clipboard)
        self.editmenu.add_command(label='戻す' , command=self.undo) 
        self.menubar.add_cascade(label='編集' , menu=self.editmenu)

        self.config(menu=self.menubar)

        self.label = ttk.Label(self)
        self.label.pack()

        self.image_path = ''
        self.directory=None # For Win2

        # PIL image object
        self.image = None # PhotoImage
        self.original = None
        self.popup = None
        self.width = 800
        self.height = 800

        self.image_arr = None
        self.kernel_value = 5
        self.sigma_value = 13

        # Pop-up Window
        self.bind('<Button-3>' , self.show_menu)

        # For Win2
        self.filelist.bind('<Button-1>' , self.on_draw)
        self.win2.bind('<Button-3>' , self.on_rename)
        self.win2.bind('<Down>' , self.on_down)
        self.win2.bind('<Up>' , self.on_up)
        self.win2.bind('<Delete>' , self.on_delete)

        # Create Signal Object For Sub-Thread
        self.watcher = None
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Sub Window for Gannma_Correction
        self.gamma = None

    def on_mirror(self , event=None):
        if self.image:
            cv2_img = np.array(ImageTk.getimage(self.image))
            cv2_img = cv2.flip(cv2_img , 1)
            self.image = ImageTk.PhotoImage(Image.fromarray(cv2_img))
            self.label.config(image=self.image)
        else:
            self.label.config(text='画像を表示してください')

    def on_sepia(self , event=None):
        if self.image:
            img_pil = ImageTk.getimage(self.image)
            if img_pil.mode == 'RGBA':
                img_pil_rgb = img_pil.convert('RGB')
                sepia_image = sepia(img_pil_rgb)
            else:
                sepia_image = sepia(img_pil)
            self.image = ImageTk.PhotoImage(Image.fromarray(sepia_image))
            self.label.config(image=self.image)
        else:
            self.label.config(text='画像を表示してください')

    def on_gray_scale(self , event=None):
        if self.image:
            cv2_image = np.array(ImageTk.getimage(self.image))
            cv2_image = cv2.cvtColor(cv2_image , cv2.COLOR_RGB2GRAY)
            self.image = ImageTk.PhotoImage(Image.fromarray(cv2_image))
            self.label.config(image=self.image)
        else:
            self.label.config('画像が表示されていません')

    def on_gamma(self , event=None):
        if not self.gamma:
            self.gamma = tk.Toplevel()
            self.gamma.title('ガンマ補正')
            self.gamma.geometry(f'+{self.winfo_x()+20}+{self.winfo_y()+20}')
            self.gamma_var = tk.DoubleVar()
            self.gamma_var.set(22)
            self.label_for_gamma = ttk.Label(self.gamma , text='補正値')
            self.scale_for_gamma = ttk.Scale(self.gamma , 
                                             from_=1 , 
                                             to=50 , 
                                             length=200 , 
                                             variable=self.gamma_var , 
                                             orient=tk.HORIZONTAL , 
                                             command=self.update_gamma_scale
                                         )
            self.update_gamma_scale('_')
            self.button_for_gamma = ttk.Button(self.gamma , text='＞' , command=self.on_exec_gamma)
            self.scale_for_gamma.grid(row=0 , column=0 , pady=10)
            self.label_for_gamma.grid(row=0 , column=1, padx=10 , pady=10)
            self.button_for_gamma.grid(row=1 , column=1 , padx=10 , pady=10)
    
    def update_gamma_scale(self , value):
        val = self.gamma_var.get()/10
        self.label_for_gamma.config(text=f'補正値:{val:.2f}')

    def on_exec_gamma(self , event=None):
        gamma = self.gamma_var.get()/10.0
        logger.debug('ガンマ補正実行 event:%s gamma_var:%s' , event , gamma)
        inv_gamma = 1.0 / gamma
        image_cv2 = np.array(ImageTk.getimage(self.image))
        corrected_image = np.power(image_cv2 / 255.0 , inv_gamma) *255.0
        corrected_image = np.clip(corrected_image , 0 , 255).astype(np.uint8)
        self.image = ImageTk.PhotoImage(Image.fromarray(corrected_image))
        self.label.config(image=self.image)
        self.gamma.destroy()
        self.gamma = False

    def close_Handler(self , event=None):
        messagebox.showinfo('確認', '閉じるボタンは無効です')
        return

    def signal_handler(self , signal , frame):
        if self.watcher:
            self.watcher.stop()
            self.watcher.join()
        exit()

    def on_delete(self, event=None):
        index = self.filelist.curselection()
        if index:
            index = index[0]
            selected_file = self.filelist.get(index)
            os.remove(os.path.join(self.directory, selected_file))
            self.filelist.delete(index)
            if index == self.filelist.size():
                # 最後の項目を削除した場合、選択を前の項目に変更
                self.filelist.selection_set(index - 1)
                self.filelist.activate(index - 1)
            else:
                self.filelist.selection_set(index)
                self.filelist.activate(index)
            self.on_draw('_')
        else:
            messagebox.showerror('エラー', 'ファイルの消去に失敗しました')


    def on_up(self, event=None):
        selected_index = self.filelist.curselection()
        if selected_index:
            active_index = selected_index[0]
            if active_index >= 0:
                self.filelist.activate(active_index)
                self.filelist.selection_clear(0,tk.END)
                self.filelist.selection_set(active_index)
                self.on_draw('_')
        else:
            return

    def on_down(self, event=None):
        selected_index = self.filelist.curselection()
        if selected_index:
            active_index = selected_index[0]
            if active_index < self.filelist.size():
                self.filelist.selection_clear(selected_index)
                self.filelist.selection_set(active_index)
                self.filelist.activate(active_index)
                self.on_draw('_')
        else:
            return

    def on_open_dir(self , event=None):
        self.directory = filedialog.askdirectory()
        if self.directory:
            if self.watcher:
                self.watcher.stop()
            self.watcher = DirectoryWatcher(directory=self.directory , filelist=self.filelist)
            self.watcher.daemon = True
            self.watcher.start()
            self.filelist.delete(0 , tk.END)
            file_list = os.listdir(self.directory)
            for filename in file_list:
                if os.path.splitext(filename)[1] not in ('.jpeg' , '.jpg' , '.png'):
                    continue
                self.filelist.insert(tk.END , filename)

    def on_rename(self , event=None):
        index = self.filelist.curselection()
        if index:
            item = self.filelist.get(index)

            new_name = simpledialog.askstring("名前の変更" , "ファイル名を入力してください" , initialvalue=item)
            if new_name:
                try:
                    os.rename(os.path.join(self.directory , item) , os.path.join(self.directory , new_name))
                    self.filelist.delete(index)
                    self.filelist.insert(index , new_name)
                    messagebox.showinfo('確認','ファイル名を変更しました')
                except OSError:
                    messagebox.showerror('エラー','ファイル名の変更に失敗しました')


    def on_draw(self , event):
        index = self.filelist.curselection()
        if index:
            target_file = self.filelist.get(index)
            full_path = os.path.join(self.directory , target_file)
            self.image = self.original = ImageTk.PhotoImage(Image.open(full_path))
            if self.image:
                self.width , self.height = self.image.width() , self.image.height()
                self.label.config(image=self.image)
                self.wm_geometry(f'{self.width}x{self.height}')
            else:
                self.label.config(text='画像ファイルが読み込めませんでした')
        

    def on_open_file(self , event=None):
        filetype =[('JPEG files' , '*.jpg;*.jpeg') , ('PNG files' , '*.png')]
        initialdir = os.getcwd()
        # prev_image_path = self.image_path
        self.image_path = filedialog.askopenfilename(filetypes=filetype , initialdir=initialdir)
        if self.image_path:
            self.image = Image.open(self.image_path)
            self.width , self.height = self.image.size
            self.image = self.original = ImageTk.PhotoImage(self.image)
            self.label.config(image=self.image)
            self.wm_geometry(str(self.width) + 'x' + str(self.height))

    def on_save_file(self , event=None):
        self.image_path = filedialog.asksaveasfilename(defaultextension='.jpg')
        if self.image_path:
            if self.image:
                name = tk_to_pil(self.image)
                name.save(self.image_path)
            else:
                self.label.config(text='画像が表示されていません')

    def on_gaussian_blur(self , event=None):
        if self.image:
            pil_image = tk_to_pil(self.image) 
            self.image_arr = np.array(pil_image)
            self.image_arr = cv2.cvtColor(self.image_arr , cv2.COLOR_RGB2BGR)
            self.popup_gaussian()
        else:
            self.label.config(text='画像を表示させてください')
            
    def on_mosaic(self , enent=None):
        global top_window
        top_window = tk.Toplevel()
        top_window.title('モザイク設定')
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
        self.undo_button = ttk.Button(self.frame_mosaic , text='元に戻す' , command=self.undo)
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
            self.label.config(image=self.image)
        else:
            self.label.config(text='画像を表示してください')

    
    def undo(self , event=None):
        if not self.original:
            self.label.config(text='画像データがありません')
            return
        image = tk_to_cv2(self.original)
        h , w , _ = image.shape
        image = cv2.resize(image , dsize=(w , h) , interpolation=cv2.INTER_LANCZOS4)
        image = cv2_to_pil(image)
        self.image = ImageTk.PhotoImage(image) 
        self.label.config(image=self.image)
        self.wm_geometry(str(w)+'x'+str(h))
    
    def update_mosaic_slidar(self , event):
        var = self.slidar_var.get()
        self.mosaic_slidar_label.config(text=f'強さ{var:.2f}')
        
    def on_exec_mosaic(self,event=None):
        if not self.image: #PhotoImage
            self.mosaic_slidar_label.config(text='画像を表示させてください')
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
        self.label.config(image=self.image)
        top_window.destroy()

    def popup_gaussian(self):
        top_window = tk.Toplevel(self)
        top_window.title('設定')
        label1 = ttk.Label(top_window , text='カーネルサイズ')
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
        button = ttk.Button(top_window , text='変換' , command=self.on_Gaussian)
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
            self.label.config(image=self.image)
        else:
            self.label.config(text="画像が取得できません")

    def show_menu(self , event=None):
        self.editmenu.post(event.x_root , event.y_root)

    def on_paste(self , event=None):
        self.image = ImageGrab.grabclipboard()
        if isinstance(self.image , Image.Image):
            self.image = self.image.convert('RGB')
            self.width , self.height = self.image.size
            self.image = ImageTk.PhotoImage(self.image)
            if self.image:
                self.original = self.image
                self.label.config(image=self.image)
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
                self.label.config(image=self.image)
                self.wm_geometry(str(self.width) + 'x' + str(self.height))
        elif isinstance(self.image , list):
            img = Image.open(self.image[0])
            self.width , self.height = img.size
            self.image = ImageTk.PhotoImage(img)
            if self.image:
                self.original = self.image
                self.label.config(image=self.image)
                self.wm_geometry(str(self.width) + 'x' + str(self.height))
        else:
            self.label.config(text='画像をコピーしてきてください')

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
            self.label.config(text='画像を表示させてください')

    def on_resize(self , event=None):
        global top_window
        top_window = tk.Toplevel()
        top_window.title('拡大・縮小')
        self.resize_label = ttk.Label(top_window , text='倍率')
        self.var = tk.DoubleVar()
        self.var.set(11.0)
        self.resize_slidar = ttk.Scale(top_window , from_=1 , to=30 , length=200 , variable=self.var , command=self.update_resize_scale)
        self.resize_button = ttk.Button(top_window , text='＞' , command=self.on_exec_resize)
        self.resize_slidar.grid(row=0 , column=0 , pady=10)
        self.resize_label.grid(row=0 , column=1 , padx=10 ,  pady=10)
        self.resize_button.grid(row=1, column=1 , padx=10 , pady=10)

    def update_resize_scale(self , value):
        var = self.var.get()/10
        self.resize_label.config(text=f'倍率:{var:.2f}')

    def on_exec_resize(self , event=None):
        if self.image:
            var = int(self.var.get())
            arr_img = np.array(ImageTk.getimage(self.image))
            rate = var / 10
            arr_img = cv2.resize(arr_img , dsize=None , fx=rate , fy=rate ,interpolation=cv2.INTER_LANCZOS4)
            height , width , _ = arr_img.shape
            self.image = ImageTk.PhotoImage(Image.fromarray(arr_img))
            self.label.config(image=self.image)
            self.wm_geometry(str(width) +'x' + str(height))
            top_window.destroy()
        else:
            self.label.config(text='画像を表示してください')
            top_window.destroy()
    
    def on_change(self,event=None):
        self.popup = tk.Toplevel()
        self.popup.title('チャンネル値変更')
        self.popup.geometry(f'+{self.winfo_x()+20}+{self.winfo_y()-20}')
        self.label1 = ttk.Label(self.popup,text='青')
        self.label2 = ttk.Label(self.popup,text='緑')
        self.label3 = ttk.Label(self.popup,text='赤')
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
        self.label1.config(text=f'青:{val:.1f}')
        
    def update_green(self , value):
        val = self.var_G.get()/10
        self.label2.config(text=f'緑:{val:.1f}')
        
    def update_red(self , value):
        val = self.var_R.get()/10
        self.label3.config(text=f'赤:{val:.1f}')
    
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
            self.label.config(image=self.image)
            self.popup.destroy()
        else:
            self.label.config(text='画像を表示させてください')
            
    def quit(self):
        self.destroy()

    def run(self):
        self.mainloop()

app = App()
app.run()