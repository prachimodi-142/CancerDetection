import wx
import os
import webbrowser
import wx.lib.agw.multidirdialog as MDD

# https://realpython.com/python-gui-with-wxpython/
# https://www.youtube.com/watch?v=tNfcvgPKgyU&t=812s

# How do you view the entire image?
# How do you index into the image?
# Pre-processing for the image before feeding it to the model

class MyFrame(wx.Frame):    
    def __init__(self):
        super().__init__(parent=None, title='Cancer Metastasis Classifier')
        panel = wx.Panel(self)
        self.currentDirectory = os.getcwd()
        print(self.currentDirectory)
        
        my_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.text_ctrl = wx.TextCtrl(panel, pos=(5, 5))
        my_sizer.Add(self.text_ctrl, 0, wx.ALL | wx.EXPAND, 5)
        
        my_btn = wx.Button(panel, label='Enter File path', pos=(5, 55))
        my_sizer.Add(my_btn, 0, wx.ALL | wx.CENTER, 5)
        my_btn.Bind(wx.EVT_BUTTON, self.on_press)
        panel.SetSizer(my_sizer)
        
        multiDirDlgBtn = wx.Button(panel, label="Browse Files")
        my_sizer.Add(multiDirDlgBtn, 0, wx.ALL | wx.CENTER, 5)
        multiDirDlgBtn.Bind(wx.EVT_BUTTON, self.onMultiDir)
        panel.SetSizer(my_sizer)

        self.Show()
        
    def on_press(self, event):
        value = self.text_ctrl.GetValue()
        if not value:
            print("You didn't enter anything!")
        else:
            # img_Path = r'{}'.format(str(value))
            img_Path = str(value)
            print("Path provided is: ", img_Path)
            os.system("python ./deepzoom/deepzoom_multiserver.py" + " -Q 100 " + str(img_Path))
            webbrowser.open('http://127.0.0.1:5000')
    
    def onMultiDir(self, event):
        """
        Create and show the MultiDirDialog
        """
        dlg = MDD.MultiDirDialog(self, title="Choose a directory:",
                                 defaultPath=self.currentDirectory,
                                 agwStyle=0)
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
            print("You chose the following file(s):")
            for path in paths:
                print(path)
                if path == self.currentDirectory:
                    cmd = "python ./deepzoom/deepzoom_multiserver.py" + " -Q 100 " + "./"
                    webbrowser.open('http://127.0.0.1:5000')
                else:
                    path = path.replace("\\", "/")
                    cmd = "python ./deepzoom/deepzoom_multiserver.py" + " -Q 100 " + str(path)
                    webbrowser.open('http://127.0.0.1:5000')
                print(cmd)
                os.system(cmd)
        dlg.Destroy()

app = wx.App()
frame = MyFrame()
frame.Show()
app.MainLoop()

