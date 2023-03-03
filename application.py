import wx
import viewer

# https://realpython.com/python-gui-with-wxpython/
# https://www.youtube.com/watch?v=tNfcvgPKgyU&t=812s

# How do you view the entire image?
# How do you index into the image?
# Pre-processing for the image before feeding it to the model

class MyFrame(wx.Frame):    
    def __init__(self):
        super().__init__(parent=None, title='Cancer Metastasis Classifier')
        panel = wx.Panel(self)
        
        my_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.text_ctrl = wx.TextCtrl(panel, pos=(5, 5))
        my_sizer.Add(self.text_ctrl, 0, wx.ALL | wx.EXPAND, 5)
        
        my_btn = wx.Button(panel, label='Enter Tiff File path', pos=(5, 55))
        my_sizer.Add(my_btn, 0, wx.ALL | wx.CENTER, 5)
        my_btn.Bind(wx.EVT_BUTTON, self.on_press)
        panel.SetSizer(my_sizer)

        self.Show()
        
    def on_press(self, event):
        value = self.text_ctrl.GetValue()
        if not value:
            print("You didn't enter anything!")
        else:
            img_Path = r'{}'.format(str(value))
            print("Path provided is: ", img_Path)
            viewer.show_image(img_Path)

app = wx.App()
frame = MyFrame()
frame.Show()
app.MainLoop()

