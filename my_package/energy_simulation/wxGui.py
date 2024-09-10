import wx
from pyfmi import load_fmu


class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="FMU Model GUI")
        self.frame.Show()
        return True


class MyFrame(wx.Frame):
    def __init__(self, *args, **kw):
        super(MyFrame, self).__init__(*args, **kw)

        # Panel
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # FMU File Input
        self.fmu_file_picker = wx.FilePickerCtrl(panel, message="Select FMU File")
        vbox.Add(self.fmu_file_picker, flag=wx.EXPAND | wx.ALL, border=10)

        # Load Button
        self.load_button = wx.Button(panel, label="Load FMU")
        vbox.Add(self.load_button, flag=wx.EXPAND | wx.ALL, border=10)
        self.load_button.Bind(wx.EVT_BUTTON, self.OnLoadFMU)

        # Variables List
        self.var_list = wx.ListBox(panel)
        vbox.Add(self.var_list, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        # Variable Value Input
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.var_name_text = wx.TextCtrl(panel, size=(200, -1))
        self.var_value_text = wx.TextCtrl(panel, size=(200, -1))
        hbox.Add(wx.StaticText(panel, label="Variable Name:"), flag=wx.ALIGN_CENTER_VERTICAL)
        hbox.Add(self.var_name_text, flag=wx.LEFT, border=10)
        hbox.Add(wx.StaticText(panel, label="Value:"), flag=wx.ALIGN_CENTER_VERTICAL | wx.LEFT, border=10)
        hbox.Add(self.var_value_text, flag=wx.LEFT, border=10)
        vbox.Add(hbox, flag=wx.EXPAND | wx.ALL, border=10)

        # Set Value Button
        self.set_value_button = wx.Button(panel, label="Set Value")
        vbox.Add(self.set_value_button, flag=wx.EXPAND | wx.ALL, border=10)
        self.set_value_button.Bind(wx.EVT_BUTTON, self.OnSetValue)

        panel.SetSizer(vbox)

    def OnLoadFMU(self, event):
        fmu_path = self.fmu_file_picker.GetPath()
        if not fmu_path:
            wx.MessageBox("Please select an FMU file.", "Error", wx.OK | wx.ICON_ERROR)
            return

        try:
            self.model = load_fmu(fmu_path)
            self.var_list.Clear()
            variables = self.model.get_variable_names()
            self.var_list.AppendItems(variables)
        except Exception as e:
            wx.MessageBox(f"Error loading FMU: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def OnSetValue(self, event):
        var_name = self.var_name_text.GetValue()
        var_value = self.var_value_text.GetValue()

        if not var_name:
            wx.MessageBox("Please enter a variable name.", "Error", wx.OK | wx.ICON_ERROR)
            return

        try:
            value = float(var_value)  # Assuming the value should be a float
            self.model.set(var_name, value)
            wx.MessageBox(f"Set {var_name} to {value}", "Info", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.MessageBox(f"Error setting value: {e}", "Error", wx.OK | wx.ICON_ERROR)


if __name__ == "__main__":
    app = MyApp()
    app.MainLoop()
