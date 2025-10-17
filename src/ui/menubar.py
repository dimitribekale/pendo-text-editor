import tkinter as tk

class MenuBar(tk.Menu):
    def __init__(self, master, commands, theme):
        super().__init__(master, bg=theme["bg"], fg=theme["fg"], activebackground=theme["select_bg"], activeforeground=theme["fg"])
        self.commands = commands
        self.theme = theme
        self._create_file_menu()
        self._create_edit_menu()

    def _create_file_menu(self):
        file_menu = tk.Menu(self, tearoff=0, bg=self.theme["widget_bg"], fg=self.theme["fg"], activebackground=self.theme["select_bg"], activeforeground=self.theme["fg"])
        self.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.commands["new_file"])
        file_menu.add_command(label="Open", command=self.commands["open_file"])
        file_menu.add_command(label="Save", command=self.commands["save_file"])
        file_menu.add_command(label="Save As", command=self.commands["save_as_file"])
        file_menu.add_separator()
        file_menu.add_command(label="Close Tab", command=self.commands["close_tab"])
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.commands["exit"])

    def _create_edit_menu(self):
        edit_menu = tk.Menu(self, tearoff=0, bg=self.theme["widget_bg"], fg=self.theme["fg"], activebackground=self.theme["select_bg"], activeforeground=self.theme["fg"])
        self.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Settings...", command=self.commands["open_settings"])

