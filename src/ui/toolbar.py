import tkinter as tk
from tkinter import ttk

class Toolbar(ttk.Frame):
    def __init__(self, master, commands):
        super().__init__(master)
        self.commands = commands


