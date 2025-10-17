import tkinter as tk
import tkinter.font

class SuggestionBox(tk.Toplevel):
    def __init__(self, master, select_callback, text_widget):
        super().__init__(master)
        self.overrideredirect(True)
        self.withdraw()
        self.select_callback = select_callback
        self.text_widget = text_widget

        self.listbox = tk.Listbox(self, exportselection=False)
        self.listbox.pack(fill=tk.BOTH, expand=True)

        self.listbox.bind("<Return>", lambda e: self.select_callback())
        self.listbox.bind("<Tab>", lambda e: self.select_callback())
        self.listbox.bind("<Up>", lambda e: self.move_selection(-1))
        self.listbox.bind("<Down>", lambda e: self.move_selection(1))
        self.listbox.bind("<ButtonRelease-1>", lambda e: self.select_callback())

    def show(self, x, y, suggestions):
        if not suggestions:
            self.hide()
            return

        self.listbox.delete(0, tk.END)
        for s in suggestions:
            self.listbox.insert(tk.END, s)
        
        self.listbox.selection_set(0)

        # Calculate required size
        font_obj = tkinter.font.Font(font=self.listbox.cget("font"))
        max_width = 0
        for s in suggestions:
            width = font_obj.measure(s)
            if width > max_width:
                max_width = width
        
        # Add some padding
        width_padding = 20 # px
        height_padding = 5 # px
        calculated_width = max_width + width_padding
        calculated_height = len(suggestions) * font_obj.metrics("linespace") + height_padding

        self.geometry(f"{int(calculated_width)}x{int(calculated_height)}+{x}+{y}")
        self.deiconify()
        self.lift()
        self.listbox.focus_force()

    def hide(self):
        self.withdraw()
        self.text_widget.focus_force()

    def is_active(self):
        return self.state() == "normal"

    def move_selection(self, delta):
        if not self.is_active():
            return
        
        current_selection = self.listbox.curselection()
        if not current_selection:
            return

        next_selection = current_selection[0] + delta
        if 0 <= next_selection < self.listbox.size():
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(next_selection)
            self.listbox.activate(next_selection)

    def get_selection(self):
        if not self.is_active() or not self.listbox.curselection():
            return None
        return self.listbox.get(self.listbox.curselection()[0])
