"""
gui_layout.py
==============

Define la interfaz gráfica del recomendador musical
utilizando Tkinter.

Este módulo SOLO se encarga de la presentación.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from src.gui.gui_commands import (
    recommend_song_command,
    get_random_song
)


# =========================
# Ventana principal
# =========================

class RecommenderGUI(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Recomendador Musical")
        self.geometry("600x400")
        self.resizable(False, False)

        self.create_widgets()

    # -------------------------

    def create_widgets(self):
        title = ttk.Label(
            self,
            text="Recomendador Musical",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=10)

        self.entry = ttk.Entry(self, width=50)
        self.entry.pack(pady=5)

        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)

        recommend_btn = ttk.Button(
            button_frame,
            text="Recomendar",
            command=self.on_recommend
        )
        recommend_btn.grid(row=0, column=0, padx=5)

        random_btn = ttk.Button(
            button_frame,
            text="Canción aleatoria",
            command=self.on_random
        )
        random_btn.grid(row=0, column=1, padx=5)

        self.result_box = tk.Text(self, height=12, width=70)
        self.result_box.pack(pady=10)
        self.result_box.config(state=tk.DISABLED)

    # -------------------------

    def on_recommend(self):
        song_name = self.entry.get().strip()

        if not song_name:
            messagebox.showwarning(
                "Entrada vacía",
                "Por favor, introduce una canción."
            )
            return

        self.show_message("Buscando recomendaciones...")

        recommend_song_command(
            song_name,
            self.update_results
        )

    # -------------------------

    def on_random(self):
        random_song = get_random_song()
        self.entry.delete(0, tk.END)
        self.entry.insert(0, random_song)
        self.on_recommend()

    # -------------------------

    def update_results(self, song_name, result):
        self.result_box.config(state=tk.NORMAL)
        self.result_box.delete("1.0", tk.END)

        if song_name is None:
            self.result_box.insert(tk.END, f"Error: {result}")
        else:
            self.result_box.insert(
                tk.END,
                f"Canción usada: {song_name}\n\n"
            )

            for _, row in result.iterrows():
                self.result_box.insert(
                    tk.END,
                    f"- {row['track_name']} | {row['artists']} | {row['track_genre']}\n"
                )

        self.result_box.config(state=tk.DISABLED)

    # -------------------------

    def show_message(self, text):
        self.result_box.config(state=tk.NORMAL)
        self.result_box.delete("1.0", tk.END)
        self.result_box.insert(tk.END, text)
        self.result_box.config(state=tk.DISABLED)


# =========================
# Ejecución
# =========================

if __name__ == "__main__":
    app = RecommenderGUI()
    app.mainloop()