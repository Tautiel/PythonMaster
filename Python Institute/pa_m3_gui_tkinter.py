#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PYTHON PROFESSIONAL 1 - MODULE 3                          ║
║                    GUI PROGRAMMING WITH TKINTER                               ║
║                    PCPP1-32-101 Section 3: 20% (8 domande) - CRITICAL!       ║
╚══════════════════════════════════════════════════════════════════════════════╝

SYLLABUS:
├── PCPP1 3.1 - GUI concepts: widgets, event-driven programming
├── PCPP1 3.2 - Tkinter basics: Tk(), mainloop(), widgets, place(), grid()
└── PCPP1 3.3 - Events, callbacks, binding, Canvas
"""

# ══════════════════════════════════════════════════════════════════════════════
# 3.1 GUI CONCEPTS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("3.1 GUI CONCEPTS")
print("=" * 70)

print("""
📋 GUI (Graphical User Interface)
   - Interfaccia visuale per interazione utente
   - Windows, buttons, text fields, menus, etc.
   
📋 WIDGET (Window Gadget)
   - Componente visuale: Button, Label, Entry, Frame, etc.
   - Ogni widget ha proprietà e metodi
   
📋 EVENT-DRIVEN PROGRAMMING
   - Il programma risponde a EVENTI (click, keypress, etc.)
   - Loop principale (mainloop) attende eventi
   - CALLBACK: funzione chiamata quando evento accade
   
📋 GUI TOOLKITS
   - Tkinter (standard Python, in esame!)
   - PyQt, PySide, wxPython (alternative)
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.2 TKINTER BASICS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.2 TKINTER BASICS")
print("=" * 70)

import tkinter as tk
from tkinter import messagebox

print("""
📋 STRUTTURA BASE:

```python
import tkinter as tk

# 1. Crea finestra principale
root = tk.Tk()
root.title("My App")
root.geometry("400x300")  # larghezza x altezza

# 2. Aggiungi widgets
label = tk.Label(root, text="Hello!")
label.pack()

# 3. Avvia main loop
root.mainloop()  # Blocca qui finché finestra aperta
```
""")

# Demo code (non eseguiamo mainloop qui)
print("\n📐 CREAZIONE FINESTRA:")
print("""
root = tk.Tk()                    # Finestra principale
root.title("My Application")      # Titolo finestra
root.geometry("400x300")          # Dimensioni (WxH)
root.geometry("+100+100")         # Posizione (+X+Y)
root.resizable(True, True)        # Ridimensionabile (W, H)
root.minsize(200, 150)            # Dimensione minima
root.maxsize(800, 600)            # Dimensione massima
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.3 WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.3 COMMON WIDGETS")
print("=" * 70)

print("""
┌─────────────────┬───────────────────────────────────────────────────┐
│ Widget          │ Descrizione                                       │
├─────────────────┼───────────────────────────────────────────────────┤
│ Label           │ Testo statico                                     │
│ Button          │ Pulsante cliccabile                               │
│ Entry           │ Campo input singola riga                          │
│ Text            │ Campo input multi-riga                            │
│ Frame           │ Container per altri widget                        │
│ Canvas          │ Area per disegno                                  │
│ Checkbutton     │ Checkbox                                          │
│ Radiobutton     │ Radio button (selezione singola)                  │
│ Listbox         │ Lista di elementi                                 │
│ Scrollbar       │ Barra di scorrimento                              │
│ Scale           │ Slider                                            │
│ Spinbox         │ Input numerico con frecce                         │
│ Menu            │ Menu dropdown                                     │
│ Toplevel        │ Finestra secondaria                               │
└─────────────────┴───────────────────────────────────────────────────┘
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.4 LABEL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.4 LABEL WIDGET")
print("=" * 70)

print("""
# Creazione Label
label = tk.Label(root, text="Hello World")

# Proprietà comuni:
label = tk.Label(root,
    text="Hello",           # Testo
    font=("Arial", 16),     # Font (nome, size)
    fg="blue",              # Foreground (testo)
    bg="white",             # Background
    width=20,               # Larghezza in caratteri
    height=2,               # Altezza in righe
    anchor="w",             # Allineamento: n,s,e,w,center
    justify="left",         # Giustificazione testo
    padx=10,                # Padding orizzontale
    pady=5                  # Padding verticale
)

# Aggiornare testo dinamicamente:
label.config(text="New text")
# oppure
label["text"] = "New text"
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.5 BUTTON
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.5 BUTTON WIDGET")
print("=" * 70)

print("""
# Callback function
def on_click():
    print("Button clicked!")

# Creazione Button
button = tk.Button(root,
    text="Click Me",
    command=on_click,       # CALLBACK - funzione da chiamare
    width=10,
    height=2,
    fg="white",
    bg="blue",
    activebackground="red", # Colore quando premuto
    state="normal"          # "normal", "disabled"
)

# Disabilitare button
button.config(state="disabled")
button["state"] = "normal"
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.6 ENTRY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.6 ENTRY WIDGET")
print("=" * 70)

print("""
# Input a singola riga
entry = tk.Entry(root,
    width=30,
    font=("Arial", 12),
    show="*"               # Per password (mostra * invece di caratteri)
)

# Ottenere il valore
text = entry.get()

# Impostare il valore
entry.delete(0, tk.END)    # Cancella tutto
entry.insert(0, "Default") # Inserisci testo

# StringVar per binding bidirezionale
text_var = tk.StringVar()
text_var.set("Initial value")
entry = tk.Entry(root, textvariable=text_var)

# Ora puoi usare:
current_text = text_var.get()
text_var.set("New value")
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.7 GEOMETRY MANAGERS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.7 GEOMETRY MANAGERS (ESAME!)")
print("=" * 70)

print("""
📋 3 GEOMETRY MANAGERS:

1. pack() - Impila widgets
   widget.pack(side="top")     # top, bottom, left, right
   widget.pack(fill="x")       # x, y, both
   widget.pack(expand=True)    # Espande per riempire spazio

2. grid() - Griglia righe/colonne
   widget.grid(row=0, column=0)
   widget.grid(rowspan=2)      # Occupa 2 righe
   widget.grid(columnspan=3)   # Occupa 3 colonne
   widget.grid(sticky="nsew")  # Allineamento: n,s,e,w

3. place() - Posizionamento assoluto/relativo
   widget.place(x=100, y=50)           # Pixel assoluti
   widget.place(relx=0.5, rely=0.5)    # Relativo (0-1)
   widget.place(anchor="center")       # Punto di ancoraggio

⚠️ NON mischiare pack() e grid() nello stesso container!
""")

print("""
📐 ESEMPIO GRID:

```python
# Layout form
tk.Label(root, text="Name:").grid(row=0, column=0, sticky="e")
tk.Entry(root).grid(row=0, column=1)

tk.Label(root, text="Email:").grid(row=1, column=0, sticky="e")
tk.Entry(root).grid(row=1, column=1)

tk.Button(root, text="Submit").grid(row=2, column=0, columnspan=2)
```
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.8 EVENT HANDLING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.8 EVENT HANDLING (ESAME!)")
print("=" * 70)

print("""
📋 EVENTS:
   - <Button-1>    Left mouse click
   - <Button-2>    Middle mouse click
   - <Button-3>    Right mouse click
   - <Double-1>    Double left click
   - <Enter>       Mouse enters widget
   - <Leave>       Mouse leaves widget
   - <Key>         Any key press
   - <Return>      Enter key
   - <Escape>      Escape key
   - <Up>, <Down>  Arrow keys

📋 BINDING:

```python
def on_click(event):
    print(f"Clicked at ({event.x}, {event.y})")

# Bind a un widget
widget.bind("<Button-1>", on_click)

# Bind globale
root.bind("<Return>", on_submit)

# Unbind
widget.unbind("<Button-1>")
```

📋 EVENT OBJECT:
   event.x, event.y    # Coordinate mouse
   event.widget        # Widget che ha generato evento
   event.char          # Carattere premuto
   event.keysym        # Nome tasto (es. "Return", "space")
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.9 CANVAS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.9 CANVAS WIDGET")
print("=" * 70)

print("""
# Canvas per disegno
canvas = tk.Canvas(root, width=400, height=300, bg="white")
canvas.pack()

# Disegnare forme:
canvas.create_line(0, 0, 100, 100, fill="black", width=2)
canvas.create_rectangle(50, 50, 150, 100, fill="blue", outline="red")
canvas.create_oval(100, 100, 200, 150, fill="green")
canvas.create_arc(200, 200, 300, 300, start=0, extent=90)
canvas.create_polygon(50, 250, 100, 200, 150, 250, fill="yellow")
canvas.create_text(200, 50, text="Hello", font=("Arial", 20))

# Ogni metodo create_* restituisce un ID
rect_id = canvas.create_rectangle(10, 10, 50, 50)

# Modificare elementi
canvas.itemconfig(rect_id, fill="red")
canvas.move(rect_id, 10, 10)  # Sposta di (dx, dy)
canvas.delete(rect_id)        # Elimina
canvas.delete("all")          # Elimina tutto
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.10 RADIOBUTTON & CHECKBUTTON
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.10 RADIOBUTTON & CHECKBUTTON")
print("=" * 70)

print("""
# RADIOBUTTON - Selezione singola (mutualmente esclusivi)
selected = tk.StringVar(value="option1")

rb1 = tk.Radiobutton(root, text="Option 1", 
                      variable=selected, value="option1")
rb2 = tk.Radiobutton(root, text="Option 2", 
                      variable=selected, value="option2")

# Valore selezionato: selected.get()

# CHECKBUTTON - Selezione multipla
var1 = tk.IntVar()  # 0 o 1
var2 = tk.IntVar()

cb1 = tk.Checkbutton(root, text="Option A", variable=var1)
cb2 = tk.Checkbutton(root, text="Option B", variable=var2)

# var1.get() restituisce 1 se checked, 0 altrimenti
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.11 DIALOG BOXES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.11 DIALOG BOXES")
print("=" * 70)

print("""
from tkinter import messagebox

# Message boxes
messagebox.showinfo("Title", "Information message")
messagebox.showwarning("Title", "Warning message")
messagebox.showerror("Title", "Error message")

# Question boxes (restituiscono risposta)
result = messagebox.askyesno("Title", "Do you want to continue?")
# result = True/False

result = messagebox.askokcancel("Title", "Proceed?")
# result = True/False

result = messagebox.askquestion("Title", "Are you sure?")
# result = "yes"/"no"

# File dialogs
from tkinter import filedialog

filename = filedialog.askopenfilename(
    title="Select file",
    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
)

filename = filedialog.asksaveasfilename(
    defaultextension=".txt"
)
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.12 OBSERVABLE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.12 OBSERVABLE VARIABLES")
print("=" * 70)

print("""
📋 VARIABLE TYPES:
   tk.StringVar()    # Per stringhe
   tk.IntVar()       # Per interi
   tk.DoubleVar()    # Per float
   tk.BooleanVar()   # Per booleani

📋 METHODS:
   var.get()         # Ottieni valore
   var.set(value)    # Imposta valore
   var.trace_add("write", callback)  # Observer quando cambia

# Esempio trace:
def on_change(*args):
    print(f"Value changed to: {text_var.get()}")

text_var = tk.StringVar()
text_var.trace_add("write", on_change)

entry = tk.Entry(root, textvariable=text_var)
# Ogni modifica nell'entry chiamerà on_change
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.13 COMPLETE EXAMPLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("3.13 COMPLETE EXAMPLE")
print("=" * 70)

print("""
import tkinter as tk
from tkinter import messagebox

def on_submit():
    name = name_entry.get()
    if name:
        messagebox.showinfo("Hello", f"Hello, {name}!")
    else:
        messagebox.showwarning("Warning", "Please enter a name")

root = tk.Tk()
root.title("Simple Form")
root.geometry("300x150")

# Label
tk.Label(root, text="Enter your name:").pack(pady=10)

# Entry
name_entry = tk.Entry(root, width=30)
name_entry.pack(pady=5)

# Button
tk.Button(root, text="Submit", command=on_submit).pack(pady=10)

# Focus su entry
name_entry.focus()

# Bind Enter key
root.bind("<Return>", lambda e: on_submit())

root.mainloop()
""")

# ══════════════════════════════════════════════════════════════════════════════
# QUIZ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("QUIZ DI VERIFICA")
print("=" * 70)

print("""
Q1. Quale metodo avvia il loop principale Tkinter?
    A) start()  B) mainloop()  C) run()  D) loop()
    → RISPOSTA: B

Q2. Quale geometry manager usa row e column?
    A) pack()  B) place()  C) grid()  D) All
    → RISPOSTA: C

Q3. Quale evento rappresenta il click sinistro?
    A) <Click>  B) <Button-1>  C) <LeftClick>  D) <Mouse-1>
    → RISPOSTA: B

Q4. Come ottenere il testo da un Entry?
    A) entry.text  B) entry.value  C) entry.get()  D) entry.read()
    → RISPOSTA: C

Q5. Quale widget è usato per input password?
    A) Password  B) Entry con show="*"  C) Text  D) Input
    → RISPOSTA: B

Q6. Quale metodo bind collega evento a widget?
    A) connect()  B) bind()  C) attach()  D) link()
    → RISPOSTA: B

Q7. Canvas create_rectangle restituisce?
    A) Widget  B) None  C) ID intero  D) Coordinates
    → RISPOSTA: C

Q8. Quale Variable per checkbox?
    A) StringVar  B) IntVar  C) CheckVar  D) BoolVar
    → RISPOSTA: B (0 o 1)
""")

print("\n" + "=" * 70)
print("GUI MODULE COMPLETATO!")
print("=" * 70)
