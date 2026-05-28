import os
import tkinter as tk
from tkinter import filedialog

def pick_directory(initial_dir=".", title="Select Directory"):
    """
    Opens a GUI dialog to let the user select a directory.
    If the GUI cannot be opened (e.g., no display), it falls back to terminal input.
    """
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        root.attributes('-topmost', True) # Bring the dialog to the front
        
        selected_dir = filedialog.askdirectory(initialdir=initial_dir, title=title)
        root.destroy()
        
        if selected_dir:
            return os.path.abspath(selected_dir)
    except Exception as e:
        print(f"GUI selection failed: {e}. Falling back to manual input.")
    
    # Fallback to manual input
    path = input(f"{title} (relative or absolute path): ").strip()
    if os.path.isdir(path):
        return os.path.abspath(path)
    else:
        print(f"Error: '{path}' is not a valid directory.")
        return None

if __name__ == "__main__":
    # Example usage for picking a test set
    print("Please select the test set directory...")
    test_dir = pick_directory(initial_dir=os.getcwd(), title="Select Test Set Directory")
    if test_dir:
        print(f"You selected: {test_dir}")
    else:
        print("No directory selected.")
