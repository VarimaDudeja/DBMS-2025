import customtkinter as ctk
from tkinter import messagebox

# ---------------------------
#  DATABASE PLACEHOLDERS
# ---------------------------

def create_user(username, password):
    print(f"[DB] Creating new user: {username}")
    messagebox.showinfo("Signup", "Account created successfully (dummy).")

def verify_user(username, password):
    print(f"[DB] Verifying user: {username}")
    if username == "admin" and password == "1234":
        return True
    else:
        messagebox.showerror("Login Failed", "Invalid credentials (demo check).")
        return False

def add_book(book_data):
    print("[DB] Adding book:", book_data)
    messagebox.showinfo("Book Added", f"Book '{book_data['book_name']}' added successfully (dummy).")

def find_book(book_name):
    """Placeholder for searching book"""
    print(f"[DB] Searching for book: {book_name}")
    if book_name.lower() == "demo":
        return {
            "book_name": "Demo Book",
            "author": "John Doe",
            "genre": "Fiction",
            "year": "2020",
            "edition": "First",
            "description": "This is a demo book for testing."
        }
    else:
        return None

# ---------------------------
#  MAIN APP CLASS
# ---------------------------

class LibraryApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("📚 Library Management System")
        self.geometry("700x600")

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.show_login_page()

    # ---------------------------
    # Utility: Clear window
    # ---------------------------
    def clear_window(self):
        for widget in self.winfo_children():
            widget.destroy()

    # ---------------------------
    # LOGIN PAGE
    # ---------------------------
    def show_login_page(self):
        self.clear_window()

        ctk.CTkLabel(self, text="🔐 Login", font=("Helvetica", 24, "bold")).pack(pady=30)

        ctk.CTkLabel(self, text="Username:").pack(pady=5)
        username_entry = ctk.CTkEntry(self, width=300)
        username_entry.pack(pady=5)

        ctk.CTkLabel(self, text="Password:").pack(pady=5)
        password_entry = ctk.CTkEntry(self, show="*", width=300)
        password_entry.pack(pady=5)

        def login_action():
            username = username_entry.get()
            password = password_entry.get()
            if verify_user(username, password):
                self.show_main_menu()

        ctk.CTkButton(self, text="Login", width=200, command=login_action).pack(pady=15)
        ctk.CTkButton(self, text="Sign Up", width=200, command=self.show_signup_page).pack(pady=5)

    # ---------------------------
    # SIGNUP PAGE
    # ---------------------------
    def show_signup_page(self):
        self.clear_window()

        ctk.CTkLabel(self, text="📝 Sign Up", font=("Helvetica", 24, "bold")).pack(pady=30)

        ctk.CTkLabel(self, text="Choose Username:").pack()
        username_entry = ctk.CTkEntry(self, width=300)
        username_entry.pack(pady=5)

        ctk.CTkLabel(self, text="Choose Password:").pack()
        password_entry = ctk.CTkEntry(self, show="*", width=300)
        password_entry.pack(pady=5)

        def signup_action():
            username = username_entry.get()
            password = password_entry.get()
            create_user(username, password)
            self.show_login_page()

        ctk.CTkButton(self, text="Create Account", width=200, command=signup_action).pack(pady=15)
        ctk.CTkButton(self, text="⬅ Back to Login", width=200, command=self.show_login_page).pack()

    # ---------------------------
    # MAIN MENU PAGE
    # ---------------------------
    def show_main_menu(self):
        self.clear_window()

        ctk.CTkLabel(self, text="🏠 Main Menu", font=("Helvetica", 22, "bold")).pack(pady=20)

        ctk.CTkButton(self, text="📚 Store Book Information", width=250, height=40,
                      command=self.show_store_book_page).pack(pady=15)

        ctk.CTkButton(self, text="🔍 Search Book", width=250, height=40,
                      command=self.show_search_book_page).pack(pady=15)

        ctk.CTkButton(self, text="🚪 Logout", fg_color="red", hover_color="#b30000",
                      command=self.show_login_page).pack(pady=25)

    # ---------------------------
    # STORE BOOK PAGE
    # ---------------------------
    def show_store_book_page(self):
        self.clear_window()

        ctk.CTkLabel(self, text="📘 Store Book Information", font=("Helvetica", 20, "bold")).pack(pady=20)

        entries = {}
        fields = ["Book Name", "Author Name", "Genre", "Year of Publication", "Edition"]
        for field in fields:
            ctk.CTkLabel(self, text=field + ":").pack()
            entry = ctk.CTkEntry(self, width=400)
            entry.pack(pady=5)
            entries[field.lower().replace(" ", "_")] = entry

        ctk.CTkLabel(self, text="Short Description:").pack(pady=5)
        desc_box = ctk.CTkTextbox(self, width=400, height=100)
        desc_box.pack()

        ctk.CTkButton(self, text="💾 Save Book", width=200,
                      command=lambda: self.save_book(entries, desc_box)).pack(pady=15)
        ctk.CTkButton(self, text="⬅ Back to Menu", width=200,
                      command=self.show_main_menu).pack()

    def save_book(self, entries, desc_box):
        book_data = {key: entry.get() for key, entry in entries.items()}
        book_data["description"] = desc_box.get("1.0", "end").strip()
        add_book(book_data)

    # ---------------------------
    # SEARCH BOOK PAGE
    # ---------------------------
    def show_search_book_page(self):
        self.clear_window()

        ctk.CTkLabel(self, text="🔍 Search Book", font=("Helvetica", 20, "bold")).pack(pady=20)

        ctk.CTkLabel(self, text="Enter Book Name:").pack()
        search_entry = ctk.CTkEntry(self, width=400)
        search_entry.pack(pady=5)

        result_box = ctk.CTkTextbox(self, width=500, height=200)
        result_box.pack(pady=10)

        def search_action():
            book_name = search_entry.get()
            result = find_book(book_name)
            result_box.delete("1.0", "end")

            if result:
                text = "\n".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in result.items()])
                result_box.insert("end", text)
            else:
                result_box.insert("end", "❌ Book not found.")

        ctk.CTkButton(self, text="Search", width=150, command=search_action).pack(pady=5)
        ctk.CTkButton(self, text="⬅ Back to Menu", width=150, command=self.show_main_menu).pack(pady=10)

# ---------------------------
# RUN THE APP
# ---------------------------
if __name__ == "__main__":
    app = LibraryApp()
    app.mainloop()
