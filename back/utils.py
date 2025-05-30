import re
import unicodedata
import mysql.connector
from mysql.connector import Error
from config import DB_CONFIG # Import DB config

# --- Function to get DB Connection ---
def get_db_connection():
    """Creates and returns a database connection."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            # print("Database connection successful.") # Optional: Verbose logging
            return conn
    except Error as e:
        print(f"FATAL: Error connecting to MySQL Database: {e}")
        return None

# --- Text Cleaning Function ---
def clean_text_for_summarization(text: str) -> str:
    """
    Cleans text specifically for input to NLP models like summarizers.
    Removes control characters, normalizes unicode & whitespace, and
    attempts to remove specific observed noise patterns.
    """
    if not isinstance(text, str):
        return ""
    try:
        # Normalize unicode characters to their canonical form
        text = unicodedata.normalize('NFKC', text)
    except Exception as e:
        print(f"Warning: Unicode normalization failed: {e}")
        # Continue with the original text if normalization fails

    # Remove control characters but keep newline and tab
    cleaned_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        # Keep printable characters, newline, tab
        # Replace line/paragraph separators with newline
        if cat != 'Cc' or ch in ('\n', '\t'):
             if cat in ('Zl', 'Zp'): # Line separator, Paragraph separator
                 cleaned_chars.append('\n')
             else:
                 cleaned_chars.append(ch)
    text = "".join(cleaned_chars)

    # Standardize whitespace
    text = text.replace('\t', ' ')        # Replace tabs with spaces
    text = re.sub(r' {2,}', ' ', text)     # Replace multiple spaces with one
    text = re.sub(r'[ \t]*\n[ \t]*', '\n', text) # Remove leading/trailing whitespace around newlines
    text = re.sub(r'\n{2,}', '\n', text)   # Replace multiple newlines with one
    text = text.strip()                    # Remove leading/trailing whitespace from the whole string

    # --- Attempt to remove specific noise patterns ---
    # Often seen in extracted text, possibly figure/table artifacts
    # Remove specific hardcoded noise
    text = text.replace("# 1_#1 _", " ")
    # Remove patterns like "# 12 3.4 pt", "#5 10cm", etc.
    text = re.sub(r'#\s*\d*\s*(\d+(\.\d*)?|\.\d+)\s*(in|cm|pt|px)\b', ' ', text, flags=re.IGNORECASE)
    # Remove sequences of "# number" like "#1 #2 #3" if they occur frequently
    text = re.sub(r'(?:#\s*\d+\s*){3,}', ' ', text)
    # Clean up remaining single hashes surrounded by spaces, or at start/end
    text = re.sub(r'\s+#\s+', ' ', text)
    text = re.sub(r'^#\s+', '', text)
    text = re.sub(r'\s+#$', '', text)

    # Final whitespace cleanup
    text = re.sub(r' {2,}', ' ', text).strip()

    return text