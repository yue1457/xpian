import nltk
try:
    nltk.download('punkt')
    print("punkt downloaded successfully.")
except Exception as e:
    print(f"Error downloading punkt: {e}")
