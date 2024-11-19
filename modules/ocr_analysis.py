import pytesseract
from PIL import Image

# pytesseract.pytesseract.tesseract_cmd = r'C:\Path\To\tesseract.exe'  # Remplacez par le chemin de Tesseract.


def extract_text(image_path):
    """Extrait le texte d'une image."""
    text = pytesseract.image_to_string(Image.open(image_path))
    return text


def verify_totals(extracted_text):
    """Vérifie les totaux dans le texte extrait."""
    lines = extracted_text.split("\n")
    total_line = [line for line in lines if "TOTAL" in line.upper()]
    if total_line:
        print(f"Trouvé : {total_line[0]}")
    else:
        print("Aucune ligne contenant 'TOTAL' n'a été trouvée.")
