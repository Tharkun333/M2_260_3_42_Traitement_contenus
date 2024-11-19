import cv2
from modules.image_processing import (
    detect_character_shift,
    detect_character_shift_with_size,
    load_image,
    extract_noise,
    detect_edges,
)
from modules.ocr_analysis import extract_text
from modules.utils import show_image

if __name__ == "__main__":
    # Charger l'image
    image_path = "data/tickets_modif/647_rajout_0.jpg"
    image = load_image(image_path)

    # # Extraire le bruit
    # noise = extract_noise(image)
    # show_image(noise, "Bruit extrait")

    # # Détecter les contours
    # edges = detect_edges(image)
    # show_image(edges, "Contours détectés")

    # # OCR et extraction de texte
    # text = extract_text(image_path)
    # print("Texte extrait :\n", text)

    # # Détecter les décalages de caractères
    # shifted_image = detect_character_shift(image, cell_size=(50, 50), threshold=20)

    anomaly_detected_image = detect_character_shift_with_size(
        image, cell_size=(50, 50), size_threshold=1.5
    )

    import matplotlib.pyplot as plt

    # Afficher l'image avec les anomalies
    show_image(anomaly_detected_image, "Décalages de caractères détectés")
