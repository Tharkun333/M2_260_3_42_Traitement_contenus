import cv2
import numpy as np
import re
from difflib import SequenceMatcher

import pytesseract

def load_image(image_path):
    """Charge une image en niveaux de gris."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"L'image {image_path} n'existe pas.")
    return image


def extract_noise(image):
    """Extrait le bruit d'une image en utilisant un filtre médian."""
    median = cv2.medianBlur(image, 5)
    noise = cv2.subtract(image, median)
    return noise


def detect_edges(image):
    """Détecte les contours dans une image."""
    edges = cv2.Canny(image, 50, 150)
    return edges


def divide_into_patches(image, patch_size=20):
    """Divise une image en blocs."""
    h, w = image.shape
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i : i + patch_size, j : j + patch_size]
            patches.append(patch)
    return patches


# Marche PAS
def detect_character_shift(image, cell_size=(50, 50), threshold=20):
    """
    Détecte des décalages de caractères par rapport à une grille définie.
    Marque les zones suspectes avec un cercle.

    Args:
        image (numpy.ndarray): Image en niveaux de gris.
        cell_size (tuple): Taille des cellules de la grille (hauteur, largeur).
        threshold (int): Valeur seuil pour détecter un décalage de caractère.

    Returns:
        numpy.ndarray: Image avec les anomalies entourées de cercles.
    """
    # Copie de l'image pour dessiner les cercles
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Dimensions de l'image
    height, width = image.shape

    # Itérer sur la grille
    for y in range(0, height, cell_size[0]):
        for x in range(0, width, cell_size[1]):
            # Extraire une cellule
            cell = image[y : y + cell_size[0], x : x + cell_size[1]]

            # Calculer la position moyenne des pixels non noirs (présumés caractères)
            non_zero_coords = np.column_stack(np.where(cell > 0))
            if non_zero_coords.size > 0:
                center_y, center_x = non_zero_coords.mean(axis=0)

                # Vérifier si le centre dépasse un seuil par rapport au centre théorique
                center_shift_y = abs(center_y - cell_size[0] / 2)
                center_shift_x = abs(center_x - cell_size[1] / 2)

                if center_shift_y > threshold or center_shift_x > threshold:
                    # Marquer la cellule comme suspecte
                    circle_center = (int(x + center_x), int(y + center_y))
                    cv2.circle(output_image, circle_center, 15, (0, 0, 255), 2)

    return output_image


# Marche PAS
def detect_pixel_anomalies(image, block_size=(10, 10), position_threshold=5):
    """
    Detect anomalies by comparing pixel alignment in a block-by-block manner.

    Args:
        image (numpy.ndarray): Grayscale image.
        block_size (tuple): Size of the blocks (height, width).
        position_threshold (int): Maximum deviation in pixel alignment to consider as normal.

    Returns:
        numpy.ndarray: Image with anomalies highlighted by circles.
    """
    # Create a copy of the image for output
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Dimensions of the image
    height, width = image.shape

    # Loop through the image block by block
    for y in range(0, height, block_size[0]):
        for x in range(0, width, block_size[1]):
            # Extract the block
            block = image[y : y + block_size[0], x : x + block_size[1]]

            # Find the positions of non-zero pixels in the block
            non_zero_coords = np.column_stack(np.where(block > 0))

            if len(non_zero_coords) > 0:
                # Compute the average position of pixels in the block
                mean_y, mean_x = non_zero_coords.mean(axis=0)

                # Check if the pixel alignment deviates significantly
                for coord in non_zero_coords:
                    pixel_y, pixel_x = coord
                    deviation_y = abs(pixel_y - mean_y)
                    deviation_x = abs(pixel_x - mean_x)

                    if (
                        deviation_y > position_threshold
                        or deviation_x > position_threshold
                    ):
                        # Mark the anomaly on the output image
                        center = (int(x + pixel_x), int(y + pixel_y))
                        cv2.circle(output_image, center, 3, (0, 0, 255), -1)

    return output_image


# Configuration de Tesseract (modifiez le chemin si nécessaire)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def levenshtein_similarity(a, b):
    """Calcule la similarité de Levenshtein entre deux chaînes."""
    return SequenceMatcher(None, a, b).ratio()

def extract_numeric_value(text):
    """Extrait une valeur numérique valide à partir d'un texte."""
    # Remplacer les virgules par des points pour les décimales
    text = text.replace(",", ".")
    # Supprimer tous les caractères non numériques ou non décimaux (hors du point)
    match = re.search(r"\d+(\.\d+)?", text)
    return float(match.group()) if match else None

def process_ticket(data, img, target_keywords):
    """
    Processus générique pour traiter un ticket selon les mots-clés donnés.
    Args:
        data: Données extraites par Tesseract.
        img: Image en cours de traitement.
        target_keywords: Liste de mots-clés à rechercher (ex. ['MONTANT'] ou ['ST/TOTAL']).
    Returns:
        tuple: (img, is_valid, levenstein_min)
    """
    montant_coords = None
    montant_found = False
    closest_match_similarity = 0
    yellow_zones = []
    red_values = []
    purple_value = None

    # Parcourir les textes extraits pour annoter et rechercher les mots-clés
    for i, text in enumerate(data["text"]):
        if text.strip() == "":
            continue  # Ignorer les textes vides

        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

        # Vérifier si le texte correspond aux mots-clés
        if text.upper() in target_keywords:
            montant_found = True
            montant_coords = (x, x + w)  # Stocker les limites horizontales
        else:
            # Calculer la similarité avec les mots-clés
            for keyword in target_keywords:
                similarity = levenshtein_similarity(text.upper(), keyword.upper())
                if similarity > closest_match_similarity:
                    closest_match_similarity = similarity

        # Identifier les rectangles jaunes (TOTAL ou PAYER)
        if "TOTAL" in text.upper() or "PAYER" in text.upper():
            yellow_zones.append((y - 20, y + h + 20))  # Ajouter la plage verticale

        # Identifier les montants alignés avec les mots-clés
        if montant_coords:
            montant_x_start, montant_x_end = montant_coords
            if re.search(r"\d", text) and montant_x_start - 20 <= x <= montant_x_end + 20:
                # Vérifier si le rectangle est sous une zone jaune
                below_yellow_zone = all(y > yellow_max for _, yellow_max in yellow_zones)

                # Ignorer si sous une zone jaune
                if below_yellow_zone:
                    continue

                value = extract_numeric_value(text)
                if value is not None:
                    red_values.append(value)
                if "TOTAL" in text.upper():
                    purple_value = value

    # Calculer la somme des montants rouges
    red_sum = sum(red_values)

    # Vérifier la correspondance entre la somme des montants et la valeur violette
    is_valid = False
    if purple_value is not None:
        is_valid = abs(red_sum - purple_value) < 0.01

    return img, is_valid, closest_match_similarity

def price_calculate(img):
    """
    Fonction principale qui traite un ticket selon son type.
    Args:
        image_path (str): Chemin de l'image.
        output_path (str): Chemin pour enregistrer l'image annotée.
    Returns:
        bool: True si tout est valide, False sinon.
    """

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extraire les données avec Tesseract
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    # Vérifier si le ticket est de type "city" ou "gémo"
    img, is_valid, levenstein_min = process_ticket(data, img, ["MONTANT"])
    
    # Vérifier les conditions pour retourner True ou False
    if is_valid or (is_valid==False and levenstein_min < 0.50):
        return True
    return False
