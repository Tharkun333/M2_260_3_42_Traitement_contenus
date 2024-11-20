import cv2
import numpy as np
import re
from difflib import SequenceMatcher
import pytesseract
from collections import Counter

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
    
###########################################
# -------- # DETECTION DU PRIX # -------- #
###########################################


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

################################################
# -------- # DETECTION DES COULEURS # -------- #
################################################
def detect_bg_color(img):
    """
    Identifie les blocs 50x50 avec une couleur dominante différente des couleurs dominantes globales
    en utilisant la somme des différences des couches (R, G, B) et une condition supplémentaire :
    l'une des composantes doit être > 230.
    Renvoie True si aucun bloc n'est détecté comme erreur, sinon False.

    Args:
        img (np.ndarray): Image à analyser.

    Returns:
        bool: True si aucune erreur, False sinon.
    """
    # Dimensions des blocs
    block_size = 50

    # Dimensions de l'image
    img_height, img_width = img.shape[:2]

    # Liste pour stocker les couleurs dominantes de chaque bloc
    dominant_colors = []

    # Calcul des couleurs dominantes par macro-bloc
    for y in range(0, img_height, block_size):
        for x in range(0, img_width, block_size):
            block = img[y:y + block_size, x:x + block_size]
            pixels_list = block.reshape(-1, 3)  # Liste 2D des pixels
            dominant_color = Counter(map(tuple, pixels_list)).most_common(1)[0][0]
            dominant_colors.append(tuple(map(int, dominant_color)))

    # Identifier les deux couleurs les plus dominantes globales
    global_dominant_colors = [color for color, _ in Counter(dominant_colors).most_common(2)]

    # Identifier les couleurs dominantes fréquentes (>= 10 blocs)
    frequent_colors = {color for color, count in Counter(dominant_colors).items() if count >= 10 or count <= 2}

    # Masque pour les erreurs
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Parcourir les blocs pour détecter les erreurs
    for y in range(0, img_height, block_size):
        for x in range(0, img_width, block_size):
            block = img[y:y + block_size, x:x + block_size]
            pixels_list = block.reshape(-1, 3)
            dominant_color = Counter(map(tuple, pixels_list)).most_common(1)[0][0]
            dominant_color_bgr = tuple(map(int, dominant_color))

            # Condition 1 : Vérifier la somme des différences avec les couleurs dominantes globales
            is_error = all(
                sum(abs(dominant_color_bgr[i] - global_color[i]) for i in range(3)) > 20
                for global_color in global_dominant_colors
            )

            # Condition 2 : Vérifier si la couleur est fréquente (>= 10 blocs)
            if dominant_color_bgr in frequent_colors:
                is_error = False

            # Condition 3 : Vérifier si l'une des composantes est > 230
            if any(component > 230 for component in dominant_color_bgr):
                is_error = True

            # Ajouter au masque si une erreur est détectée
            if is_error:
                cv2.rectangle(mask, (x, y), (x + block_size, y + block_size), 255, -1)

    # Trouver les contours uniques dans le masque
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Renvoie False s'il y a des erreurs, sinon True
    return len(contours) == 0


import cv2
import os
from collections import Counter

import cv2
import os
from collections import Counter
import numpy as np

def detect_bg_color_visuel(img, name):
    """
    Identifie les blocs 50x50 avec une couleur dominante différente des couleurs dominantes globales
    et applique un cercle englobant tous les blocs d'anomalies connectés.
    Enregistre l'image dans le dossier "results" avec le nom spécifié.

    Args:
        img (numpy.ndarray): Image à analyser (sous forme de matrice numpy).
        name (str): Nom du fichier de sortie (sans extension).

    Returns:
        None
    """
    # Dimensions des blocs
    block_size = 50

    # Liste pour stocker les couleurs dominantes de chaque bloc
    dominant_colors = []

    # Parcourir l'image par macro-blocs pour déterminer la couleur dominante de chaque bloc
    img_height, img_width = img.shape[:2]
    for y in range(0, img_height, block_size):
        for x in range(0, img_width, block_size):
            # Extraire le macro-bloc
            block = img[y:y + block_size, x:x + block_size]

            # Trouver la couleur dominante dans le bloc
            pixels = block.reshape(-1, 3)  # Convertir les pixels en une liste 2D
            pixels_list = [tuple(pixel) for pixel in pixels]  # Convertir chaque pixel en tuple
            dominant_color = Counter(pixels_list).most_common(1)[0][0]  # Couleur la plus fréquente

            # Ajouter la couleur dominante à la liste
            dominant_colors.append(tuple(map(int, dominant_color)))

    # Identifier les deux couleurs les plus dominantes globales
    global_dominant_colors = [color for color, _ in Counter(dominant_colors).most_common(2)]

    # Identifier les couleurs dominantes fréquentes (>= 10 blocs)
    frequent_colors = {color for color, count in Counter(dominant_colors).items() if count >= 10}

    # Liste des centres des blocs en erreur
    error_centers = []

    # Parcourir à nouveau pour évaluer la différence avec les deux couleurs dominantes globales
    for y in range(0, img_height, block_size):
        for x in range(0, img_width, block_size):
            # Extraire le macro-bloc
            block = img[y:y + block_size, x:x + block_size]

            # Trouver la couleur dominante dans le bloc
            pixels = block.reshape(-1, 3)
            pixels_list = [tuple(pixel) for pixel in pixels]
            dominant_color = Counter(pixels_list).most_common(1)[0][0]
            dominant_color_bgr = tuple(map(int, dominant_color))

            # Vérifier si la couleur dominante est considérée comme une erreur
            is_error = True

            # Condition 1 : Vérifier la somme des différences avec les couleurs dominantes globales
            for global_color in global_dominant_colors:
                if sum(abs(dominant_color_bgr[i] - global_color[i]) for i in range(3)) <= 30:
                    is_error = False
                    break

            # Condition 2 : Vérifier si la couleur est fréquente (>= 10 blocs)
            if dominant_color_bgr in frequent_colors:
                is_error = False

            # Condition 3 : Vérifier si l'une des composantes est > 230
            if is_error and any(component > 230 for component in dominant_color_bgr):
                is_error = True
            else:
                is_error = False

            # Stocker le centre des blocs en erreur
            if is_error:
                center_x = x + block_size // 2
                center_y = y + block_size // 2
                error_centers.append((center_x, center_y))

    # Si des anomalies ont été trouvées, dessiner un cercle englobant
    if error_centers:
        # Calculer le cercle englobant
        points = np.array(error_centers)
        (x_center, y_center), radius = cv2.minEnclosingCircle(points)

        # Dessiner le cercle sur une copie de l'image
        result_img = img.copy()
        center = (int(x_center), int(y_center))
        radius = int(radius)
        cv2.circle(result_img, center, radius, (0, 0, 255), 2)

        # Enregistrer l'image
        os.makedirs("results", exist_ok=True)
        output_path = os.path.join("results", f"{name}.png")
        cv2.imwrite(output_path, result_img)



