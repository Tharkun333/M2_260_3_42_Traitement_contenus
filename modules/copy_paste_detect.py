import cv2
import numpy as np
from pytesseract import pytesseract

def detect_characters(image):
    boxes = pytesseract.image_to_boxes(image)
    h, w = image.shape[:2]
    chars = []
    for box in boxes.splitlines():
        b = box.split(' ')
        char = b[0]
        #if char.isalnum():  # Première vérification pour caractère alphanumérique
        if char.isdigit():  # Première vérification pour caractère alphanumérique
            # Double vérification: vérifier que le contour n'est pas vide
            char_image = image[h - int(b[4]):h - int(b[2]), int(b[1]):int(b[3])]
            if np.count_nonzero(char_image) > 10:  # Seuil à ajuster selon les besoins
                chars.append({
                    'char': char,
                    'x1': int(b[1]),
                    'y1': h - int(b[4]),
                    'x2': int(b[3]),
                    'y2': h - int(b[2])
                })
    return chars

def find_duplicate_characters_with_similar_pixels(chars, image, threshold=400):
    duplicates = []
    for i in range(len(chars)):
        for j in range(i + 1, len(chars)):
            if chars[i]['char'] == chars[j]['char']:
                dx = chars[i]['x1'] - chars[j]['x1']
                dy = chars[i]['y1'] - chars[j]['y1']
                distance = np.sqrt(dx * dx + dy * dy)
                if distance < threshold:
                    # Extraire les pixels des deux caractères
                    char1 = image[chars[i]['y1']:chars[i]['y2'], chars[i]['x1']:chars[i]['x2']]
                    char2 = image[chars[j]['y1']:chars[j]['y2'], chars[j]['x1']:chars[j]['x2']]
                    # Calculer le pourcentage de ressemblance
                    if char1.shape == char2.shape:
                        diff = cv2.absdiff(char1, char2)
                    else:
                        # Pad the smaller character image pour correspondre à la taille de la plus grande
                        height1, width1 = char1.shape[:2]
                        height2, width2 = char2.shape[:2]
                        height = max(height1, height2)
                        width = max(width1, width2)

                        padded_char1 = np.zeros((height, width), dtype=char1.dtype)
                        padded_char1[:height1, :width1] = char1

                        padded_char2 = np.zeros((height, width), dtype=char2.dtype)
                        padded_char2[:height2, :width2] = char2

                        diff = cv2.absdiff(padded_char1, padded_char2)
                    diff_pixels = np.count_nonzero(diff)
                    total_pixels = diff.size
                    similarity = (1 - diff_pixels / total_pixels) * 100
                    if similarity > 99:
                        if chars[i]['x1'] != chars[j]['x1'] or chars[i]['y1'] != chars[j]['y1']:
                            duplicates.append((chars[i], chars[j]))
    return duplicates

def detect_copy_paste(image, output_path = None):
    falsified = False

    # Prétraitement de l'image
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un filtre bilatéral pour réduire le bruit tout en préservant les bords
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Utiliser un seuillage adaptatif pour gérer les variations d'éclairage
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Effectuer une fermeture morphologique pour combler les petites lacunes dans le texte
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)  

    characters = detect_characters(processed_image)

    duplicate_chars = find_duplicate_characters_with_similar_pixels(characters, processed_image)

    for pair in duplicate_chars:
        for char in pair:
            cv2.rectangle(image, (char['x1'], char['y1']), (char['x2'], char['y2']), (0, 0, 255), 2)
            cv2.putText(image, char['char'], (char['x1'], char['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            falsified = True

    if output_path is not None:
        # Sauvegarder l'image avec les annotations
        if falsified:
            cv2.imwrite(output_path, image)
    return falsified