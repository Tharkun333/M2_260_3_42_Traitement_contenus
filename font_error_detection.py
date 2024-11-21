import cv2
import numpy as np
from tesserocr import PyTessBaseAPI, PSM, RIL
from PIL import Image

def get_texte_with_tesseract_OCR(image):
    tessdata_path = "C:\Program Files\Tesseract-OCR\tessdata"
    output_path = "detected_fonts_tesserocr.png"

    # Extraire le bruit
    noise = extract_noise(image)

    # Détecter les contours
    edges = detect_edges(image)

    # inverser les couleurs
    return cv2.bitwise_not(edges)

def extract_noise(image):
    """Extrait le bruit d'une image en utilisant un filtre médian."""
    median = cv2.medianBlur(image, 5)
    median = cv2.GaussianBlur(image, (3, 3), 0)
    noise = cv2.subtract(image, median)
    return noise
def detect_edges(image):
    """Détecte les contours dans une image."""
    edges = cv2.Canny(image, 175, 275)
    # Améliorer a qualité des contours
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # remplire les trous dans les contours avec un seuil pour pas que les 8 ou 9 soit tous blanc
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges

def detect_font_fraud_within_blocks(image_path, output_path, tessdata_path=None):
    # Charger l'image
    image = cv2.imread(image_path)

    # Sauvegarder l'image temporairement au format requis par Tesseract
    temp_image_path = "temp_image.png"
    cv2.imwrite(temp_image_path, get_texte_with_tesseract_OCR(image))

    # Charger l'image
    processed_image = cv2.imread(temp_image_path)

    falsified = False
    # image = processed_image

    # Configurer Tesseract avec le chemin `tessdata`
    with PyTessBaseAPI(path=tessdata_path, psm=PSM.SPARSE_TEXT) as api:
        # Vérifier les dimensions de l'image
        if len(processed_image.shape) == 2:  # Image en niveaux de gris
            height, width = processed_image.shape
            channels = 1  # 1 canal pour les images en niveaux de gris
            strides = width
        elif len(processed_image.shape) == 3:  # Image en couleur
            height, width, channels = processed_image.shape
            strides = processed_image.strides[0]
        else:
            print("Format d'image non pris en charge.")
            return
        
        api.SetImageBytes(processed_image.tobytes(), width, height, channels, strides)

        # Obtenir les blocs de texte (niveaux `RIL.BLOCK`)
        blocks = api.GetComponentImages(RIL.BLOCK, True)
        # Obtenir les caractères dans le bloc
        symbols = api.GetComponentImages(RIL.SYMBOL, True)
        charcount = 0

        # Analyse par bloc
        for i, (block_image, bounding_box, _, _) in enumerate(blocks):
            # S'assurer que les coordonnées sont des entiers
            x, y, w, h = bounding_box['x'], bounding_box['y'], bounding_box['w'], bounding_box['h']
            ecart = 15
            if x >= ecart and x + w < width-2*ecart:
                x -= ecart
                w += 2*ecart

            if y >= ecart and y + h < height-2*ecart:
                y -= ecart
                h += 2*ecart
            
            # Afficher le bloc
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # cv2.imshow("Bloc", image[y:y+h, x:x+w])
            # cv2.waitKey(0)

            # Définir la région d’intérêt (ROI) pour le bloc
            api.SetRectangle(x, y, w, h)

            # Obtenir les caractères dans le bloc
            # symbols = api.GetComponentImages(RIL.SYMBOL, True)

            # Encadrer les blocs en bleu
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Analyser les tailles des caractères
            font_sizes = []
            char_positions = []
            char_bounding_boxes = []
            for i, (symbol, char_box, _, _) in enumerate(symbols):
                char_x, char_y, char_w, char_h = char_box['x'], char_box['y'], char_box['w'], char_box['h']  # S'assurer que les coordonnées sont des entiers

                if char_x >= x and char_y >= y and char_x + char_w <= x + w and char_y + char_h <= y + h:
                    font_size = char_w # * char_h
                    font_sizes.append(font_size)
                    char_positions.append(char_y + char_h // 2)
                    char_bounding_boxes.append((char_x, char_y, char_w, char_h, font_size))
            
            # Calcul des statistiques pour les tailles de caractères dans ce bloc
            if font_sizes:
                mean_size = np.median(font_sizes)
                std_size = np.std(font_sizes)
                threshold = 3
                # threshold_up = abs(mean_size + threshold * std_size)  # Détecte des anomalies
                # threshold_down = abs(mean_size - threshold * std_size)  # Détecte des anomalies
                
                q1 = np.percentile(font_sizes, 25)
                q3 = np.percentile(font_sizes, 75)
                threshold_up = q3 + threshold
                threshold_down = min(q1 - threshold , 0)

                median_y = np.median(char_positions)  # Ligne médiane des caractères
                y_threshold_up = median_y + threshold  # Seuil pour détecter des anomalies
                y_threshold_down = median_y - threshold

                for (char_x, char_y, char_w, char_h, font_size) in char_bounding_boxes:
                    # cv2.rectangle(
                    #     image,
                    #     (char_x, char_y),
                    #     (char_x + char_w, char_y + char_h),
                    #     (0, 255, 0), 2
                    # )
                    # print(i, threshold_up, threshold_down, charcount, font_size, image_path)
                    # Ajouter un label avec les caractéristiques approximées
                    # cv2.putText(
                    #     image,
                    #     f"{charcount}{i}",
                    #     (char_x, char_y - 10),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.5,
                    #     (0, 255, 0),
                    #     1,
                    #     cv2.LINE_AA,
                    # )
                    charcount += 1
                    if font_size > threshold_up or font_size < threshold_down:   # Si la taille dépasse le seuil
                        if api.GetUTF8Text().strip():
                            cv2.rectangle(
                                image,
                                (char_x, char_y),
                                (char_x + char_w, char_y + char_h),
                                (0, 0, 255), 2  # Rouge pour les anomalies
                            )
                            falsified = True
                        # else:
                        #     cv2.rectangle(
                        #         image,
                        #         (char_x, char_y),
                        #         (char_x + char_w, char_y + char_h),
                        #         (0, 255, 255), 2  # Jaune pour les anomalies vides
                        #     )
                    
                    # char_center_y = char_y + char_h // 2
                    # if char_center_y > y_threshold_up or char_center_y < y_threshold_down:
                    #     cv2.rectangle(
                    #         image,
                    #         (char_x, char_y),
                    #         (char_x + char_w, char_y + char_h),
                    #         (255, 0, 255), 2  # Violet pour décalage vertical
                    #     )
                    #     cv2.putText(
                    #         image,
                    #         "Decalage",
                    #         (char_x, char_y - 10),
                    #         cv2.FONT_HERSHEY_SIMPLEX,
                    #         0.5,
                    #         (255, 0, 255),
                    #         1,
                    #         cv2.LINE_AA,
                    #     )

    # Sauvegarder l'image avec les encadrements
    if falsified:
        cv2.imwrite(output_path, image)
        print(f"Résultat sauvegardé dans : {output_path}")
    return falsified

# Exemple d'utilisation
tessdata_path = "C:/Program Files/Tesseract-OCR/tessdata"  # Modifier selon votre configuration
# detect_font_fraud_within_blocks("data/tickets_modif/43_nique_sa_mere.jpg", "resultat_detecte_par_bloc.jpg", tessdata_path=tessdata_path)


# parcourir toutes les images du dossier data/tickets_modif
import os

for file in os.listdir("data/tickets_modif"):
    detect_font_fraud_within_blocks(
        os.path.join("data/tickets_modif", file),
        "results_mylan/" + file,
        tessdata_path=tessdata_path
    )