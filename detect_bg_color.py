import cv2 as cv
import numpy as np
from collections import Counter

def detect_bg_color(image_path):
    """
    Identifie les blocs 50x50 avec une couleur dominante différente des couleurs dominantes globales
    en utilisant la somme des différences des couches (R, G, B) et une condition supplémentaire :
    l'une des composantes doit être > 230.
    Dessine uniquement les contours rouges autour des blocs différents sur l'image originale.

    Args:
        image_path (str): Chemin de l'image à analyser.

    Returns:
        None
    """
    # Chargement de l'image
    img = cv.imread(image_path)

    if img is None:
        print(f"Erreur : L'image '{image_path}' n'a pas été trouvée.")
        return

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
    frequent_colors = {color for color, count in Counter(dominant_colors).items() if count >= 10 or count <=2}

    # Créer une copie de l'image pour dessiner uniquement les contours rouges
    result_img = img.copy()

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
                if sum(abs(dominant_color_bgr[i] - global_color[i]) for i in range(3)) <= 20:
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

            # Dessiner un contour rouge uniquement pour les blocs en erreur
            if is_error:
                cv.rectangle(result_img, (x, y), (x + block_size, y + block_size), (0, 0, 255), 2)

    # Redimensionner l'image pour qu'elle tienne dans l'écran
    screen_width = 1920
    screen_height = 1080
    scale = min(screen_width / img_width, screen_height / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    resized_img = cv.resize(result_img, (new_width, new_height), interpolation=cv.INTER_AREA)

    # Affichage de l'image redimensionnée
    cv.imshow("Macro Blocks with Red Contours", resized_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Afficher les deux couleurs dominantes globales
    print("Les deux couleurs dominantes globales sont :")
    for color in global_dominant_colors:
        print(f"Couleur (BGR) : {color}")

    # Afficher les couleurs fréquentes
    print("Couleurs fréquentes (>= 10 blocs) :")
    for color in frequent_colors:
        print(f"Couleur (BGR) : {color}")

