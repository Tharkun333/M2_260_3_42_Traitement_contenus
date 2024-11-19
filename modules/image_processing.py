import cv2
import numpy as np
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

def detect_bg_color(image_path):
    """
    Identifie les blocs 50x50 avec une couleur dominante différente des couleurs dominantes globales
    en utilisant la somme des différences des couches (R, G, B) et une condition supplémentaire :
    l'une des composantes doit être > 230.
    Dessine des cercles englobants rouges autour des zones détectées comme erreurs.

    Args:
        image_path (str): Chemin de l'image à analyser.

    Returns:
        None
    """
    # Chargement de l'image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Erreur : L'image '{image_path}' n'a pas été trouvée.")
        return

    # Dimensions des blocs
    block_size = 50

    # Liste pour stocker les blocs détectés comme erreurs
    error_blocks = []

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
    frequent_colors = {color for color, count in Counter(dominant_colors).items() if count >= 10 or count <= 2}

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

            # Stocker les blocs en erreur
            if is_error:
                error_blocks.append((x, y, x + block_size, y + block_size))

    # Fusionner les blocs contigus pour générer des contours uniques
    result_img = img.copy()
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for block in error_blocks:
        cv2.rectangle(mask, (block[0], block[1]), (block[2], block[3]), 255, -1)

    # Trouver les contours uniques
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner des cercles englobants rouges sur l'image
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(result_img, center, radius, (0, 0, 255), 2)

    # Redimensionner l'image pour qu'elle tienne dans l'écran
    screen_width = 1920
    screen_height = 1080
    scale = min(screen_width / img_width, screen_height / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    resized_img = cv2.resize(result_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Affichage de l'image redimensionnée
    cv2.imshow("Macro Blocks with Red Circles", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Afficher les deux couleurs dominantes globales
    print("Les deux couleurs dominantes globales sont :")
    for color in global_dominant_colors:
        print(f"Couleur (BGR) : {color}")

    # Afficher les couleurs fréquentes
    print("Couleurs fréquentes (>= 10 blocs) :")
    for color in frequent_colors:
        print(f"Couleur (BGR) : {color}")

