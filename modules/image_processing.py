import cv2
import numpy as np


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
