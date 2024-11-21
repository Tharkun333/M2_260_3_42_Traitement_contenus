import os
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
from PIL.ExifTags import TAGS

def calculate_boxplot_statistics(base_dir, iqr_factor=2.0):
    """
    Calcule les statistiques nécessaires pour le boxplot à partir des images de base.
    """
    sizes = []
    widths = []
    heights = []

    # Parcourir les images dans le dossier de base
    for img_name in os.listdir(base_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(base_dir, img_name)
            with Image.open(img_path) as img:
                width, height = img.size
                sizes.append(width * height)
                widths.append(width)
                heights.append(height)

    # Calcul des quartiles, de l'écart interquartile (IQR) et de la médiane
    sizes = np.array(sizes)
    widths = np.array(widths)
    heights = np.array(heights)

    Q1_size = np.percentile(sizes, 25)
    Q3_size = np.percentile(sizes, 75)
    IQR_size = Q3_size - Q1_size
    median_size = np.median(sizes)
    lower_bound_size = Q1_size - iqr_factor * IQR_size
    upper_bound_size = Q3_size + iqr_factor * IQR_size

    Q1_width = np.percentile(widths, 25)
    Q3_width = np.percentile(widths, 75)
    IQR_width = Q3_width - Q1_width
    lower_bound_width = Q1_width - iqr_factor * IQR_width
    upper_bound_width = Q3_width + iqr_factor * IQR_width

    Q1_height = np.percentile(heights, 25)
    Q3_height = np.percentile(heights, 75)
    IQR_height = Q3_height - Q1_height
    lower_bound_height = Q1_height - iqr_factor * IQR_height
    upper_bound_height = Q3_height + iqr_factor * IQR_height

    return (sizes, lower_bound_size, upper_bound_size, median_size,
            lower_bound_width, upper_bound_width,
            lower_bound_height, upper_bound_height)

def detect_fraud_by_boxplot(image_path, lower_bound_size, upper_bound_size, median_size,
                            lower_bound_width, upper_bound_width,
                            lower_bound_height, upper_bound_height, median_threshold=0.2):
    """
    Détecte une fraude potentielle en comparant la taille, la largeur et la hauteur d'une image donnée avec les seuils du boxplot et la médiane.
    """
    with Image.open(image_path) as img:
        width, height = img.size
        size = width * height

    # Vérifier si la taille, la largeur ou la hauteur sont en dehors des bornes calculées
    size_anomaly = size < lower_bound_size or size > upper_bound_size
    median_anomaly = abs(size - median_size) / median_size > median_threshold
    width_anomaly = width < lower_bound_width or width > upper_bound_width
    height_anomaly = height < lower_bound_height or height > upper_bound_height

    # Considérer une image comme frauduleuse si au moins deux des critères échouent
    anomalies = [size_anomaly, median_anomaly, width_anomaly, height_anomaly]
    return sum(anomalies) >= 2

def extract_metadata(image_path):
    """
    Extrait les métadonnées EXIF de l'image et retourne un dictionnaire avec les informations pertinentes.
    """
    metadata = {}
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    metadata[tag] = value
    except Exception as e:
        print(f"Erreur lors de l'extraction des métadonnées pour {image_path}: {e}")
    return metadata

def generate_fraud_detection(base_dir, all_images_dir, output_file, iqr_factor=2.0):
    """
    Applique la détection de fraude sur toutes les images et génère un fichier XML.
    """
    # Collecter tous les chemins des fichiers dans le dossier
    all_images = [os.path.join(all_images_dir, img) for img in os.listdir(all_images_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # Trier par nom d'image (numérique, si applicable)
    all_images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # Calculer les statistiques du boxplot pour les images de base uniquement
    (base_sizes, lower_bound_size, upper_bound_size, median_size,
     lower_bound_width, upper_bound_width,
     lower_bound_height, upper_bound_height) = calculate_boxplot_statistics(base_dir, iqr_factor)

    # Générer le fichier XML
    root = ET.Element("GT")

    for img_path in all_images:
        # Détecter les anomalies de taille, largeur et hauteur par rapport aux bornes du boxplot et à la médiane
        size_fraud = detect_fraud_by_boxplot(img_path, lower_bound_size, upper_bound_size, median_size,
                                             lower_bound_width, upper_bound_width,
                                             lower_bound_height, upper_bound_height)
        is_fraud = size_fraud

        # Extraire les métadonnées de l'image
        metadata = extract_metadata(img_path)

        # Ajouter les résultats au fichier XML
        doc_element = ET.SubElement(root, "doc", id=os.path.splitext(os.path.basename(img_path))[0],
                                    modified=str(int(is_fraud)))
        for key, value in metadata.items():
            meta_element = ET.SubElement(doc_element, "metadata", name=key)
            meta_element.text = str(value)

    # Sauvegarder le fichier XML
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Fichier de résultat généré avec détection de fraudes : {output_file}")

# Chemins vers les dossiers
base_dir = "../data/tickets"  # Dossier des images de base (pour le calcul de la taille)
all_images_dir = "../data/img_Djit_Ouma_Rania/img"  # Dossier de toutes les images à analyser
output_file = "../détection_fraude_test.xml"

# Générer le fichier XML avec détection de fraudes
generate_fraud_detection(base_dir, all_images_dir, output_file, iqr_factor=2.0)