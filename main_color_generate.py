import os
import cv2 as cv
from modules.image_processing import detect_bg_color_visuel

if __name__ == "__main__":

    folder_path = "data/tickets_mimimimi"
    
    # Liste spécifique de fichiers à traiter
    files_to_process = [
        "188.jpg",
        "81.png"
    ]

    total_files = len(files_to_process)

    # folder_path = "data/tickets_mimimimi"
    # files_to_process = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # total_files = len(files_to_process)
    
    # Parcourir uniquement les fichiers spécifiés avec une barre de progression
    for i, filename in enumerate(files_to_process):

        # Vérifier si le fichier existe dans le dossier
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            print(f"\nFichier non trouvé : {filename}")
            continue

        # Afficher la barre de progression
        progress = (i + 1) / total_files * 100
        print(f"\rProgression : {progress:.2f}% ({i + 1}/{total_files})", end="")

        # Charger l'image
        image = cv.imread(file_path)

        # Appeler la fonction pour détecter les anomalies
        detect_bg_color_visuel(image, filename.split('.')[0])  # Utiliser le nom de fichier sans extension
