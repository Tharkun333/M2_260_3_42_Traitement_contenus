import os
import cv2 as cv
from modules.image_processing import detect_bg_color_visuel

if __name__ == "__main__":
    folder_path = "data/img_modif"
    results_path = "results"

    # Supprimer toutes les images du dossier results
    if os.path.exists(results_path):
        for file in os.listdir(results_path):
            file_path = os.path.join(results_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Toutes les images dans le dossier '{results_path}' ont été supprimées.")
    else:
        os.makedirs(results_path)  # Créer le dossier s'il n'existe pas
        print(f"Dossier '{results_path}' créé.")

    # Liste spécifique de fichiers à traiter
    #files_to_process = ["81.png"]

    # folder_path = "data/tickets_mimimimi"
    files_to_process = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    total_files = len(files_to_process)

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
