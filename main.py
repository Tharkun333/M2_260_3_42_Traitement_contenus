import os
import cv2 as cv
from modules.image_processing import load_image, price_calculate, detect_bg_color
from modules.taille import calculate_boxplot_statistics, detect_fraud_by_boxplot, extract_metadata
# utiliser la première et mettre les résultats dans la 2eme, la 3eme s'utilise toute seule
import xml.etree.ElementTree as ET
from xml.dom import minidom

def write_results_to_xml(results, output_path="results.xml"):
    """
    Écrit les résultats dans un fichier XML avec des sauts de ligne après chaque balise.

    Args:
        results (list): Liste des résultats, chaque élément est un dictionnaire avec :
                        - id : ID du document
                        - modified : 0 ou 1 (si modifié ou non)
                        - raisons : Liste des raisons pour modification
        output_path (str): Chemin du fichier XML de sortie.
    """
    # Créer la structure XML
    root = ET.Element("GT")

    for result in results:
        doc = ET.SubElement(root, "doc", id=str(result["id"]), modified=str(result["modified"]))
        raisons = ", ".join(result["raisons"]) if result["raisons"] else ""
        doc.set("raison", raisons)

    # Convertir en chaîne XML formatée
    xml_string = ET.tostring(root, encoding="unicode")
    formatted_xml = minidom.parseString(xml_string).toprettyxml(indent="  ")

    # Écrire dans le fichier avec les sauts de ligne
    with open(output_path, "w", encoding="utf-8") as xml_file:
        xml_file.write(formatted_xml)


if __name__ == "__main__":

    folder_path = "data/tickets"
    results = []

    images_origine = "data/tickets"
    images_modifies = "data/img_modif"

    (sizes, lower_bound_size, upper_bound_size, median_size,lower_bound_width, upper_bound_width,lower_bound_height, upper_bound_height) = calculate_boxplot_statistics(images_origine)


    # Liste des fichiers dans le dossier
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    total_files = len(files)

    
    # Parcourir toutes les images du dossier avec une barre de progression
    for i, filename in enumerate(files):
        # Afficher la barre de progression
        progress = (i + 1) / total_files * 100
        print(f"\rProgression : {progress:.2f}% ({i + 1}/{total_files})", end="")

        # Construire le chemin complet du fichier
        file_path = os.path.join(folder_path, filename)



        # Charger l'image
        image = cv.imread(file_path)

        # Initialiser les raisons
        raisons = []

        # Détecter si l'image a une taille anormale (dépassant l'écart interquartile)
        fraude_boxplot = detect_fraud_by_boxplot(file_path, lower_bound_size, upper_bound_size, median_size,lower_bound_width, upper_bound_width,lower_bound_height, upper_bound_height, median_threshold=0.2)
        if fraude_boxplot:
            raisons.append("Taille de l'image anormale")

        metadata = extract_metadata(file_path)
        if metadata:
            raisons.append("Metadata obtenues, traces de modifications")
        # DETECTION DE MAUVAIS CALCUL
        # is_price_good = price_calculate(image)
        # if not is_price_good:
        #     raisons.append("Mauvais calcul")

        # DETECTION DE COULEUR INCORRECTE
        is_color_good = detect_bg_color(image)
        if not is_color_good:
            raisons.append("Tâche ou couleur incorrecte")

        # Ajouter le résultat dans la liste
        results.append({
            "id": i,
            "modified": 1 if raisons else 0,
            "raisons": raisons
        })

    # Sauter une ligne après la barre de progression
    print()

    # Écrire les résultats dans le fichier XML
    write_results_to_xml(results, output_path="results.xml")
    print("Résultats écrits dans results.xml")
