import os
import cv2 as cv
from modules.image_processing import load_image, price_calculate
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
