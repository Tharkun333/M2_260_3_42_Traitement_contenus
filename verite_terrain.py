import os
import xml.etree.ElementTree as ET


def generate_ground_truth_xml(tickets_dir, img_modif_dir, output_file):
    # Collecter les noms des fichiers dans les dossiers
    tickets_images = [(int(img.split('.')[0]), 0) for img in os.listdir(tickets_dir) if
                      img.endswith(('.png', '.jpg', '.jpeg'))]
    img_modif_images = [(int(img.split('.')[0]), 1) for img in os.listdir(img_modif_dir) if
                        img.endswith(('.png', '.jpg', '.jpeg'))]

    # Combiner et trier les fichiers
    all_images = tickets_images + img_modif_images
    all_images.sort(key=lambda x: x[0])

    # Création du fichier XML
    root = ET.Element("GT")
    comment = ET.Comment("Le document 0 est un original\nLe document 1 a été modifié")
    root.append(comment)

    for img_name, modified in all_images:
        doc_element = ET.SubElement(root, "doc", id=str(img_name), modified=str(modified))

    # Sauvegarde du fichier XML
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Fichier de vérité terrain généré : {output_file}")


# Chemins vers les dossiers
tickets_dir = "./data/tickets"
img_modif_dir = "./data/img_modif"
output_file = "verite_terrain.xml"

generate_ground_truth_xml(tickets_dir, img_modif_dir, output_file)
