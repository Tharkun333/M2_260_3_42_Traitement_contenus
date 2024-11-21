import xml.etree.ElementTree as ET

def compare_ground_truth_and_detection(gt_file, detection_file):
    """
    Compare les fichiers XML de vérité terrain et de détection générée pour évaluer la précision de la détection de fraude.
    """
    # Charger les fichiers XML
    gt_tree = ET.parse(gt_file)
    detection_tree = ET.parse(detection_file)

    gt_root = gt_tree.getroot()
    detection_root = detection_tree.getroot()

    # Lire les valeurs de vérité terrain
    gt_dict = {}
    for doc in gt_root.findall('doc'):
        doc_id = doc.get('id')
        modified = int(doc.get('modified'))
        gt_dict[doc_id] = modified

    # Lire les valeurs de détection
    detection_dict = {}
    for doc in detection_root.findall('doc'):
        doc_id = doc.get('id')
        modified = int(doc.get('modified'))
        detection_dict[doc_id] = modified

    # Comparer les deux dictionnaires
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for doc_id in gt_dict:
        gt_value = gt_dict[doc_id]
        detection_value = detection_dict.get(doc_id, 0)

        if gt_value == 1 and detection_value == 1:
            true_positives += 1
        elif gt_value == 0 and detection_value == 0:
            true_negatives += 1
        elif gt_value == 0 and detection_value == 1:
            false_positives += 1
        elif gt_value == 1 and detection_value == 0:
            false_negatives += 1

    # Résultats de la comparaison
    total = len(gt_dict)
    accuracy = (true_positives + true_negatives) / total
    print(f"Total documents : {total}")
    print(f"True Positives : {true_positives}")
    print(f"True Negatives : {true_negatives}")
    print(f"False Positives : {false_positives}")
    print(f"False Negatives : {false_negatives}")
    print(f"Accuracy : {accuracy:.2f}")

# Utilisation de la fonction
gt_file = "verite_terrain.xml"  # Fichier XML de vérité terrain
detection_file = "détection_fraude_test.xml"  # Fichier XML généré par la détection

compare_ground_truth_and_detection(gt_file, detection_file)