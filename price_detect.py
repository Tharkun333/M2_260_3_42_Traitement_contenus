import cv2 as cv
import pytesseract
import re

# Configuration de Tesseract (modifiez le chemin si nécessaire)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def process_invoice(image_path):
    """
    Traite une image de facture pour vérifier que le prix total est bien la somme
    des prix de chaque élément.

    Args:
        image_path (str): Chemin de l'image de la facture.
    """
    # Charger l'image
    img = cv.imread(image_path)

    if img is None:
        print(f"Erreur : L'image '{image_path}' n'a pas été trouvée.")
        return

    # Convertir l'image en niveaux de gris
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Utiliser Tesseract pour extraire le texte avec les positions des éléments
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    # Étape 1 : Trouver le texte "MONTANT"
    montant_coords = None
    for i, text in enumerate(data["text"]):
        if "MONTANT" in text.upper():
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            montant_coords = (x, y, w, h)
            # Dessiner un rectangle vert autour du texte "MONTANT"
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break

    if montant_coords is None:
        print("Le texte 'MONTANT' n'a pas été trouvé.")
        return

    # Étape 2 : Trouver les prix sous "MONTANT"
    montant_bottom_y = montant_coords[1] + montant_coords[3]  # Bas de "MONTANT"
    prices = []
    for i, text in enumerate(data["text"]):
        if montant_bottom_y < data["top"][i]:  # En dessous de "MONTANT"
            if re.match(r"^\d+,\d{2}€$", text):  # Format "12,34€"
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                prices.append((text, x, y, w, h))
                # Dessiner un rectangle rouge autour des prix
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if not prices:
        print("Aucun prix détecté sous 'MONTANT'.")
        return

    # Étape 3 : Trouver le texte "TOTAL A PAYER" et le prix associé
    total_coords = None
    total_price = None
    for i, text in enumerate(data["text"]):
        if "TOTAL" in text.upper() and "PAYER" in text.upper():
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            total_coords = (x, y, w, h)
            # Chercher le prix à droite de "TOTAL A PAYER"
            for j in range(len(data["text"])):
                if data["top"][j] >= y and data["top"][j] <= y + h:  # Aligné verticalement
                    if data["left"][j] > x + w:  # À droite
                        if re.match(r"^\d+,\d{2}€$", data["text"][j]):  # Format "12,34€"
                            total_price = data["text"][j]
                            total_price_coords = (
                                data["left"][j],
                                data["top"][j],
                                data["width"][j],
                                data["height"][j],
                            )
                            # Dessiner un rectangle bleu autour du total
                            cv.rectangle(
                                img,
                                (total_price_coords[0], total_price_coords[1]),
                                (
                                    total_price_coords[0] + total_price_coords[2],
                                    total_price_coords[1] + total_price_coords[3],
                                ),
                                (255, 0, 0),
                                2,
                            )
                            break
            break

    if total_coords is None:
        print("Le texte 'TOTAL A PAYER' n'a pas été trouvé.")
        return

    if total_price is None:
        print("Le prix associé à 'TOTAL A PAYER' n'a pas été trouvé.")
        return

    # Afficher les prix trouvés
    print("Prix détectés sous 'MONTANT' :")
    for price in prices:
        print(price[0])

    print(f"Prix total détecté à côté de 'TOTAL A PAYER' : {total_price}")

    # Afficher l'image annotée
    cv.imshow("Facture Annotée", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


