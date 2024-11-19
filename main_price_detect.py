import price_detect

# Exemple d'appel
if __name__ == "__main__":
    image_path = "./data/tickets_modif/43_tache_blanche.jpg"  # Remplacez par le chemin de l'image de la facture
    price_detect.process_invoice(image_path)