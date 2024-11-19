import modules.image_processing as image_processing
# Exemple d'appel de la fonction
image_name = "data/tickets_modif/43_tache_blanche.jpg"
texts = image_processing.detect_bg_color(image_name)
