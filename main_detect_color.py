import detect_bg_color
# Exemple d'appel de la fonction
image_name = "img/4.jpg"  # Remplacez par le chemin de votre image
texts = detect_bg_color.detect_bg_color(image_name)
print("\nTextes détectés :")
for text in texts:
    print(f"- {text}")
