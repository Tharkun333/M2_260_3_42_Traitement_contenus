import matplotlib.pyplot as plt
import cv2
def show_image(image, title="Image"):
    """Affiche une image."""
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_image(image, output_path):
    """Sauvegarde une image."""
    cv2.imwrite(output_path, image)
