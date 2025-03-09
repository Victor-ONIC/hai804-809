import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def ppm_to_hue_histogram(ppm_path):
    # read image and convert to hsv
    image = cv2.imread(ppm_path)
    if image is None:
        raise ValueError("Error loading image. Check file path and format.")
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # create hist
    hue_channel = hsv_image[:, :, 0]
    hist, bins = np.histogram(hue_channel, bins=180, range=(0, 180))
    
    # draw hist
    plt.figure(figsize=(10, 5))
    plt.bar(range(180), hist, color='orange', edgecolor='black')
    plt.xlabel('Hue Value')
    plt.ylabel('Frequency')
    plt.title('Hue Histogram')
    plt.show()
    
    return hist



def wheel(hist):
    # passer des bins aux angles
    num_bins = len(hist)
    theta = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)

    radii = hist / np.max(hist)  # hist normalisé

    # créer le plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_frame_on(False)

    # cercle bg
    res = 1000
    theta_bg = np.linspace(0, 2 * np.pi, res)
    r_bg = np.ones(res)
    colors_bg = [plt.cm.hsv(i / res) for i in range(res)]
    ax.scatter(theta_bg, r_bg, c=colors_bg, s=2, alpha=1)

    # pics de couleurs
    for angle, radius, hue in zip(theta, radii, range(num_bins)):
        color = plt.cm.hsv(hue / num_bins)
        ax.plot([angle, angle], [1, 1 - radius], color=color, linewidth=2)

    plt.show()



if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    hist = ppm_to_hue_histogram(dir_path+"/Test.ppm")
    wheel(hist)

