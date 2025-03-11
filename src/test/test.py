from PIL import Image
from src.utils.Histogramme import Histogramme

couleurs = ["rouge  ", "orange ", "jaune  ", "3      ", "vert   ", "5      ", "cyan   ", "7      ", "bleu   ", "violet ", "magenta", "11     "]

nameImIn = "perso.ppm"
imIn = Image.open(nameImIn)
hist = Histogramme(12)
hist.load(imIn)
nbx = imIn.size[0]
nby = imIn.size[1]
total = nbx*nby
for i in range(12) :
    print(couleurs[i], ":", round(hist.data[i]/float(total)*100, 2), "%")