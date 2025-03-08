from PIL import Image
from Histogramme import Histogramme

nameImIn = "perso.ppm"
imIn = Image.open(nameImIn)
hist = Histogramme(12)
hist.load(imIn)
nbx = imIn.size[0]
nby = imIn.size[1]
total = nbx*nby
for i in range(12) :
    print(i, round(hist.data[i]/float(total)*100, 2), "%")