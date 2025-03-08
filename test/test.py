from PIL import Image
from Histogramme import Histogramme

nameImIn = "perso.ppm"
imIn = Image.open(nameImIn)
hist = Histogramme(16)
hist.load(imIn)
nbx = imIn.size[0]
nby = imIn.size[1]
total = nbx*nby
for i in range(16) :
    print(i, round(hist.data[i]/float(total)*100, 2), "%")