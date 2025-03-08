from PIL import Image
from Histogramme import Histogramme

nameImIn = "perso.ppm"
imIn = Image.open(nameImIn)
nbx = imIn.size[0]
nby = imIn.size[1]
total = nbx*nby
hist = Histogramme(16)
hist.load(imIn)
for i in range(16) :
    print(i, round(hist.data[i]/float(total)*100, 2), "%")