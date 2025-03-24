from colorsys import rgb_to_hsv as r2h
from PIL import Image
from Modele import Modele
        
def voisinsSup(i, j, nbx, nby) :
    v = []
    if (i < nbx-1) :
        v.append([i+1, j])
    if (j < nby-1) :
        v.append([i, j+1])
    return v

def voisins(i, j, nbx, nby) :
    v = []
    if (i > 0) :
        v.append([i-1, j])
    if (i < nbx-1) :
        v.append([i+1, j])
    if (j > 0) :
        v.append([i, j-1])
    if (j < nby-1) :
        v.append([i, j+1])
    return v
    
class Voisinage :
    def __init__(self, image, seuil) :
        self.seuil = seuil
        nbx = image.size[0]
        nby = image.size[1]
        self.data = [0 for i in range(nbx*nby)]
        data = image.load()
        hues = [0 for i in range(nbx*nby)]
        for i in range(nbx) :
            for j in range(nby) :
                r, g, b = [c/255.0 for c in data[i, j]]
                h, s, v = r2h(r, g, b)
                hues[j*nbx+i] = h
        value = 1
        for i in range(nbx) :
            for j in range(nby) :
                if (self.data[j*nbx+i] == 0) :
                    self.data[j*nbx+i] = value
                    voisin = voisinsSup(i, j, nbx, nby)
                    for v in voisin :
                        if (self.data[v[1]*nbx+v[0]] == 0) :
                            voisin_local = voisins(v[0], v[1], nbx, nby)
                            for vl in voisin_local :
                                if (self.data[vl[1]*nbx+vl[0]] == value and Modele.distance_congru(hues[vl[1]*nbx+vl[0]], hues[v[1]*nbx+v[0]]) < self.seuil) :
                                    self.data[v[1]*nbx+v[0]] = value
                            if (self.data[v[1]*nbx+v[0]] == value) :
                                for vl in voisin_local :
                                    if (self.data[vl[1]*nbx+vl[0]] == 0) :
                                        voisin.append(vl)
                value += 1
        """
        def voisinsSup(i, j, nbx, nby) :
            v = []
            if (i < nbx-1) :
                v.append([i+1, j])
            if (j < nby-1) :
                v.append([i, j+1])
            return v
        
        def voisins(i, j, nbx, nby) :
            v = []
            if (i > 0) :
                v.append([i-1, j])
            if (i < nbx-1) :
                v.append([i+1, j])
            if (j > 0) :
                v.append([i, j-1])
            if (j < nby-1) :
                v.append([i, j+1])
            return v"""

nameImIn = "../images/colorful.ppm"
imIn = Image.open(nameImIn)
vois = Voisinage(imIn, 0.04)
nbx = imIn.size[0]
nby = imIn.size[1]
dataIn = imIn.load()
imOut = Image.new(imIn.mode, imIn.size)
dataOut = imOut.load()
vData = vois.data
v=0
for i in vData :
    if (i>v) :
        v=i
for i in range(nbx) :
    for j in range(nby) :
        r, g, b = [int(255*vData[j*nbx+i]/v) for c in range(3)]
        dataOut[i, j] = (r, g, b)
print(v)
imIn.close()
imOut.save("../images/test_voisinage.ppm")
imOut.close()