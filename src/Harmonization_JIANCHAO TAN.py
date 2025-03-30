from PIL import Image
import scipy.spatial as ss #import ConvexHull
import numpy as np

def loadData(imIn) :
    # récupère les données d'une image pour les mettre dans un array numpy au bon format
    nbx = imIn.size[0]
    nby = imIn.size[1]
    dataIn = imIn.load()
    ldata = [[0, 0, 0] for i in range(nbx*nby)]
    for i in range(nbx) :
        for j in range(nby) :
            ldata[j*nbx+i][0], ldata[j*nbx+i][1], ldata[j*nbx+i][2] = dataIn[i, j]
    npdata = np.array(ldata)
    return npdata

def harmonization(imIn) :
    #0 Récupérer les données de l'image
    data = loadData(imIn)
    #1 calcul de l'enveloppe convexe RGB
    RGB_conv = ss.ConvexHull(data)
    RGB_vertices = data[RGB_conv.vertices]
    #1.5 réduire le nombre de point de l'enveloppe en fonction d'un nombre arbitraire de couleur
    #2 calcul de la palette
    #3 calcul de l'enveloppe convexe RGBXY
    #4 calcul des coordonnées barycentrique RGB et RGBXY
    #5 multiplication des matrices de coordonnées pour avoir la proportion de chaque couleur de la palette 
    pass

nameImIn = "images\colorful.ppm"
imIn = Image.open(nameImIn)
harmonization(imIn)