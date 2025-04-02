import math
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

def extractVT(hull, data) :
    v = hull.vertices
    vertices = data[v]
    dico = {val : i for i, val in enumerate(v)}
    triangles = [[dico[t[0]], dico[t[1]], dico[t[2]]] for t in hull.simplices]
    return (vertices, triangles)

def distance(v1, v2) :
    res = 0
    for i in range(len(v1)) :
        res += (v1[i]-v2[i])**2
    return math.sqrt(res)

def moyenne(vertices) :
    res = [0 for i in range(len(vertices[0]))]
    for vertex in vertices :
        for i in range(len(vertex)) :
            res[i] += vertex[i]
    for i in range(len(res)) :
        res[i] /= len(vertices)
    return res

def reducHull(vertices, triangles, targetNbr) :
    currentNbr = len(vertices)
    verticesPacks = [[i] for i in vertices]
    newVertices = [i for i in vertices]
    while (currentNbr > targetNbr) :
        min = distance(vertices[0], vertices[1])
        index = [0, 1]
        for i in range(currentNbr-1) :
            for j in range(i+1, currentNbr) :
                d = distance(newVertices[i], newVertices[j])
                if (d < min) :
                    min = d
                    index = [i, j]
        verticesPacks[index[0]] += verticesPacks.pop(index[1])
        newVertices[index[0]] = moyenne(verticesPacks[index[0]])
        newVertices.pop(index[1])
        currentNbr -= 1
    return newVertices

def harmonization(imIn, nbColor) :
    #0 Récupérer les données de l'image
    data = loadData(imIn)
    #1 calcul de l'enveloppe convexe RGB
    RGB_hull = ss.ConvexHull(data)
    RGB_vertices, RGB_triangles = extractVT(RGB_hull, data)
    #2 calcul de la palette
    RGB_reduc_conv = reducHull(RGB_vertices, RGB_triangles, nbColor)
    #2.5 étirer les points pour réenvelopper ce qui est sorti
    #3 calcul de l'enveloppe convexe RGBXY
    #4 calcul des coordonnées barycentrique RGB et RGBXY
    #5 multiplication des matrices de coordonnées pour avoir la proportion de chaque couleur de la palette 
    pass

nameImIn = "images\colorful.ppm"
imIn = Image.open(nameImIn)
harmonization(imIn, 8)