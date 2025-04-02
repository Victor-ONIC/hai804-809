import math
from PIL import Image
import scipy.spatial as ss #import ConvexHull
import numpy as np

def m(v1, v2) :
    return [v1[i]-v2[i] for i in range(len(v1))]
def vect(v1, v2) :
    t = len(v1)
    return [v1[(i+1)%t]*v2[(i+2)%t] - v1[(i+2)%t]*v2[(i+1)%t] for i in range(t)]
def normalize(v) :
    l = 0
    for i in v :
        l += i**2
    l = math.sqrt(l)
    return [v[i]/l for i in range(len(v))]
def dot(v1, v2) :
    res = 0
    for i in range(len(v1)) :
        res += v1[i]*v2[i]
    return res
def n(v) :
    return [-v[i] for i in range(len(v))]
def p(v1, v2) :
    return [v1[i]+v2[i] for i in range(len(v1))]

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
    transform = [i for i in range(len(vertices))]
    currentNbr = len(vertices)
    verticesPacks = [[v] for v in vertices]
    newVertices = [v for v in vertices]
    
    # fusionne points
    while (currentNbr > targetNbr) :
        min = float('inf')
        i0 = 0
        i1 = 1
        for i in range(len(vertices)-1) :
            for j in range(i+1, len(vertices)) :
                d = distance(newVertices[i], newVertices[j])
                if (d < min) :
                    min = d
                    i0, i1 = (i, j)
        verticesPacks[i0] += verticesPacks[i1]
        verticesPacks[i1] = []
        newVertices[i0] = moyenne(verticesPacks[i0])
        transform[i1] = i0
        currentNbr -= 1
    
    # retire les triangles inutiles et réassigne aux nouveaux points
    it = 0
    while (it < len(triangles)) :
        for p in range(3) :
            id = triangles[it][p]
            while (id != transform[id]) :
                id = transform[id]
            triangles[it][p] = id
        if (triangles[it][0] == triangles[it][1] or triangles[it][0] == triangles[it][2] or triangles[it][1] == triangles[it][2]) :
            triangles.pop(it)
        else :
            it += 1
    
    # réindex les points et les triangles
    V = [newVertices[0]]
    carte = [0 for i in range(len(vertices))]
    for i in range(1, len(vertices)) :
        carte[i] = carte[i-1]
        if (verticesPacks[i] == []) :
            carte[i] += 1
        else :
            V.append(newVertices[i])
    for it in range(len(triangles)) :
        for p in range(3) :
            triangles[it][p] -= carte[triangles[it][p]]
    return V, triangles

def computeNormal(v, t) :
    triangleNormals = [0 for i in range(len(t))]
    vertexNormals = [[0, 0, 0] for i in range(len(v))]
    nbNormals = [0 for i in range(len(v))]
    for i in range(len(t)) :
        # calcul la normal du triangle
        v1 = m(v[t[i][2]], v[t[i][0]])
        v2 = m(v[t[i][1]], v[t[i][0]])
        triangleNormals[i] = normalize(vect(v1, v2))
        # sens de la normal
        p = v[0]
        if (t[i][0] == 0 or t[i][1] == 0 or t[i][2] == 0) :
            p = v[1]
            if (t[i][0] == 1 or t[i][1] == 1 or t[i][2] == 1) :
                p = v[2]
                if (t[i][0] == 2 or t[i][1] == 2 or t[i][2] == 2) :
                    p = v[3]
        p = normalize(m(v[t[i][0]], p))
        if dot(triangleNormals[i], p) < 0 :
            triangleNormals[i] = n(triangleNormals[i])
        # calcul la normal des points
        for j in range(3) :
            vertexNormals[t[i][j]] = p(vertexNormals[t[i][j]], triangleNormals[i])
            nbNormals += 1
    for i in range(len(vertexNormals)) :
        for j in range(3) :
            vertexNormals[i][j] /= nbNormals[i]
    return vertexNormals
        

def harmonization(imIn, nbColor) :
    #0 Récupérer les données de l'image
    data = loadData(imIn)
    #1 calcul de l'enveloppe convexe RGB
    RGB_hull = ss.ConvexHull(data)
    RGB_vertices, RGB_triangles = extractVT(RGB_hull, data)
    #2 calcul de la palette
    RGB_vertices, RGB_triangles = reducHull(RGB_vertices, RGB_triangles, nbColor)
    RGB_normals = computeNormal(RGB_vertices, RGB_triangles)
    #2.5 étirer les points pour réenvelopper ce qui est sorti
    #3 calcul de l'enveloppe convexe RGBXY
    #4 calcul des coordonnées barycentrique RGB et RGBXY
    #5 multiplication des matrices de coordonnées pour avoir la proportion de chaque couleur de la palette 
    pass

nameImIn = "images\colorful.ppm"
imIn = Image.open(nameImIn)
harmonization(imIn, 8)