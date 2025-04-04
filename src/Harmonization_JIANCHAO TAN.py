import math
from PIL import Image
import numpy as np
import scipy.spatial as spat #import ConvexHull Delaunay
import scipy.sparse as spar # coo_matrix

import matplotlib.pyplot as plt

maxRGB = 255

def m(v1, v2) :
    return [v1[i]-v2[i] for i in range(len(v1))]
def plus(v1, v2) :
    return [v1[i]+v2[i] for i in range(len(v1))]
def n(v) :
    return [-v[i] for i in range(len(v))]
def s(s, v) :
    return [s*v[i] for i in range(len(v))]
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
def distance(v1, v2) :
    res = 0
    for i in range(len(v1)) :
        res += (v1[i]-v2[i])**2
    return math.sqrt(res)
def moyenne(vertices) :
    res = [0 for i in range(len(vertices[0]))]
    for vertex in vertices :
        res = plus(res, vertex)
    for i in range(len(res)) :
        res[i] /= len(vertices)
    return res

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
    vertices = [[p[0], p[1], p[2]] for p in data[v]]
    dico = {val : i for i, val in enumerate(v)}
    triangles = [[dico[t[0]], dico[t[1]], dico[t[2]]] for t in hull.simplices]
    return (vertices, triangles)

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
            if verticesPacks[i] != [] :
                for j in range(i+1, len(vertices)) :
                    if verticesPacks[j] != [] :
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
    finalVertices = [newVertices[0]]
    carte = [0 for i in range(len(vertices))]
    for i in range(1, len(vertices)) :
        carte[i] = carte[i-1]
        if (verticesPacks[i] == []) :
            carte[i] += 1
        else :
            finalVertices.append(newVertices[i])
    for it in range(len(triangles)) :
        for p in range(3) :
            triangles[it][p] -= carte[triangles[it][p]]
    return finalVertices, triangles

def computeNormal(v, t) :
    triangleNormals = [[0, 0, 0] for i in range(len(t))]
    vertexNormals = [[0, 0, 0] for i in range(len(v))]
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
            vertexNormals[t[i][j]] = plus(vertexNormals[t[i][j]], triangleNormals[i])
    for i in range(len(vertexNormals)) :
        vertexNormals[i] = normalize(vertexNormals[i])
    return vertexNormals

def projectHull(v, n) :
    newHull = [[0, 0, 0] for i in range(len(v))]
    for i in range(len(v)) :
        min = float('inf')
        for d in range(3) :
            if n[i][d] != 0 :
                distance = max((maxRGB-v[i][d])/n[i][d], (0-v[i][d])/n[i][d]) # x = (m-v)/n
                if distance < min :
                    min = distance
        newHull[i] = plus(v[i], s(min, n[i]))
    return newHull

def loadXYData(imIn) :
    # récupère les données d'une image pour les mettre dans un array numpy au bon format
    nbx = imIn.size[0]
    nby = imIn.size[1]
    dataIn = imIn.load()
    ldata = [[0, 0, 0, 0, 0] for i in range(nbx*nby)]
    for i in range(nbx) :
        for j in range(nby) :
            ldata[j*nbx+i][0], ldata[j*nbx+i][1], ldata[j*nbx+i][2] = dataIn[i, j]
            ldata[j*nbx+i][3] = i
            ldata[j*nbx+i][4] = j
    npdata = np.array(ldata)
    return npdata

def Delaunay_coordinates(vertices, data) : # Adapted from Gareth Rees
    # Compute Delaunay tessellation.
    tri = spat.Delaunay(vertices)
    # Find the tetrahedron containing each target (or -1 if not found).
    simplices = tri.find_simplex(data, tol=1e-6)
    assert (simplices != -1).all() # data contains outside vertices.
    # Affine transformation for simplex containing each datum.
    X = tri.transform[simplices, :data.shape[1]]
    # Offset of each datum from the origin of its simplex.
    Y = data - tri.transform[simplices, data.shape[1]]
    # Compute the barycentric coordinates of each datum in its simplex.
    b = np.einsum('...jk,...k->...j', X, Y)
    barycoords = np.c_[b,1-b.sum(axis=1)]
    # Return the weights as a sparse matrix.
    rows = np.repeat(np.arange(len(data)).reshape((-1,1)), len(tri.simplices[0]), 1).ravel()
    cols = tri.simplices[simplices].ravel()
    vals = barycoords.ravel()
    return spar.coo_matrix((vals, (rows, cols)), shape=(len(data), len(vertices))).tocsr()

def Star_coordinates(vertices, data) :
    # Find the star vertex
    star = np.argmin(np.linalg.norm(vertices, axis=1))
    # Make a mesh for the palette
    hull = spat.ConvexHull(vertices)
    # Star tessellate the faces of the convex hull
    simplices = [[star] + list(face) for face in hull.simplices if star not in face]
    barycoords = -1*np.ones((data.shape[0], len(vertices)))
    # Barycentric coordinates for the data in each simplex
    for s in simplices:
        s0 = vertices[s[:1]]
        b = np.linalg.solve((vertices[s[1:]]-s0).T, (data-s0).T).T
        b = np.append(1-b.sum(axis=1)[:,None], b, axis=1)
        # Update barycoords whenever the data is inside the current simplex.
        mask = (b>=0).all(axis=1)
        barycoords[mask] = 0.
        barycoords[np.ix_(mask,s)] = b[mask]
    return barycoords

def harmonization(imIn, nbColor) :
    #0 Récupérer les données de l'image
    data = loadData(imIn)
    #1 calcul de l'enveloppe convexe RGB
    RGB_hull = spat.ConvexHull(data)
    RGB_vertices, RGB_triangles = extractVT(RGB_hull, data)
    #2 calcul de la palette
    RGB_vertices, RGB_triangles = reducHull(RGB_vertices, RGB_triangles, nbColor)
    RGB_normals = computeNormal(RGB_vertices, RGB_triangles)
    palette = projectHull(RGB_vertices, RGB_normals)
    print(palette)
    for i in range(nbColor) :
        couleur = [(palette[i][0]/maxRGB, palette[i][1]/maxRGB, palette[i][2]/maxRGB)]
        plt.scatter(i, 0, c = couleur, s=200)
    plt.show()
    #3 calcul de l'enveloppe convexe RGBXY
    XYdata = loadXYData(imIn)
    XY_hull = spat.ConvexHull(XYdata)
    XY_vertices = XYdata[XY_hull.vertices]
    #4 calcul des coordonnées barycentrique RGB et RGBXY
    W_XY = Delaunay_coordinates(XY_vertices, XYdata)
    W_RGB = Star_coordinates(np.array(palette), XY_vertices[:,:3])
    #5 multiplication des matrices de coordonnées pour avoir la proportion de chaque couleur de la palette
    W = W_XY.dot( W_RGB )
    newPalette = [[0, 0, 0] for i in range(nbColor)]
    for i in range(nbColor) :
        r = float(input("new color R"))
        g = float(input("new color G"))
        b = float(input("new color B"))
        newPalette[i] = [r, g, b]
        couleur = [(newPalette[i][0]/maxRGB, newPalette[i][1]/maxRGB, newPalette[i][2]/maxRGB)]
        plt.scatter(i, 0, c = couleur, s=200)
    plt.show()

nameImIn = "images\colorful.ppm"
imIn = Image.open(nameImIn)
harmonization(imIn, 4)