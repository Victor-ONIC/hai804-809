import math
from PIL import Image
import numpy as np
import scipy.spatial as spat #import ConvexHull Delaunay
from scipy.spatial import KDTree
import scipy.sparse as spar # coo_matrix

import matplotlib.pyplot as plt

maxRGB = 255

def m(v1, v2) :
    return [v1[i]-v2[i] for i in range(len(v1))]
def plus(v1, v2) :
    return [v1[i]+v2[i] for i in range(len(v1))]
def neg(v) :
    return [-v[i] for i in range(len(v))]
def s(s, v) :
    return [s*v[i] for i in range(len(v))]
def normSquare(v) :
    res = 0
    for i in v :
        res += i*i
    return res
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

def Gauss3(u, v, n, c) : # retourne (b, z), tel que xu + yv + zn = c et b = solutionExist
    i = 0
    if (u[0] == 0) :
        if (u[1] == 0) :
            i = 2
        else :
            i = 1
    j = (i+1)%3
    k = (i+2)%3
    uji = u[j]/u[i]
    vji = v[j]-v[i]*uji
    if (vji == 0) :
        j = k
        k = (i+1)%3
        uji = u[j]/u[i]
        vji = v[j]-v[i]*uji
        if (vji == 0) :
            return [False]
    uki = u[k]/u[i]
    vki = (v[k]-v[i]*uki)/vji
    nji = n[j]-n[i]*uji
    z1 = n[k]-n[i]*uki-nji*vki
    if (z1 == 0) :
        return [False]
    cji = c[j]-c[i]*uji
    return [True, (c[k] - c[i]*uki - cji*vki)/z1]
    

def loadData(imIn) :
    # récupère les données d'une image pour les mettre dans un array numpy au bon format
    nbx = imIn.size[0]
    nby = imIn.size[1]
    dataIn = imIn.load()
    ldata = [[0, 0, 0] for i in range(nbx*nby)]
    for i in range(nbx) :
        for j in range(nby) :
            ldata[i*nby+j][0], ldata[i*nby+j][1], ldata[i*nby+j][2] = dataIn[i, j]
    npdata = np.array(ldata)
    return npdata

def extractVT(hull, data) :
    v = hull.vertices
    vertices = [[p[0], p[1], p[2]] for p in data[v]]
    dico = {val : i for i, val in enumerate(v)}
    triangles = [[dico[t[0]], dico[t[1]], dico[t[2]]] for t in hull.simplices]
    return (vertices, triangles)

def nearPointSeg(a, b, c, d) :
    ba = m(b, a)
    dc = m(d, c)
    ac = m(a, c)
    badc = dot(ba, dc)
    dcac = dot(dc, ac)
    A = badc*badc
    B = badc*dcac
    C = dcac*dcac
    D = dot(ba, ba)
    E = dot(ba, ac)
    F = dot(ac, ac)
    aa = A*E - B*D
    bb = A*F - C*D
    x = [0, 1]
    im = [C/F, (A+2*B+C)/(D+2*E+F)]
    if (aa == 0) :
        if (bb != 0) :
            x0 = (B*F - C*E)/bb
            if (x0>0 and x0<1) :
                under = D*x0*x0+2*E*x0+F
                if (under != 0) :
                    x.append(x0)
                    im.append((A*x0*x0+2*B*x0+C)/under)
    else :
        delta = bb*bb - 4*aa*(B*F - C*E)
        if (delta == 0) :
            x0 = -bb/(2*aa)
            if (x0>0 and x0<1) :
                under = D*x0*x0+2*E*x0+F
                if (under != 0) :
                    x.append(x0)
                    im.append((A*x0*x0+2*B*x0+C)/under)
        if (delta > 0) :
            sd = math.sqrt(delta)
            x0 = (-bb+sd)/(2*aa)
            if (x0>0 and x0<1) :
                under0 = D*x0*x0+2*E*x0+F
                if (under0 != 0) :
                    x.append(x0)
                    im.append((A*x0*x0+2*B*x0+C)/under0)
            x1 = (-bb-sd)/(2*aa)
            if (x1>0 and x1<1) :
                under1 = D*x1*x1+2*E*x1+F
                if (under1 != 0) :
                    x.append(x1)
                    im.append((A*x1*x1+2*B*x1+C)/under1)
    index = 0
    for i in range(1, len(x)) :
        if (im[i] > im[index]) :
            index = i
    if (x[index] == 0) :
        return (a, 1-im[index])
    elif (x[index] == 1) :
        return (b, 1-im[index])
    else :
        return (plus(a, s(x[index], ba)), 1-im[index])

def reducHull2(vertices0, triangles0, targetNbr) :
    vertices = [v for v in vertices0]
    triangles = [[t[0], t[1], t[2]] for t in triangles0]
    transform = [i for i in range(len(vertices0))]
    nbVertices = len(vertices)
    trianglesPerVertices = [[] for i in range(len(vertices))]
    edges = []
    for t in range(len(triangles)) :
        for v in range(3) :
            trianglesPerVertices[triangles[t][v]].append(t)
            edge = [triangles[t][v], triangles[t][(v+1)%3]]
            if (edge[0] > edge[1]) :
                edge = [edge[1], edge[0]]
            if (not (edge in edges)) :
                edges.append(edge)
    while (nbVertices > targetNbr) :
        minDist = 10000000
        newVertex = []
        oldEdge = []
        bestId = 0
        for id, edge in enumerate(edges) :
            if (edge != []) :
                edges2 = [[], []]
                triangles2 = [[], []]
                for p in range(2) :
                    for t in trianglesPerVertices[edge[p]] :
                        triangle = triangles[t]
                        if (triangle != []) :
                            v = [0, 0, 0]
                            while (triangle[v[0]] != edge[p]) :
                                v[0] += 1
                            v[1] = (v[0]+1)%3
                            v[2] = (v[0]+2)%3
                            if (triangle[v[1]] != edge[not p] and triangle[v[2]] != edge[not p]) :
                                triangles2[p].append([triangle[v[1]], triangle[v[2]]])
                                for i in range(1, 3) :
                                    if (not (triangle[v[i]] in edges2[p])) :
                                        edges2[p].append(triangle[v[i]])
                minInter = [10000000, 10000000]
                minVertex = [[], []]
                for p in range(2) :
                    for e in edges2[p] :
                        maxInter = 0
                        n = m(vertices[e], vertices[edge[p]])
                        c = m(vertices[edge[p]], vertices[edge[not p]])
                        for triangle in triangles2[not p] :
                            u = m(vertices[edge[not p]], vertices[triangle[0]])
                            v = m(vertices[edge[not p]], vertices[triangle[1]])
                            inter = Gauss3(u, v, n, c)
                            if (inter[0]) :
                                if (inter[1]>maxInter) :
                                    maxInter = inter[1]
                        if (maxInter != 0) :
                            vertex = plus(vertices[edge[p]], s(-maxInter, n))
                            v1 = m(vertex, vertices[edge[p]])
                            v2 = neg(c)
                            temp = dot(v1, v2)
                            cos2 = temp*temp/normSquare(v2)/normSquare(v1)
                            sin2 = 1-cos2
                            if (sin2 < minInter[p] and sin2 > 0) :
                                minInter[p] = sin2
                                minVertex[p] = vertex
                nearPoint = []
                if (minVertex[0] == []) :
                    nearPoint.append(minVertex[1])
                    v1 = m(minVertex[1], vertices[edge[0]])
                    v2 = m(vertices[edge[1]], vertices[edge[0]])
                    temp = dot(v1, v2)
                    cos2 = temp*temp/normSquare(v2)/normSquare(v1)
                    sin2 = 1-cos2
                    nearPoint.append(sin2)
                else :
                    nearPoint = nearPointSeg(minVertex[0], minVertex[1], vertices[edge[0]], vertices[edge[1]])
                test = True
                for i in range(3) :
                    if (nearPoint[0][i]<0 or nearPoint[0][i]>maxRGB) :
                        test = False
                if (test) :
                    if (nearPoint[1] < minDist) :
                        minDist = nearPoint[1]
                        newVertex = nearPoint[0]
                        oldEdge = edge
                        bestId = id
        if (oldEdge == []) :
            break
        vertices[oldEdge[0]] = newVertex
        vertices[oldEdge[1]] = newVertex
        transform[oldEdge[1]] = oldEdge[0]
        edges[bestId] = []
        for i, t in enumerate(trianglesPerVertices[oldEdge[0]]) :
            if (oldEdge[0] in triangles[t]) and (oldEdge[1] in triangles[t]) :
                triangles[t] = []
                trianglesPerVertices[oldEdge[0]].pop(i)
        for i, t in enumerate(trianglesPerVertices[oldEdge[1]]) :
            if (oldEdge[0] in triangles[t]) and (oldEdge[1] in triangles[t]) :
                s0 = 0
                while (triangles[t][s0] == oldEdge[0] or triangles[t][s0] == oldEdge[1]) :
                    s0 += 1
                edge = [min(triangles[t][s0], oldEdge[1]), max(triangles[t][s0], oldEdge[1])]
                for e in range(len(edges)) :
                    if (edges[e] == edge) :
                        edges[e] = []                
                trianglesPerVertices[oldEdge[1]].pop(i)
        nbVertices -= 1

    newTriangles = []
    for t in triangles :
        if (t != []) :
            newTriangles.append(t)
            for p in range(3) :
                id = t[p]
                while (id != transform[id]) :
                    id = transform[id]
                newTriangles[-1][p] = id
    
    finalVertices = [vertices[0]]
    carte = [0 for i in range(len(vertices))]
    for i in range(1, len(vertices)) :
        carte[i] = carte[i-1]
        if (transform[i] != i) :
            carte[i] += 1
        else :
            finalVertices.append(vertices[i])
    for it in range(len(newTriangles)) :
        for p in range(3) :
            newTriangles[it][p] -= carte[newTriangles[it][p]]
    return finalVertices, newTriangles

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
            triangleNormals[i] = neg(triangleNormals[i])
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
            ldata[i*nby+j][0], ldata[i*nby+j][1], ldata[i*nby+j][2] = dataIn[i, j]
            ldata[i*nby+j][3] = i
            ldata[i*nby+j][4] = j
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

def computePalette(imIn, nbColor) :
    #0 Récupérer les données de l'image
    data = loadData(imIn)
    #1 calcul de l'enveloppe convexe RGB
    RGB_vertices = []
    oldSize = -1
    size = 0
    RGB_hull = spat.ConvexHull(data)
    RGB_vertices, RGB_triangles = extractVT(RGB_hull, data)
    while (size != oldSize) :
        RGB_hull = spat.ConvexHull(data)
        RGB_vertices, RGB_triangles = extractVT(RGB_hull, data)
        #2 calcul de la palette
        RGB_vertices, RGB_triangles = reducHull2(RGB_vertices, RGB_triangles, len(RGB_vertices)-1)
        oldSize = size
        size = len(RGB_vertices)
        data = np.array(RGB_vertices)
        print(size)
    #RGB_normals = computeNormal(RGB_vertices, RGB_triangles)
    palette = RGB_vertices #projectHull(RGB_vertices, RGB_normals)
    return palette

#basically basically just sends them to the closest thing we got
def project_onto_hull(hull_vertices, points):
    tree = KDTree(hull_vertices) #convert our valuses to a tree we can traverse
    _, indices = tree.query(points) #basically find the points in that tree
    return hull_vertices[indices] #return that

def computeW(imIn, palette) :
    #3 calcul de l'enveloppe convexe RGBXY
    XYdata = loadXYData(imIn)
    XY_hull = spat.ConvexHull(XYdata)
    XY_vertices_hull = XYdata[XY_hull.vertices]
    #4 calcul des coordonnées barycentrique RGB et RGBXY
    W_XY = Delaunay_coordinates(XY_vertices_hull, XYdata)
    print("xy \n", W_XY, "\nyo")

    #project the dawgs
    rgb_components = XY_vertices_hull[:,:3]
    projected_rgb = project_onto_hull(np.array(palette), rgb_components)

    W_RGB = Star_coordinates(np.array(palette), projected_rgb)
    print("rgb \n", W_RGB)
    #5 multiplication des matrices de coordonnées pour avoir la proportion de chaque couleur de la palette
    W = W_XY.dot(W_RGB)
    return W

def projectPalette(palette) :
    newPalette = []
    for vertex in palette :
        v = []
        for c in vertex :
            if (c>maxRGB) :
                v.append(maxRGB)
            elif (c<0) :
                v.append(0)
            else :
                v.append(c)
        newPalette.append(v)
    return newPalette

def harmonization(imIn, nbColor) :
    hull_vertex = computePalette(imIn, nbColor)
    palette = projectPalette(hull_vertex)
    paletteSize = len(palette)
    print(palette)
    for i in range(paletteSize) :
        couleur = [(palette[i][0]/maxRGB, palette[i][1]/maxRGB, palette[i][2]/maxRGB)]
        plt.scatter(i, 0, c = couleur, s=200)
    plt.show()
    W = computeW(imIn, hull_vertex)
    print(W)
    newPalette = [[0, 0, 0] for i in range(paletteSize)]
    for i in range(paletteSize) :
        # r = float(input("new color R"))
        # g = float(input("new color G"))
        # b = float(input("new color B"))
        # newPalette[i] = [r, g, b]
        newPalette[i] = palette[i]
        couleur = [(newPalette[i][0]/maxRGB, newPalette[i][1]/maxRGB, newPalette[i][2]/maxRGB)]
        plt.scatter(i, 0, c = couleur, s=200)
    plt.show()
    imOut = Image.new(imIn.mode, imIn.size)
    dataOut = imOut.load()
    nbx = imIn.size[0]
    nby = imIn.size[1]
    print(len(W))
    print(nbx*nby)
    for i in range(nbx) :
        for j in range(nby) :
            w = W[i*nby+j]
            couleur = [0, 0, 0]
            for p in range(paletteSize) :
                couleur = plus(couleur, s(w[p], newPalette[p]))
                r, g, b = couleur
            dataOut[i, j] = (int(r), int(g), int(b))
            #print(w)
    return imOut

nameImIn = "./src/images/peacock.jpg"
imIn = Image.open(nameImIn)
imOut = harmonization(imIn, 8)
imIn.close()
imOut.save("./src/images/TestPeacock.jpg")
imOut.close()