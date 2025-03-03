from PIL import Image
from colorsys import rgb_to_hsv as r2h
from colorsys import hsv_to_rgb as h2r

R, G, B, Y, C, P, O = [0.0, 1.0/3, 2.0/3, 1.0/6, 1.0/2, 5.0/6, 1.0/12] # red, green, blue, yellow, cyan, purple, orange

nameImIn = "perso.ppm"#input("name of image : ")
nameImOut = "Test3.ppm"
colorTarget = R#input("color target : ") # 0 to 1
rayonTarget = 0.1#input("rayon of image : ") # 0 to 0.5

imIn = Image.open(nameImIn)
sx = imIn.size[0]
sy = imIn.size[1]
dataIn = imIn.load()
imOut = Image.new(imIn.mode, imIn.size)
dataOut = imOut.load()

colorBase = Y
rayonBase = 0.05 # 0 to 0.5

for i in range(sx) :
    for j in range(sy) :
        r, g, b = [c/255.0 for c in dataIn[i, j]]
        h, l, s = r2h(r, g, b)
        h2 = abs(h-colorBase)
        if (h2<rayonBase) :
            if (h2>0.5) :
                h2 = 1-h2
            if (h<colorBase) :
                h2 *= -1
            h = colorTarget + h2*rayonTarget/rayonBase
        r, g, b = [int(255*c) for c in h2r(h, l, s)]
        dataOut[i, j] = (r, g, b)

imIn.close()
imOut.save(nameImOut)
imOut.close()