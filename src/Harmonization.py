## Harmonization.py
# Color Harmonization based on Cohen-Or et al. 2006

import math
from PIL import Image
from utils.Histogramme import Histogramme
from colorsys import rgb_to_hsv as r2h
from colorsys import hsv_to_rgb as h2r
from utils.Modele import Modele

def to_degrees(alpha):
    return alpha * 180 / math.pi

def to_radians(alpha):
    return alpha * math.pi / 180

#returns the closest edge to the hue, the border and the center of the sector
def assign_closest_sector_index(hue, model: Modele):
    min_distance = float('inf')
    closest_edge_index = None
    
    for i in range(len(model.C)):
        distance = model.distance_secteur(hue, i)
        if distance == -1: 
            return i
        if distance < min_distance:
            min_distance = distance
            closest_edge_index = i

    return closest_edge_index

def assign_closest_edge(hue, model: Modele):
    sector_index = assign_closest_sector_index(hue, model)
    bord = model.bord(sector_index)
    closest_edge = -float("inf")
    dist1 = Modele.distance_congru(hue , bord[0])
    dist2 = Modele.distance_congru(hue , bord[1])

    if(dist1 < dist2):
        return bord[1]
    else : 
        return bord[0]
    
def assign_closest_edge_opti(hue, model: Modele , i , j):
    sector_index = assign_closest_sector_index_optimized(hue, model , i , j)
    bord = model.bord(sector_index)
    closest_edge = -float("inf")
    dist1 = Modele.distance_congru(hue , bord[0])
    dist2 = Modele.distance_congru(hue , bord[1])

    if(dist1 < dist2):
        return bord[1]
    else : 
        return bord[0]


#FIXME : just does not really work
def assign_closest_sector_index_optimized(h , model: Modele, image, i, j):
    pixels = get_neighboring_pixels(image, i, j)  # Get neighbors once
    omega = get_omega(pixels, image)  

    min_energy = float('inf')
    closest_sector_index = None

    for idx in range(len(model.C)):  
        central_hue = model.C[idx]

        #Deviation from Central Hue
        E1 = 0
        for hn, sn , _ in omega : 
            hvp = project_hue(central_hue , model.w[idx]/2 , hn)
            E1 += Modele.distance_congru(hn,hvp)  * sn 


        # ---- Compute E2: Local Contrast Preservation ----
        E2 = 0
        for p in range(len(omega)):
            for q in range(len(omega)):
                vp = project_hue(central_hue , model.w[idx]/2, omega[p][0])
                vq = project_hue(central_hue , model.w[idx]/2, omega[q][0])

                if vp != vq :
                    #recalculate v(p) and v(q) and do the epsilon check
                    h_p, s_p = (omega[p][0],omega[p][1])
                    h_q, s_q = (omega[q][0],omega[q][1])
    
                    Smax_pq = max(s_p, s_q)
                    hue_diff = Modele.distance_congru(h_p , h_q)  # Prevent division by zero
                    E2 += Smax_pq * (1 / hue_diff) 
 
        # ---- Total Energy ----
        E = E1 + E2

        # ---- Select the Best Sector ----
        if E < min_energy:  
            min_energy = E
            closest_sector_index = idx

    return closest_sector_index




def get_neighboring_pixels(i, j):
    pixels = []
    for x in range(i-1, i+1):
        for y in range(j-1, j+1):
            pixels.append([x, y])
    
    return pixels

def get_omega(pixels, image):
    omega = []
    #fill omega with the hue of the pixels
    for p in pixels:
        r, g, b = image.getpixel((p[0], p[1]))
        h, s, v = r2h(r / 255.0, g / 255.0, b / 255.0)
        omega.append((h, s , v))
    return omega


#F(X,(m,alpha)) = sum ||H(X) - E_tm(alpha)|| * S(X) in the paper
def harmony_by_template(image, model: Modele):
    sum_harmony_value = 0
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            r, g, b = image.getpixel((i, j))
            hue , saturation , _ = r2h(r / 255.0, g / 255.0, b / 255.0)
            #print("closest_edge_index : ", closest_edge_index)
            E = assign_closest_edge(hue , model) #old
            #E = assign_closest_edge_opti(hue , model , i , j) #new does not work
            #print("E : ", E)
            sum_harmony_value += Modele.distance_congru(hue , E) * saturation  # Use closest edge

    return sum_harmony_value

#M(X , Tm) = (m , alpha0) avec alpha0 = argmin alpha F(X , (m,alpha)) in the paper
def best_angle_radians(image, template):
    best_alpha = 0
    best_harmony_value = -float('inf')
    
    original_C = template.C[:] 
    
    for alpha in [to_radians(a) for a in range(0, 360, 10)]:
        template.radRotate(alpha) 
        harmony_value = harmony_by_template(image, template)
        
        if harmony_value > best_harmony_value:
            best_harmony_value = harmony_value
            best_alpha = alpha
    
        template.C = original_C[:]  # Restore original template

    return best_alpha
#B(X) = (m0 , alpha0) avec m0 = argmin m M(X , Tm) in the paper
def best_template(image):
    best_harmony_value = 0.0
    best_template = None
    best_alpha = None
    for template_name in Modele.get_liste_modeles():
        try:
            template = Modele(template_name)
            alpha = best_angle_radians(image, template)
            harmony_value = harmony_by_template(image, template)
            
            # Ensure harmony_value is a valid number
            if not isinstance(harmony_value, (int, float)) or math.isnan(harmony_value) or math.isinf(harmony_value):
                raise ValueError(f"Invalid harmony value: {harmony_value}")
            
            if harmony_value > best_harmony_value:
                best_harmony_value = harmony_value
                best_template = template
                best_alpha = alpha
        except Exception as e:
            print(f"Error processing template {template_name}: {e}")
            continue

    if best_template is None or best_alpha is None:
        raise ValueError("No valid template or angle found for the given image.")

    return best_template, best_alpha

#d = H - C (idk what c is yet but tkt)
#w = the width of a zone

#Esperance = C
#Variance = w/2

def project_hue(center , variance , h):
    d = Modele.distance_congru(h, center)
    new_h = center + (variance) * (1 - math.exp(-1/2 * (( (d) ** 2) / (variance ** 2))) / math.sqrt(variance*2*math.pi))
    new_h = Modele.congru(new_h) 
    return new_h


#H'(X) = C(p) + (w/2) * (1 - Gaussian_w/2(||H(p)-C(p))) in the paper
def harmonize(image, imageOut, template: Modele, alpha):
    template.radRotate(alpha)
    
    image = image.convert("RGB")
    width, height = image.size
    pixels = image.load()
    pixels_out = imageOut.load()

    for i in range(width):
        for j in range(height):
            r, g, b = pixels[i, j]
            h, s, v = r2h(r / 255.0, g / 255.0, b / 255.0)
            closest_sector_index = assign_closest_sector_index(h, template)
            central_hue = template.C[closest_sector_index]
            width = template.w[closest_sector_index]
            if template.distance_secteur(h, closest_sector_index) == -1:
                new_h = h
            else:
                new_h = project_hue(central_hue , width/2 , h)

            new_r, new_g, new_b = [int(c * 255) for c in h2r(new_h, s, v)]
            pixels_out[i, j] = (new_r, new_g, new_b)

    return imageOut

#FIXME : optimized edge detection is fraudulent too
def harmonize_opti(image , imageOut , template : Modele , alpha):
    # Step 1: Rotate the template
    template.radRotate(alpha)
    
    # Step 2: Convert image to HSV
    image = image.convert("RGB")
    width, height = image.size
    pixels = image.load()
    pixels_out = imageOut.load()

    # Step 3: Apply harmonization
    for i in range(width):
        for j in range(height):
            r, g, b = pixels[i, j]
            h, s, v = r2h(r / 255.0, g / 255.0, b / 255.0)
            closest_edge_index = assign_closest_sector_index_optimized(h , template , image , i , j)
            central_hue = template.C[closest_edge_index]
            width = template.w[closest_edge_index]
            if template.distance_secteur(h, closest_edge_index) == -1:
                new_h = h
            else:
                # Apply the formula for color adjustment
                new_h = project_hue(central_hue , width/2 , h)

                # Convert back to RGB
                new_r, new_g, new_b = [int(c * 255) for c in h2r(new_h, s, v)]
                pixels_out[i, j] = (new_r, new_g, new_b)

    return imageOut

#TODO : when it works, implement the optimized edge detection
def harmonize_auto_angle(image , imageOut , template : Modele):
    alpha = best_angle_radians(image, template)
    print("Best angle : ", to_degrees(alpha))
    return harmonize(image, imageOut, template, alpha)

def harmonize_auto(image , imageOut):
    template , alpha = best_template(image)
    print("Best template : ", template.type)
    print("Best angle : ", to_degrees(alpha))
    return harmonize(image, imageOut, template, alpha)

#--------------------------------------------------tests-----------------------------------------------------------
nameImage = "./src/images/colorful.ppm"
im = Image.open(nameImage)
imOut = Image.new(im.mode, im.size)



template = Modele("I")


print("Harmonization with manual angle and template")
harmonize(im, imOut, template, to_radians(45))
imOut.save("./src/images/harmonized_manual_angle2.ppm")

print("Harmonization opti")
imOut2 = Image.new(im.mode, im.size)
harmonize_opti(im, imOut2, template, to_radians(45))
imOut2.save("./src/images/harmonized_opti2.ppm")


# print("Harmonization with auto angle")
# harmonize_auto_angle(im, imOut, template)
# imOut.save("./src/images/harmonized_auto_angle2.ppm")

# print("Harmonization with auto template and angle")
# imOut2 = Image.new(im.mode, im.size)
# harmonize_auto(im, imOut2)
# imOut2.save("./src/images/harmonized_auto2.ppm")