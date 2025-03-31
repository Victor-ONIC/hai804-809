## Harmonization.py
# Color Harmonization based on Cohen-Or et al. 2006

import math
from PIL import Image
from utils.Histogramme import Histogramme
from colorsys import rgb_to_hsv as r2h
from colorsys import hsv_to_rgb as h2r
from utils.Modele import Modele
import time


############################################################## Utils ################################################################################""

def to_degrees(alpha):
    '''
        Radians to degrees
    '''
    return alpha * 180 / math.pi


def to_radians(alpha):
    '''
        Degrees to radians
    '''
    return alpha * math.pi / 180

############################################################## Closest border n dem ###################################################################

def assign_closest_sector_index(hue, model: Modele):
    ''' 
        Return the center of the closest edge, us just calculating distance.
    '''
    
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
    '''
        Returns the closest border based on the closest edge
    '''

    sector_index = assign_closest_sector_index(hue, model)
    if sector_index is None:
        return hue 

    bord = model.bord(sector_index)

    dist1 = Modele.distance_congru(hue , bord[0])
    dist2 = Modele.distance_congru(hue , bord[1])
    if(dist1 < dist2):
        return bord[1]
    else : 
        return bord[0]

 
def assign_closest_edge_opti(hue, image , model: Modele , i , j):
    '''
        Returns the closest border based on the closest edge
    '''

    sector_index = assign_closest_sector_index_optimized(model , image , i , j)
    if sector_index is None:
        return hue 
    
    bord = model.bord(sector_index)

    dist1 = Modele.distance_congru(hue , bord[0])
    dist2 = Modele.distance_congru(hue , bord[1])
    if(dist1 < dist2):
        return bord[1]
    else : 
        return bord[0]


def assign_closest_sector_index_optimized(model: Modele, image, i, j):
    '''
        Return the center of the closest edge, using the Energy calculation.

        After tests i realized that E2 is a fraud so i clocked it (E2 ended up yielding the same results as before optimization but with
        much longer computing times)
    '''


    pixels = get_neighboring_pixels(image, i, j) 
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



        #E2 is long to calculate, and seems useless in the thing

        # E2 = 0
        # for p in range(len(omega)):
        #     for q in range(len(omega)):
        #         vp = project_hue(central_hue , model.w[idx]/2, omega[p][0])
        #         vq = project_hue(central_hue , model.w[idx]/2, omega[q][0])

        #         if vp != vq :
        #             #recalculate v(p) and v(q) and do the epsilon check
        #             h_p, s_p = (omega[p][0],omega[p][1])
        #             h_q, s_q = (omega[q][0],omega[q][1])

        #             Smax_pq = max(s_p, s_q)
        #             epsilon = 1e-6
        #             hue_diff = max(epsilon, Modele.distance_congru(h_p , h_q))
        #             E2 += Smax_pq * (1 / hue_diff)

 
        #Multiply by a constant and add E2 if you want to try the real formula
        E = E1

        if E < min_energy:  
            min_energy = E
            closest_sector_index = idx

    return closest_sector_index


############################################################## Neigboring ###############################################################################

def get_neighboring_pixels(image, i, j):
    '''
        Returns a list of the neighboring pixels indexes
    '''
    
    width, height = image.size
    pixels = []

    for x in range(max(0, i - 1), min(width, i + 2)):
        for y in range(max(0, j - 1), min(height, j + 2)):
            if (x, y) != (i, j):  # Exclude the central pixel
                pixels.append((x, y))

    return pixels


def get_omega(pixels, image):
    '''
        Returns the hue values (list) of the neighboring pixels given their indexes 
    '''
    omega = []
    for p in pixels:
        r, g, b = image.getpixel((p[0], p[1]))
        h, s, v = r2h(r / 255.0, g / 255.0, b / 255.0)
        omega.append((h, s , v))
    return omega

############################################################# Harmony utils ##############################################################################

def harmony_by_template(image, model: Modele):
    '''
        Calculate the harmony value of the image. This is a sort of arbitrary value I guess, but it's in the paper.
    '''
    
    sum_harmony_value = 0
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            r, g, b = image.getpixel((i, j))
            hue , saturation , _ = r2h(r / 255.0, g / 255.0, b / 255.0)
            #E = assign_closest_edge(hue , model) #old
            E = assign_closest_edge_opti(hue , image , model , i , j) #new
            sum_harmony_value += Modele.distance_congru(hue , E) * saturation  # Use closest edge

    return sum_harmony_value


def best_angle_radians(image, template):
    '''
        Iterate over the given template at different angles and return the best one
    '''
    
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


def best_template(image):
    '''
        Iterate over each template and angle for each template to get the highest harmony level.
    '''

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


def project_hue(center , variance , h):
    '''
        Project the given hue according to the projection formula in da paper
    '''

    d = Modele.distance_congru(h, center)
    new_h = center + (variance) * (1 - math.exp(-1/2 * (( (d) ** 2) / (variance ** 2))) / math.sqrt(variance*2*math.pi))
    new_h = Modele.congru(new_h) 
    return new_h


############################################################# Different harmonization methods ########################################################

def harmonize(image, imageOut, template: Modele, alpha):
    ''' 
        Harmonize a given image with a manually set template and angle.
        This uses the naive one that just sends pixels with no regard to neighboring pixels
    '''

    #Rotate the template
    template.radRotate(alpha)
    
    #Convert to HSV
    image = image.convert("RGB")
    width, height = image.size
    pixels = image.load()
    pixels_out = imageOut.load()

    #Harmonization
    for i in range(width):
        for j in range(height):
            r, g, b = pixels[i, j]
            h, s, v = r2h(r / 255.0, g / 255.0, b / 255.0)
            closest_sector_index = assign_closest_sector_index(h, template)
            central_hue = template.C[closest_sector_index]
            sector_width = template.w[closest_sector_index]

            #If already in the template, keep it
            if template.distance_secteur(h, closest_sector_index) == -1:
                new_h = h
            else:
                new_h = project_hue(central_hue , sector_width/2 , h)

            new_r, new_g, new_b = [int(c * 255) for c in h2r(new_h, s, v)] #Convert back to RGB
            pixels_out[i, j] = (new_r, new_g, new_b)

    return imageOut


def harmonize_opti(image , imageOut , template : Modele , alpha):
    ''' 
        Harmonize a given image with a manually set template and angle.
        This uses the optimized algorithm that fixes neighboring pb
    '''

    #Rotate the template
    template.radRotate(alpha)
    
    #Convert to HSV
    image = image.convert("RGB")
    width, height = image.size
    pixels = image.load()
    pixels_out = imageOut.load()

    #Harmonization
    for i in range(width):
        for j in range(height):
            r, g, b = pixels[i, j]
            h, s, v = r2h(r / 255.0, g / 255.0, b / 255.0)
            closest_edge_index = assign_closest_sector_index_optimized(template , image , i , j)
            central_hue = template.C[closest_edge_index]
            sector_width = template.w[closest_edge_index]

            #If already in the template, keep it
            if template.distance_secteur(h, closest_edge_index) == -1:
                new_h = h
                
            else:
                new_h = project_hue(central_hue , sector_width/2 , h)
            
            new_r, new_g, new_b = [int(c * 255) for c in h2r(new_h, s, v)] #Convert back to RGB
            pixels_out[i, j] = (new_r, new_g, new_b)

    return imageOut


def harmonize_auto_angle(image , imageOut , template : Modele):
    ''' 
        Harmonize a given image with a manually set template.
        Automatically calculate the angle
    '''

    alpha = best_angle_radians(image, template)

    print("Best angle : ", to_degrees(alpha))

    return harmonize_opti(image, imageOut, template, alpha)


def harmonize_auto(image , imageOut):
    ''' 
        Harmonize a given image. Automatically calculate the best angle and template.
    '''

    template , alpha = best_template(image)

    print("Best template : ", template.type)
    print("Best angle : ", to_degrees(alpha))

    return harmonize(image, imageOut, template, alpha)

############################################################ Tests ########################################################################################

#Opening the image
nameImage = "./src/images/30.jpg"
im = Image.open(nameImage)
imOut = Image.new(im.mode, im.size)

template = Modele("I")

print("Harmonization opti")
# Measure time for harmonize_opti
start_time = time.time()
imOut2 = Image.new(im.mode, im.size)
harmonize_opti(im, imOut2, template, to_radians(45))
imOut2.save("./src/images/30on30HSV.jpg")
print("Time for harmonize_opti:", time.time() - start_time, "seconds")

# Measure time for harmonize_auto_angle
start_time = time.time()
imOut4 = Image.new(im.mode, im.size)
harmonize_auto_angle(im, imOut4, template)
imOut4.save("./src/images/30on30AutoAngle.jpg")
print("Time for harmonize_auto_angle:", time.time() - start_time, "seconds")

# Measure time for harmonize_auto
start_time = time.time()
imOut3 = Image.new(im.mode, im.size)
harmonize_auto(im, imOut3)
imOut3.save("./src/images/30on30Auto.jpg")
print("Time for harmonize_auto:", time.time() - start_time, "seconds")

