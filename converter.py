import cv2
import numpy as np

def extract_contours(image_bytes, threshold1=100, threshold2=200, scale=1.0, use_canny=True, simplification=0.1, tool_diameter=0.0):
    """
    Extrait les contours d'une image PNG, applique une simplification et une compensation d'outil.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if use_canny:
        edges = cv2.Canny(blurred, threshold1, threshold2)
    else:
        _, edges = cv2.threshold(blurred, threshold1, 255, cv2.THRESH_BINARY_INV)
    
    # Compensation du diamètre de l'outil (Offset) via morphologie mathématique
    # On travaille en pixels : offset_px = (diamètre / 2) / scale
    if tool_diameter > 0:
        offset_px = int(round((tool_diameter / 2) / scale))
        if offset_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (offset_px*2+1, offset_px*2+1))
            # Pour un contour externe sur fond noir (edges), dilater agrandit le tracé vers l'extérieur
            edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = img.shape[:2]
    scaled_contours = []
    
    for cnt in contours:
        # Simplification du contour (Ramer-Douglas-Peucker)
        # epsilon est la distance max entre le contour original et le simplifié
        epsilon = simplification * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) < 2: continue
        
        scaled_cnt = approx.astype(np.float32).copy()
        # Mise à l'échelle
        scaled_cnt[:, 0, 0] *= scale # X
        scaled_cnt[:, 0, 1] *= scale # Y
        
        # Inversion axe Y (CNC)
        scaled_cnt[:, 0, 1] = (h * scale) - scaled_cnt[:, 0, 1]
        
        scaled_contours.append(scaled_cnt)
        
    return scaled_contours, img

def get_arc_params(p1, p2, p3):
    """
    Calcule le centre et le rayon du cercle passant par 3 points.
    Retourne (center_x, center_y, radius, is_clockwise)
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Méthode des médiatrices
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-6:
        return None # Points alignés
    
    center_x = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    center_y = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    
    radius = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
    
    # Déterminer le sens (produit vectoriel)
    # Vecteur AB et BC
    v1 = (x2 - x1, y2 - y1)
    v2 = (x3 - x2, y3 - y2)
    z = v1[0] * v2[1] - v1[1] * v2[0]
    is_clockwise = z < 0
    
    return (center_x, center_y, radius, is_clockwise)

def generate_gcode(contours, z_safe, z_depth, z_pass, feed_rate, feed_rate_z, use_arcs=False):
    """
    Génère le G-code avec support optionnel des arcs G2/G3.
    """
    gcode = []
    gcode.append("; G-code genere par PNG-to-Gcode Expert (Arc support)")
    gcode.append("G21 ; mm")
    gcode.append("G90 ; Abs")
    gcode.append("G94 ; mm/min")
    gcode.append("M3 S1000")
    gcode.append(f"G0 Z{z_safe:.3f}")
    
    num_passes = int(np.ceil(z_depth / z_pass)) if z_pass > 0 else 1
    
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        if len(pts) < 2: continue
        
        # Aller au point de départ
        gcode.append(f"G0 X{pts[0][0]:.3f} Y{pts[0][1]:.3f}")
        
        for p in range(1, num_passes + 1):
            current_z = -min(p * z_pass, z_depth)
            gcode.append(f"G1 Z{current_z:.3f} F{feed_rate_z:.0f}")
            
            i = 0
            while i < len(pts):
                p1 = pts[i]
                # Point suivant (bouclage si dernier point pour fermer le contour)
                idx2 = (i + 1) % len(pts)
                idx3 = (i + 2) % len(pts)
                
                p2 = pts[idx2]
                p3 = pts[idx3]
                
                arc = None
                if use_arcs and len(pts) > 2:
                    arc = get_arc_params(p1, p2, p3)
                
                if arc:
                    cx, cy, r, cw = arc
                    # On ne fait un arc que si le rayon est raisonnable (pas trop grand/infini)
                    if 0.1 < r < 1000:
                        cmd = "G2" if cw else "G3"
                        # I, J sont relatifs au point de départ (p1)
                        I = cx - p1[0]
                        J = cy - p1[1]
                        gcode.append(f"{cmd} X{p3[0]:.3f} Y{p3[1]:.3f} I{I:.3f} J{J:.3f} F{feed_rate:.0f}")
                        i += 2 # On a sauté p2
                        if i >= len(pts): break
                        continue
                
                # Sinon simple ligne G1
                gcode.append(f"G1 X{p2[0]:.3f} Y{p2[1]:.3f} F{feed_rate:.0f}")
                i += 1
            
            # Retour au point de depart pour fermer (si pas déjà fait par le dernier arc/ligne)
            gcode.append(f"G1 X{pts[0][0]:.3f} Y{pts[0][1]:.3f} F{feed_rate:.0f}")
            
        gcode.append(f"G0 Z{z_safe:.3f}")
        
    gcode.append("M5")
    gcode.append("G0 X0 Y0")
    gcode.append("M30")
    
    return "\n".join(gcode)
