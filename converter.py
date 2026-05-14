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

def generate_gcode(contours, z_safe, z_depth, z_pass, feed_rate, feed_rate_z):
    """
    Génère le G-code avec gestion multi-passes et vitesses séparées.
    """
    gcode = []
    
    # En-tête
    gcode.append("; G-code genere par PNG-to-Gcode Expert")
    gcode.append("G21 ; Unites en mm")
    gcode.append("G90 ; Positionnement absolu")
    gcode.append("G94 ; Avance en mm/min")
    gcode.append("M3 S1000 ; Broche ON")
    gcode.append(f"G0 Z{z_safe:.3f} ; Hauteur de securite")
    
    num_passes = int(np.ceil(z_depth / z_pass)) if z_pass > 0 else 1
    
    for cnt in contours:
        start_point = cnt[0][0]
        
        # Deplacement rapide au depart du contour
        gcode.append(f"G0 X{start_point[0]:.3f} Y{start_point[1]:.3f}")
        
        for p in range(1, num_passes + 1):
            current_z = -min(p * z_pass, z_depth)
            
            # Plongee en Z avec sa propre vitesse
            gcode.append(f"G1 Z{current_z:.3f} F{feed_rate_z:.0f}")
            
            # Suivi du contour
            for pt in cnt:
                x, y = pt[0]
                gcode.append(f"G1 X{x:.3f} Y{y:.3f} F{feed_rate:.0f}")
                
            # Retour au point de depart pour fermer le contour
            gcode.append(f"G1 X{start_point[0]:.3f} Y{start_point[1]:.3f} F{feed_rate:.0f}")
            
        # Remontee en Z securise apres avoir fini toutes les passes du contour
        gcode.append(f"G0 Z{z_safe:.3f}")
        
    # Pied de page
    gcode.append("M5 ; Broche OFF")
    gcode.append("G0 X0 Y0 ; Retour origine")
    gcode.append("M30 ; Fin de programme")
    
    return "\n".join(gcode)
