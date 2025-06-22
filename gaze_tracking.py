import os
import cv2
import math
import time
import numpy as np
import mediapipe as mp

from utils import read_text
from utils import play_audio_with_xdg_open

from config import DATA_DIR, BLINK_THRESHOLD, RIGHT_THRESHOLD, LEFT_THRESHOLD, BLACK, WHITE, BLUE, RED, GREEN, RIGHT_EYE_INNER_CORNER, RIGHT_EYE_OUTER_CORNER, LEFT_EYE_INNER_CORNER, LEFT_EYE_OUTER_CORNER, RIGHT_EYE_TOP_LID, RIGHT_EYE_BOTTOM_LID, LEFT_EYE_TOP_LID, LEFT_EYE_BOTTOM_LID, RIGHT_IRIS_POINTS, LEFT_IRIS_POINTS

def colorBackgroundText(img, text, font, fontScale, textPos, textThickness=1, textColor=(0, 255, 0), bgColor=(0, 0, 0), pad_x=3, pad_y=3):
    (t_w, t_h), _ = cv2.getTextSize(text, font, fontScale, textThickness)
    x, y = textPos
    cv2.rectangle(img, (x - pad_x, y + pad_y), (x + t_w + pad_x, y - t_h - pad_y), bgColor, -1)
    cv2.putText(img, text, textPos, font, fontScale, textColor, textThickness)
    return img

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_mesh_coordinates(img_height, img_width, results):
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        mesh_coor = [(int(p.x * img_width), int(p.y * img_height)) for p in face_landmarks]
        return mesh_coor
    return None

def get_iris_center(mesh_coor, iris_points_indices):
    iris_landmarks = np.array([mesh_coor[i] for i in iris_points_indices])
    center_x = int(np.mean(iris_landmarks[:, 0]))
    center_y = int(np.mean(iris_landmarks[:, 1]))
    return (center_x, center_y)

def is_eye_closed(mesh_coor, top_lid_idx, bottom_lid_idx, outer_corner_idx, inner_corner_idx, threshold=0.22):
    eye_top = mesh_coor[top_lid_idx]
    eye_bottom = mesh_coor[bottom_lid_idx]
    eye_outer = mesh_coor[outer_corner_idx]
    eye_inner = mesh_coor[inner_corner_idx]

    ver_dist = euclidean_distance(eye_top, eye_bottom)
    hor_dist = euclidean_distance(eye_outer, eye_inner)

    if hor_dist == 0:
        return True

    ear = ver_dist / hor_dist
    return ear < threshold

def estimate_gaze_direction(mesh_coor, iris_center, inner_corner_idx, outer_corner_idx):
    iris_x = iris_center[0]
    physical_outer_x = mesh_coor[outer_corner_idx][0]
    physical_inner_x = mesh_coor[inner_corner_idx][0]
    
    denominator = physical_inner_x - physical_outer_x
    if abs(denominator) < 1e-6:
        return "CENTER"
        
    normalized_pos = (iris_x - physical_outer_x) / denominator

    if normalized_pos > RIGHT_THRESHOLD:
        return "RIGHT"
    elif normalized_pos < LEFT_THRESHOLD:
        return "LEFT"
    else:
        return "CENTER"

def run_gaze_tracking(page_file_name): 
    
    page_file_path = read_text(os.path.join(DATA_DIR, page_file_name))
    page_dict = {}
    line = 0
    for parrafo in page_file_path.split("\n\n"):
        for linea in parrafo.replace("\n\n","\n").split("\n"):
            line +=1
            page_dict[line] = linea
    last_line = list(page_dict.keys())[-1]

    mp_face_mesh = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        exit()

    right_phase_active = False
    line_jump_detected = False
    line_count = 1
    show_first_line = True

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.7
    ) as face_mesh:
        while cap.isOpened() and line_count < last_line:
            ret, frame = cap.read()
            if not ret:
                print("No se pudo capturar el frame.")
                break

            img_h, img_w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            final_gaze_direction = "LOOKING AROUND" # Estado predeterminado

            if show_first_line:
                print(f"Reading line {line_count} -> {page_dict[line_count]}")
                play_audio_with_xdg_open(line_count)
                show_first_line = False

            if results.multi_face_landmarks:
                mesh_coor = get_mesh_coordinates(img_h, img_w, results)

                if mesh_coor:
                    right_eye_closed = is_eye_closed(mesh_coor, RIGHT_EYE_TOP_LID, RIGHT_EYE_BOTTOM_LID,
                                                    RIGHT_EYE_OUTER_CORNER, RIGHT_EYE_INNER_CORNER, BLINK_THRESHOLD)
        
                    left_eye_closed = is_eye_closed(mesh_coor, LEFT_EYE_TOP_LID, LEFT_EYE_BOTTOM_LID,
                                                    LEFT_EYE_OUTER_CORNER, LEFT_EYE_INNER_CORNER, BLINK_THRESHOLD)

                    # Si ambos ojos no están cerrados, intenta determinar la dirección
                    if not right_eye_closed and not left_eye_closed:
                        iris_center_r = get_iris_center(mesh_coor, RIGHT_IRIS_POINTS)
                        iris_center_l = get_iris_center(mesh_coor, LEFT_IRIS_POINTS)

                        gaze_right_eye = estimate_gaze_direction(mesh_coor, iris_center_r,
                                                                RIGHT_EYE_INNER_CORNER, RIGHT_EYE_OUTER_CORNER)
                        gaze_left_eye = estimate_gaze_direction(mesh_coor, iris_center_l,
                                                                LEFT_EYE_INNER_CORNER, LEFT_EYE_OUTER_CORNER)

                        # Determinar la dirección combinada, priorizando LEFT/RIGHT si ambos coinciden o uno es claro
                        if gaze_right_eye == "RIGHT" and gaze_left_eye == "RIGHT":
                            final_gaze_direction = "RIGHT"
                            if not right_phase_active:
                                right_phase_active = True
                                line_jump_detected = False  # Se permite detectar nuevo salto
                        elif gaze_right_eye == "LEFT" and gaze_left_eye == "LEFT":
                            final_gaze_direction = "LEFT"
                            if right_phase_active and not line_jump_detected:
                                line_count += 1
                                # print("Salto de línea detectado. Total líneas leídas:", line_count)
                                print(f"Reading line {line_count} -> {page_dict[line_count]}")
                                play_audio_with_xdg_open(line_count)
                                line_jump_detected = True
                                right_phase_active = False  # Reset para esperar nuevo ciclo
                        else:
                            final_gaze_direction = "LOOKING AROUND"
                    elif right_eye_closed and left_eye_closed:
                        final_gaze_direction = "CLOSED" # Ambos ojos cerrados
                    elif right_eye_closed or left_eye_closed:
                        final_gaze_direction = "BLINKING" # Un ojo cerrado (posible parpadeo)
            
            text_color = WHITE
            bg_color = BLACK
            if final_gaze_direction == "RIGHT":
                text_color = GREEN
                bg_color = BLACK
            elif final_gaze_direction == "LEFT":
                text_color = RED
                bg_color = BLACK
            elif final_gaze_direction == "CLOSED" or final_gaze_direction == "BLINKING":
                text_color = WHITE
                bg_color = BLUE
            else:
                text_color = WHITE
                bg_color = BLACK
                
            colorBackgroundText(frame, final_gaze_direction, cv2.FONT_HERSHEY_COMPLEX, 1, (img_w // 2 - 100, 50), 2, text_color, bg_color, 10, 10)

            time.sleep(0.1)
            cv2.imshow("Eye Gaze Tracking", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    time.sleep(2)
    print("La lectura del texto ha finalizado.")