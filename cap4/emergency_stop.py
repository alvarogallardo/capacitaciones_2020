#!/usr/bin/env python

"""
Este programa implementa un freno de emergencia para evitar accidentes en Duckietown.
"""

import sys
import argparse
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
import numpy as np
import cv2
lower_yellow = np.array([20, 210, 160])  #20 210 160
upper_yellow = np.array([60, 255, 255])  #60 255  255
min_area = 2500
def mov_duckiebot(key):
    # La acción de Duckiebot consiste en dos valores:
    # velocidad lineal y velocidad de giro
    actions = {ord('w'): np.array([1.0, 0.0]),
               ord('s'): np.array([-1.0, 0.0]),
               ord('a'): np.array([0.0, 1.0]),
               ord('d'): np.array([0.0, -1.0]),
               ord('q'): np.array([0.3, 1.0]),
               ord('e'): np.array([0.3, -1.0])
               }

    action = actions.get(key, np.array([0.0, 0.0]))
    return action
def det_duckie(obs):
    ### DETECTOR HECHO EN LA MISI�N ANTERIOR
    image_out = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(image_out, lower_yellow, upper_yellow)
    img_out = cv2.bitwise_and(image_out, image_out, mask = mask)
    kernel = np.ones((5,5),np.uint8)
    img_out = cv2.erode(img_out, kernel, iterations = 1)
    img_out = cv2.dilate(img_out, kernel, iterations =1)
    #opening = cv2.morphologyEx(img_out, cv2.MORPH_OPEN, kernel)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dets = list()
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        AREA = cv2.contourArea(cnt)
        if AREA > min_area:
            # En lugar de dibujar, se agrega a la lista
            dets.append((x,y,w,h))
    return dets
def draw_dets(obs, dets):
    for d in dets:
        x1, y1 = d[0], d[1]
        x2 = x1 + d[2]
        y2 = y1 + d[3]
        cv2.rectangle(obs, (int(x1), int(y1)), (int(x2),int(y2)), (0,255,0), 3)
    return obs
def red_alert(obs):
    red_img = np.zeros((480, 640, 3), dtype = np.uint8)
    red_img[:,:,0] = 255
    blend = cv2.addWeighted(obs, 0.5, red_img, 0.5, 0)
    return blend
if __name__ == '__main__':
    # Se leen los argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default="Duckietown-udem1-v1")
    parser.add_argument('--map-name', default='free')
    args = parser.parse_args()
    # Definición del environment
    if args.env_name and args.env_name.find('Duckietown') != -1:
        env = DuckietownEnv(
            map_name = args.map_name,
            domain_rand = False,
        )
    else:
        env = gym.make(args.env_name)
    # Se reinicia el environment
    env.reset()
    # Inicialmente no hay alerta
    alert = False
    # Posición del pato en el mapa (fija)
    duck_pos = np.array([2,0,2])
    # Constante que se debe calcular
    f=723.1875
    # f * dr (f es constante, dr es conocido)
    c=f*0.08
    while True:
        # Captura la tecla que está siendo apretada y almacena su valor en key
        key = cv2.waitKey(0)
        # Si la tecla es Esc, se sale del loop y termina el programa
        if key == 27:
            break
        # Se define la acción dada la tecla presionada
        action = mov_duckiebot(key)
        # Si hay alerta evitar que el Duckiebot avance
        if alert:
            action[0]=np.array([-0.5])
            pass
        # Se ejecuta la acción definida anteriormente y se retorna la observación (obs),
        # la evaluación (reward), etc
        obs, reward, done, info = env.step(action)
        # Detección de patos, retorna lista de detecciones
        dets = det_duckie(obs)
        # Dibuja las detecciones
        obs = draw_dets(obs, dets)
        # Obtener posición del duckiebot
        dbot_pos = env.cur_pos
        # Calcular distancia real entre posición del duckiebot y pato
        # esta distancia se utiliza para calcular la constante
        dist = np.sqrt(np.sum(((duck_pos - env.cur_pos )**2))) 
        # La alerta se desactiva (opción por defecto)
        alert = False      
        for d in dets:
            # Alto de la detección en pixeles
            p = d[3]
            # La aproximación se calcula según la fórmula mostrada en la capacitación
            d_aprox = c/p
            # Muestra información relevante
            print('p:', p)
            print('Da:', d_aprox)
            print('Dr:', dist)
            # Si la distancia es muy pequeña activa alerta
            if d_aprox < 0.3:
                # Activar alarma
                alert= True
                # Muestra ventana en rojo
                obs=red_alert(obs)
        # Se muestra en una ventana llamada "patos" la observación del simulador
        cv2.imshow('patos', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    # Se cierra el environment y termina el programa
    env.close()
