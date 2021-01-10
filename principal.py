import numpy as np
import cv2
import argparse
import math
import copy
from imutils import paths
from matplotlib import pyplot as plt
from funcoes import DImage, divisao, getMarkers, getPoints, getPos, geraGrafo, geraCostura, costuraImagens, apoioPoints, borderPoints


def aplicaCostura(images, x, y, flag):
    resultado = 0
    idx1 = getPos(x, y, columnSize)
    idx2 = getPos(x, y+1, columnSize)
    print("Indexes Atuais: ", idx1, " ", idx2)
    # encontra os pontos chaves das imagens com SIFT
    print("Procurando KeyPoints...", end=" ")
    good, kp1, kp2 = getPoints(images[idx2], images[idx1])
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        print("OK")
        # Matriz de homografia com RANSAC
        print("Gerando Matriz Homográfica...", end=" ")
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        print("OK")
        # Posiciona uma imagem em relação a outra
        print("Posicionando Imagens...", end=" ")
        h, w = images[idx2].shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        bx, by, bwidth, bheight = cv2.boundingRect(dst)
        aux = cv2.warpPerspective(images[idx2], M, (bwidth, bheight))
        # pano = cv2.addWeighted(aux, 0.6, images[idx1], 0.4, 0.0)
            
        print("OK")
        # Gera imagem de diferença entre as duas
        print("Gerando Imagem Diferenciada...", end=" ")
        dif = DImage(aux, images[idx1], bwidth, bheight)
        print("OK")
        print(bwidth)
        # Gera o marcador das imagens
        print("Gerando Marcador...", end=" ")
        markers = getMarkers(dif)
        print("OK")
        dif = cv2.merge((dif, dif, dif))
        dif2 = copy.deepcopy(dif)
        # Aplica o algoritmo Watershed
        print("Aplicando WaterShed...", end=" ")
        markers = cv2.watershed(dif, markers)
        print("OK")
        dif3 = copy.deepcopy(dif)
        dif3[markers == -1] = [255, 0, 0]
        dif4 = copy.deepcopy(dif3)
        for q in range(255):
            dif4[markers == q] = [0, q, 255]
        print("Gerando Grafo...", end=" ")
        grafo, nodes = geraGrafo(markers, 1)
        print("OK")
        print("MaxFlow...", end=" ")
        flow = grafo.maxflow()
        print("OK")
        print(flow)
        print("Pintando Costura...", end=" ")
        markers = geraCostura(markers, grafo, nodes)
        print("OK")
        print("Costurando Imagens...", end=" ")
        resultado = costuraImagens(images[idx1], aux, markers, grafo, nodes, bwidth, bheight, flag)
        print("OK")
        dif[markers == -1] = [255, 0, 0]
        dif5 = copy.deepcopy(dif)
        dif5[markers != -1] = [0, 0, 0]
            
        crop_img = resultado[0:bheight, 0:bwidth].copy()
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) 
        cv2.imwrite(args["output"], resultado)
        return crop_img
        fig = plt.figure(figsize=(8, 8))
        columns = 3
        rows = 2
        # for i in range(1, columns*rows + 1):
        fig.add_subplot(rows, columns, 1)
        plt.imshow(aux, 'gray')
        plt.xticks([]), plt.yticks([])
        fig.add_subplot(rows, columns, 2)
        plt.imshow(dif2, 'brg')
        plt.xticks([]), plt.yticks([])
        fig.add_subplot(rows, columns, 3)
        plt.imshow(dif3, 'brg')
        plt.xticks([]), plt.yticks([])
        fig.add_subplot(rows, columns, 4)
        plt.imshow(dif4, 'brg')
        plt.xticks([]), plt.yticks([])
        fig.add_subplot(rows, columns, 5)
        plt.imshow(resultado, 'brg')
        plt.xticks([]), plt.yticks([])
        fig.add_subplot(rows, columns, 6)
        plt.imshow(dif, 'brg')
        plt.xticks([]), plt.yticks([])
        plt.show()
    else:
        print("Nada encontrado")


# Argumentos do programa 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Pasta com imagens")
ap.add_argument("-l", "--linha", required=True, help="Numero de linhas")
ap.add_argument("-c", "--coluna", required=True, help="Numero de colunas")
ap.add_argument("-o", "--output", required=True, help="Imagem de saída")
ap.add_argument("-x", "--extra", required=True, help="Imagem de saída")
args = vars(ap.parse_args())
route = "artigo/apoio/" + args["extra"]
cont = 0
rowSize = int(args["linha"])
resultado3 = cv2.imread(route, 0)
columnSize = int(args["coluna"])
imagePaths = sorted(list(paths.list_images(args["input"])))
images = []
list_cols = []
resultado2 = cv2.imread(route, 1)
print("Carregando Imagens...", end=" ")
# Carrega todas as imagens da pasta
for imagePath in imagePaths:
    image = cv2.imread(imagePath, 0)
    images.append(image)
print("OK")
# encontra a imagem central
print("Calculando Centro...", end=" ")
midX = math.ceil(rowSize/2)
midY = math.ceil(columnSize/2)
print("OK")
for x in range(rowSize):
    for y in range(columnSize-1):
        resultado = aplicaCostura(images, x, y, False)
    list_cols.append(copy.deepcopy(resultado))
newImage = []
for y in range(columnSize):
    h, w = list_cols[y].shape
    print("Calculando Pontos de apoio...", end=" ")
    pontos_apoio, intervaloA = apoioPoints(list_cols[y], 100, 20)
    print("OK")
    print("Calculando Pontos de borda...", end=" ")
    pontos_borda, intervaloB = borderPoints(w, 20)
    print("OK")
    newImage.append(divisao(list_cols[y], borderPoints, intervaloB))
resultado = aplicaCostura(list_cols, 0, 0, True)
resultado = cv2.resize(resultado, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

cv2.imwrite(args["output"], resultado2)
plt.imshow(resultado3, 'gray')
plt.xticks([]), plt.yticks([])
plt.show()