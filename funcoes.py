import numpy as np
import cv2
import math
import maxflow as mf


def DImage(img01, img02, width, height):
    dif = np.zeros((height, width), dtype='uint8')
    x1, y1 = img01.shape
    x2, y2 = img02.shape
    for x in range(height):
        for y in range(width):
            if x < x1 and y < y1 and x < x2 and y < y2:
                if img01.item(x, y) > img02.item(x, y):
                    mex = img01.item(x, y) - img02.item(x, y)
                else:
                    mex = img02.item(x, y) - img01.item(x, y)
            else:
                if x < x1 and y < y1:
                    mex = img01.item(x, y)
                elif x < x2 and y < y2:
                    mex = img02.item(x, y)
                else:
                    mex = 0
            dif.itemset((x, y), 255 - mex)
    return dif


def getMarkers(img):
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Marker labelling
    ret, markers = cv2.connectedComponents(opening)
    return markers


def getPoints(img01, img02):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img01, None)
    kp2, des2 = sift.detectAndCompute(img02, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.5*n.distance:
            good.append(m)
    return good, kp1, kp2


def getPos(x, y, col):
    return x * col + y


def valida(alt, larg, x, y):
    for i in x:
        if i < 0 or i >= alt:
            return False
    for j in y:
        if j < 0 or j >= larg:
            return False
    return True


def geraGrafo(img, flag):  # flag = 0 costura imagem na vertical, 1 costura imagem na horizontal
    tam = np.max(img) + 1
    g = mf.Graph[int](tam, tam)
    nodes = g.add_nodes(tam)
    grafo = [[-1 for x in range(tam)] for y in range(tam)]
    alt = len(img)
    larg = len(img[0])
    for x in range(alt):
        for y in range(larg):
            if(img[x][y] == -1):
                if valida(alt, larg, [x-1, x+1], [y-1, y+1]):
                    if img[x-1][y] != -1 and img[x+1][y] != -1 and img[x-1][y] != img[x+1][y]:
                        i = img[x-1][y]
                        j = img[x+1][y]
                        if grafo[i][j] == -1:
                            grafo[i][j] = grafo[j][i] = 1
                        else:
                            grafo[i][j] = grafo[j][i] = grafo[i][j] + 1
                    elif img[x][y-1] != -1 and img[x][y+1] != -1 and img[x][y-1] != img[x][y+1]:
                        i = img[x][y-1]
                        j = img[x][y+1]
                        if grafo[i][j] == -1:
                            grafo[i][j] = grafo[j][i] = 1
                        else:
                            grafo[i][j] = grafo[j][i] = grafo[i][j]+1
                elif flag:
                    if y-1 < 0:
                        g.add_tedge(nodes[img[x][y+1]], 1, 10000)
                    else:
                        g.add_tedge(nodes[img[x][y-1]], 10000, 1)
                else:
                    if x-1 < 0:
                        g.add_tedge(nodes[img[x+1][y]], 1, 10000)
                    else:
                        g.add_tedge(nodes[img[x-1][y]], 10000, 1)
    for x in range(tam):
        for y in range(tam):
            if grafo[x][y] != -1:
                g.add_edge(nodes[x], nodes[y], grafo[x][y], grafo[x][y])

    return g, nodes


def divisao(images, pontos, intervalo):
    i = 1


def geraCostura(img, g, nodes):
    alt = len(img)
    larg = len(img[0])
    for x in range(alt):
        for y in range(larg):
            if(img[x][y] == -1):
                if valida(alt, larg, [x-1, x+1], [y-1, y+1]):
                    if img[x-1][y] != -1 and img[x+1][y] != -1 and img[x-1][y] != img[x+1][y]:
                        i = img[x-1][y]
                        j = img[x+1][y]
                        if g.get_segment(nodes[i]) == g.get_segment(nodes[j]):
                            img[x][y] = img[x-1][y]
                    elif img[x][y-1] != -1 and img[x][y+1] != -1 and img[x][y-1] != img[x][y+1]:
                        i = img[x][y-1]
                        j = img[x][y+1]
                        if g.get_segment(nodes[i]) == g.get_segment(nodes[j]):
                            img[x][y] = img[x][y-1]
                    else:
                        img[x][y] = 0
                else:
                    img[x][y] = 0
    return img


def costuraImagens(img01, img02, markers, g, nodes, w, h, flag):
    img = np.zeros([h, w, 3], dtype=np.uint8)
    img.fill(255)
    x1, y1 = img01.shape
    x2, y2 = img02.shape
    for x in range(h):
        for y in range(w):
            if markers[x][y] != -1:
                if g.get_segment(nodes[markers[x][y]]) == 1:
                    if x < x1 and y < y1:
                        img[x][y] = img01[x][y]
                    elif x < x2 and y < y2:
                        img[x][y] = img02[x][y]
                else:
                    if x < x2 and y < y2 and np.any(img02[x][y]) != 0:
                        img[x][y] = img02[x][y]
                    elif x < x1 and y < y1:
                        img[x][y] = img01[x][y]
            elif x < x1 and y < y1:
                res = interpolacao(img01, img02, x, y, h, w)
                if res == 0:
                    res = math.trunc((int(img01[x-1][y])+int(img02[x+1][y]))/2)
                img[x][y] = res
                # img[x-1][y] = math.trunc((int(img01[x-1][y])+res)/2)
                # img[x+1][y] = math.trunc((res+int(img02[x+1][y]))/2)
            else:
                img[x][y] = img02[x][y]
    if flag:
        for x in range(h):
            for y in range(w):
                if markers[x][y] == -1:
                    res = interpolacao2(img, x, y, h, w)
                    img[x][y] = res

    return img


def copyOver(source, destination):
    result_grey = source  # cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(result_grey, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    roi = cv2.bitwise_and(source, source, mask=mask)
    im2 = cv2.bitwise_and(destination, destination, mask=mask_inv)
    result = cv2.add(im2, roi)
    return result


def borderPoints(width, quant):
    list_points = []
    intervalo = math.trunc(width/quant)
    aux = 0
    while aux <= width:
        list_points.append(aux)
        aux = aux + intervalo
    return list_points, intervalo


def apoioPoints(image, col, quant):
    list_points = []
    h, w = image.shape
    intervalo = math.trunc(h/quant)
    aux = 0
    while aux <= h:
        list_points.append(image[aux][col])
        aux = aux + intervalo
    return list_points, intervalo


def somaRGB(soma, valor, cont):
    if valor != 0:
        soma += valor
        cont += 1
    return soma, cont


def interpolacao(img01, img02, x, y, h, w):
    cont = 0
    aux = int(img01[x][y])
    soma = 0
    if aux != 0:
        if y-1 > 0:
            aux = int(img01[x][y-1])
            soma, cont = somaRGB(soma, aux, cont)
        if y+1 < w:
            aux = int(img02[x][y+1])
            soma, cont = somaRGB(soma, aux, cont)
        if x-1 > 0:
            aux = int(img01[x-1][y])
            soma, cont = somaRGB(soma, aux, cont)
        if x+1 < h:
            aux = int(img02[x+1][y])
            soma, cont = somaRGB(soma, aux, cont)
        if cont > 0:
            soma = math.trunc(soma/cont)
        return soma
    return 0


def interpolacao2(img, x, y, h, w):
    cont = 0
    aux = (img[x][y])
    soma = 0
    if aux[0] != 0:
        if y-1 > 0:
            aux = (img[x][y-1])
            soma, cont = somaRGB(soma, aux[0], cont)
        if y+1 < w:
            aux = (img[x][y+1])
            soma, cont = somaRGB(soma, aux[0], cont)
        if x-1 > 0:
            aux = (img[x-1][y])
            soma, cont = somaRGB(soma, aux[0], cont)
        if x+1 < h:
            aux = (img[x+1][y])
            soma, cont = somaRGB(soma, aux[0], cont)
        if cont > 0:
            soma = math.trunc(soma/cont)
        return soma
    return 0
