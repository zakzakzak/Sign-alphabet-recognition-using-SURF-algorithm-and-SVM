# import the necessary packages
from Tkinter import *
from numpy import genfromtxt
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import cv2
import numpy as np
from sklearn.externals import joblib
import math
# from multiprocessing import Pool
# import time
hessian_tres = 200
# 200
# syarat = 80000
juml = 24
clus = 36

alphabet = []
alphabet.append('a')
alphabet.append('b')
alphabet.append('c')
alphabet.append('d')
alphabet.append('e')

alphabet.append('f')
alphabet.append('g')
alphabet.append('h')
alphabet.append('i')

alphabet.append('k')
alphabet.append('l')
alphabet.append('m')
alphabet.append('n')
alphabet.append('o')

alphabet.append('p')
alphabet.append('q')
alphabet.append('r')
alphabet.append('s')
alphabet.append('t')

alphabet.append('u')
alphabet.append('v')
alphabet.append('w')
alphabet.append('x')
alphabet.append('y')

# region Description : Keluarga method

def feature_descriptor(arr_sep):
    arr_feat = []
    for i in arr_sep:
        hor = convolution2d_real(i, kernel_hor, 0)
        ver = convolution2d_real(i, kernel_ver, 0)

        a1 = integral_image(hor)
        a2 = integral_image(ver)
        a3 = integral_image_abs(hor)
        a4 = integral_image_abs(ver)

        arr_feat.append(a1)
        arr_feat.append(a2)
        arr_feat.append(a3)
        arr_feat.append(a4)

    return arr_feat


def convert_res_1(img):
    img2 = np.int16(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i, j] > 100):
                img2[i, j] = -1
    return img2


def convert_res_2(img):
    # img2 = []
    img2 = np.int16(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i, j] > 100):
                img2[i, j] = -2
    return img2


def convolution2d(image1, kernel, bias):
    m, n = kernel.shape
    tambah = m - 1
    image = np.zeros((image1.shape[0] + int(tambah), image1.shape[1] + int(tambah)))

    image[int(tambah / 2):image1.shape[0] + int(tambah / 2),
    int(tambah / 2):image1.shape[1] + int(tambah / 2)] = image1[0:image1.shape[0], 0:image1.shape[1]]
    # cv2.imshow('ok', np.uint8(image))

    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                temp = np.sum(image[i:i + m, j:j + m] * kernel) + bias
                temp = map2(temp)
                new_image[i][j] = temp
                # if (new_image[i][j] < 0):
                #     new_image[i][j] = 0
                # if (new_image[i][j] > 255):
                #     new_image[i][j] = 255

        return new_image


def convolution2d_real(image1, kernel, bias):
    # konvolusi tanpa normalisasi
    m, n = kernel.shape
    tambah = m - 1
    image = np.zeros((image1.shape[0] + int(tambah), image1.shape[1] + int(tambah)))

    image[int(tambah / 2):image1.shape[0] + int(tambah / 2),
    int(tambah / 2):image1.shape[1] + int(tambah / 2)] = image1[0:image1.shape[0], 0:image1.shape[1]]
    # cv2.imshow('ok', np.uint8(image))

    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                temp = np.sum(image[i:i + m, j:j + m] * kernel) + bias
                # temp = map2(temp)
                new_image[i][j] = temp
                # if (new_image[i][j] < 0):
                #     new_image[i][j] = 0
                # if (new_image[i][j] > 255):
                #     new_image[i][j] = 255

        return new_image


def kernel_H_scale(scale):
    row = scale * 3
    col_isi = scale * 2 + 1
    sisa = row - col_isi
    col = col_isi + sisa

    arr = np.zeros((row, col))
    putih = np.ones((scale, col_isi))
    hitam = np.ones((scale, col_isi)) * -2

    mulai = int(math.floor(sisa / 2))

    batas2 = scale * 2
    batas3 = scale * 3

    arr[0:scale, mulai:mulai + col_isi] = putih
    arr[scale:batas2, mulai:mulai + col_isi] = hitam
    arr[batas2:batas3, mulai:mulai + col_isi] = putih

    return arr


def kernel_XY_scale(scale):
    arr = np.zeros((scale * 3, scale * 3))
    hitam = np.ones((scale, scale)) * -1
    putih = np.ones((scale, scale))

    arr_hp = np.zeros((scale, (scale * 2) + 1))
    arr_ph = np.zeros((scale, (scale * 2) + 1))
    arr_hp[0:scale, 0:scale], arr_ph[0:scale, 0:scale] = hitam, putih
    arr_hp[0:scale, scale + 1:scale * 2 + 1], arr_ph[0:scale, scale + 1:scale * 2 + 1] = putih, hitam

    sisa = (scale * 3) - (scale * 2 + 1)
    mulai = int(sisa / 2)

    arr[mulai:scale + mulai, mulai: scale * 2 + 1 + mulai] = arr_hp
    arr[mulai + scale + 1:scale + mulai + scale + 1, mulai: scale * 2 + 1 + mulai] = arr_ph

    return arr


def arr_hes(conv1, conv2, conv3):
    list = []
    arr_hes = np.zeros((conv1.shape[0], conv1.shape[1]))

    for i in range(conv1.shape[0] - 85):
        i = i + 40
        for j in range((conv1.shape[1]) - 80):
            j = j + 40
            val = conv2[i, j] * conv3[i, j] - (0.912 * conv1[i, j]) ** 2

            val = map2(val)

            ###############################arr_hes[i, j] = val
            if (val > hessian_tres):
                # arr_hes[i, j] = 255
                arr_hes[i, j] = val
                list.append((i, j))

                belum_dihapus = True
                # ----------------pake while biar lebih cepat------------------
                if (i > 1 and i < arr_hes.shape[0] - 1 and j > 1 and j < arr_hes.shape[1] - 1):
                    for ii in range(i - 1, i + 2):
                        for jj in range(j - 1, j + 2):
                            if (not (ii == i and jj == j) and belum_dihapus):
                                val2 = conv2[ii, jj] * conv3[ii, jj] - (0.912 * conv1[ii, jj]) ** 2
                                val2 = map2(val2)
                                if (val2 >= val and belum_dihapus):
                                    arr_hes[i, j] = 0
                                    list.pop()
                                    belum_dihapus = False

                # --------------------------------------------------------------
            else:
                arr_hes[i, j] = 0
    return arr_hes, list


def separate(arr):
    arr2 = []
    for i in range(4):
        for j in range(4):
            arr2.append(arr[i * 10:(i + 1) * 10, j * 10:(j + 1) * 10])

    return arr2


def integral_image(img):
    sum = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for ii in range(i + 1):
                for jj in range(j + 1):
                    sum[i, j] = sum[i, j] + img[ii, jj]
    return sum[img.shape[0] - 1, img.shape[1] - 1]


def integral_image_abs(img):
    sum = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        # print(i)
        for j in range(img.shape[1]):
            for ii in range(i + 1):
                for jj in range(j + 1):
                    sum[i, j] = sum[i, j] + abs(img[ii, jj])
                    # print(sum[i,j], i,j,ii,jj)
    return sum[img.shape[0] - 1, img.shape[1] - 1]


def skin_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # gray = cv2.equalizeHist(gray)
    # print(gray)
    x = int(math.floor(gray.shape[0] / 2))
    y = int(math.floor(gray.shape[1] / 2))

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            # if(math.sqrt( ((gray[x, y, 0] - gray[i, j, 0]) ** 2) +
            #          ((gray[x, y, 0] - gray[i, j, 0]) ** 2) +
            #          ((gray[x, y, 0] - gray[i, j, 0]) ** 2)) > 410):
            #     gray[i, j, 0] = 0
            #     gray[i, j, 1] = 0
            #     gray[i, j, 2] = 0
            y = gray[i, j, 0]
            cb = gray[i, j, 2]
            cr = gray[i, j, 1]

            y1 = 0  # 90
            y2 = 300
            cb1 = 50  # 20
            cb2 = 150
            cr1 = 100
            cr2 = 200

            # y1 = 0  # 90
            # y2 = 200
            # cb1 = 50  # 20
            # cb2 = 150
            # cr1 = 100
            # cr2 = 200
            # y1  = 0
            # y2  = 200
            # cb1 = 80
            # cb2 = 200
            # cr1 = 0
            # cr2 = 200

            # if (not (y < y2 and y > y1 and cb < cb2 and cb > cb1 and cr < cr2 and cr > cr1)):
            #     gray[i, j, 0] = 0
            #     gray[i, j, 1] = 0
            #     gray[i, j, 2] = 0

            if ((y < y2 and y > y1 and cb < cb2 and cb > cb1 and cr < cr2 and cr > cr1)):
                gray[i, j, 0] = gray[i, j, 0]
                # gray[i, j, 0] = 62
                # gray[i, j, 1] = 140
                # gray[i, j, 2] = 120
            else:

                gray[i, j, 0] = 16
                gray[i, j, 1] = 128
                gray[i, j, 2] = 128
                # print('liat1', gray[i, j, 0])
                # print('liat2', gray[i, j, 1])
                # print('liat3', gray[i, j, 2])
                # print('---')

    # gray[i, j, 1] = gray[i, j, 1] + 10
    # gray[i, j, 2] = gray[i, j, 2] + 10

    gray = cv2.cvtColor(gray, cv2.COLOR_YCR_CB2BGR)
    # print('skin ', gray.shape)
    return gray


def cropper(gray, img):
    gray2 = np.zeros((gray.shape[0], gray.shape[1]))
    # print (np.uint8(gray).shape)

    # pembagian warna menjadi 0 atau tetap
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            # if(gray[i,j]-255 = 0)
            # print(type(gray[i,j]-255))
            if (gray[i, j] > 55):
                gray2[i, j] = gray[i, j]
            else:
                gray2[i, j] = 0

    gray2 = convolution2d_real(gray2, kernel_d, 0)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if (gray2[i, j] > 55):
                gray2[i, j] = gray[i, j]
            else:
                gray2[i, j] = 0
    # for i in range(gray.shape[0]):
    #     for j in range(gray.shape[1]):
    #         # if(gray[i,j]-255 = 0)
    #         # print(type(gray[i,j]-255))
    #         if(gray[i,j] > 55):
    #             gray2[i,j] = gray[i,j]
    #         else :
    #             gray2[i, j] = 0
    # SHOW sebelum crop
    # cv2.imshow('a', np.uint8(gray2))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # SHOW sebelum crop : end

    # PEMOTONGAN SAMPING
    res = cv2.resize(gray2, None, fx=1, fy=float(1), interpolation=cv2.INTER_LINEAR)
    # print(res.shape)
    ''''''
    lineY = []
    for j in range(res.shape[1]):
        jum = 0
        for i in range(res.shape[0]):
            jum = jum + res[i, j]
        lineY.append(jum)

    # print(lineY)
    ''''''

    titik = []

    for j in range(len(lineY) - 1):
        if ((lineY[j] == 0 and lineY[j + 1] != 0) or (lineY[j] != 0 and lineY[j + 1] == 0)):
            # print(lineY[j])
            # print(lineY[j+1])
            # print('----')
            titik.append(j)

    # print(titik)
    # PEMOTONGAN ATAS

    # crop1 = np.uint8(gray2[:,titik[0]:titik[1]])
    # SHOW CROP ATAS
    # cv2.imshow('aa',np.uint8(crop1))
    # cv2.waitKey(0)
    # SHOW CROP ATAS : END
    ''''''
    lineX = []
    for i in range(res.shape[0]):
        jum = 0
        for j in range(res.shape[1]):
            jum = jum + res[i, j]
        # print('ini', str(jum))
        lineX.append(jum)

    # print(lineX)
    ''''''
    titik2 = []

    for j in range(len(lineX) - 1):
        if ((lineX[j] == 0 and lineX[j + 1] != 0) or (lineX[j] != 0 and lineX[j + 1] == 0)):
            titik2.append(j)
            # print(j)

    # print(titik2)
    #
    # ''''''
    # print(len(gray2))
    result = np.uint8(img[titik2[0]:img.shape[1], titik[0]:titik[1]])
    # result = cropper(gray, img)
    # print(result.shape)

    # STRECHING RESULT
    x = 300. / result.shape[1]
    y = 300. / result.shape[0]
    result = cv2.resize(result, None, fx=x, fy=y, interpolation=cv2.INTER_LINEAR)
    # STRECHING RESULT : END

    temp = np.zeros((300, 300, 3), dtype="uint8")
    # print(temp.shape)
    temp[300 - result.shape[0]:300, 0:result.shape[1], :] = result

    return temp


def map(angka):
    besar1 = 65025 * 2
    besar2 = 255
    banyak1 = angka + 65025
    persen1 = banyak1 / besar1
    jumlah2 = besar2 * persen1
    return jumlah2


def map2(angka):
    besar1 = 10000 * 2
    besar2 = 255
    banyak1 = angka + 10000
    persen1 = banyak1 / besar1
    jumlah2 = besar2 * persen1
    return jumlah2


def distance(arr1, arr2):
    jum = 0
    for i in range(len(arr1)):
        jum = jum + (arr1[i] - arr2[i]) ** 2
    return math.sqrt(jum)


def inisialisasi_img(gambar):
    # print(gambar)
    img = cv2.imread(gambar)
    # print(img)
    img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
    # print('ok')
    # img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
    # print('ok')
    img = skin_detection(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cropper(gray, img)
    # print('ok')
    # print('ok')

    '''proses padding'''
    baru = np.ones((img.shape[0] + 90, img.shape[1] + 90, img.shape[2])) * 16
    # print(img.shape)
    # print(baru.shape)
    baru[int(90 / 2):img.shape[0] + int(90 / 2),
    int(90 / 2):img.shape[1] + int(90 / 2), 0] = img[0:img.shape[0], 0:img.shape[1], 0]

    baru[int(90 / 2):img.shape[0] + int(90 / 2),
    int(90 / 2):img.shape[1] + int(90 / 2), 1] = img[0:img.shape[0], 0:img.shape[1], 1]

    baru[int(90 / 2):img.shape[0] + int(90 / 2),
    int(90 / 2):img.shape[1] + int(90 / 2), 2] = img[0:img.shape[0], 0:img.shape[1], 2]

    baru = baru.astype(np.uint8)
    '''proses padding : end'''

    # img  = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(baru, cv2.COLOR_BGR2GRAY)
    return baru, gray


def arr_hes_all(gray):
    """_KONVOLUSI & DETERMINANT of HESSIAN MATRIX : BEGIN_"""
    # ------_KERNEL :  9 x  9_----------------------
    # conv1 = convolution2d(gray, kernel9_xy, 0)
    # conv2 = convolution2d(gray, kernel9_h, 0)
    # conv3 = convolution2d(gray, kernel9_v, 0)
    # arr_hes0, list9x9 = arr_hes(conv1, conv2, conv3)

    # print("Kernel 9x9 done...")
    # ------_KERNEL : 15 x 15_----------------------
    conv1 = convolution2d(gray, kernel15_xy, 0)
    conv2 = convolution2d(gray, kernel15_h, 0)
    conv3 = convolution2d(gray, kernel15_v, 0)
    arr_hes1, list15x15 = arr_hes(conv1, conv2, conv3)

    # print("Kernel 15x15 done...")
    # ------_KERNEL : 21 x 21_-----------------------
    # conv1 = convolution2d(gray, kernel21_xy, 0)
    # conv2 = convolution2d(gray, kernel21_h, 0)
    # conv3 = convolution2d(gray, kernel21_v, 0)
    # arr_hes2, list21x21 = arr_hes(conv1, conv2, conv3)
    # print("Kernel 21x21 done...")
    # ------_HESSIAN MATRIX COMPUTATION : END_-------

    return arr_hes1, list15x15


def feature_n_draw(list15x15, img):
    arr_feat_point = []
    # arr_feat_point.append(k)

    for i in range(len(list15x15)):
        # print(len(list15x15))
        crop1 = img[list15x15[i][0] - 20:list15x15[i][0] + 20, list15x15[i][1] - 20:list15x15[i][1] + 20]

        arr_sep = separate(crop1)
        arr_feat = feature_descriptor(arr_sep)
        cv2.circle(img, (list15x15[i][1], list15x15[i][0]), 3, (255, 0, 255), 1, 1, 0)

        # penambahan point feature
        point = np.ravel(np.array([list15x15[i][1], list15x15[i][0]], dtype='f8'))
        # aa = np.concatenate((point, arr_feat))
        # # print(aa)
        # arr_feat_point.append(np.concatenate((np.array([k]),aa)))
        # # ganti HOG dengan haar wav
        # arr_fd, hog_image = hog(crop1, orientations=8, pixels_per_cell=(40, 40),
        #                     cells_per_block=(1, 1), visualize=True, multichannel=False)
        #
        #
        #
        # fd = arr_fd
        # arr_feat.append([list15x15[i][1], list15x15[i][0], fd[ 0], fd[ 1], fd[ 2], fd[ 3], fd[ 4], fd[ 5], fd[ 6], fd[ 7]
        #                                                  , fd[ 8], fd[ 9], fd[10], fd[11], fd[12], fd[13], fd[14], fd[15]
        #                                                  , fd[16], fd[17], fd[18], fd[19], fd[20], fd[21], fd[22], fd[23]
        #                                                  , fd[24], fd[25], fd[26], fd[27], fd[28], fd[29], fd[30], fd[31]])
        # print(arr_feat_point)
        arr_feat_point.append(np.concatenate((point, arr_feat)))
    # print("---list15-----")
    # print(  arr_feat_point)
    # print("---list15-----")

    return arr_feat_point, img


def proses_feature(gambar):
    # method proses_feature
    # input  : gambar / string
    # output : img / np , arr_feature / list

    img, gray = inisialisasi_img(gambar)

    gray_skin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arr_hes1, list15x15 = arr_hes_all(gray_skin)

    arr_feat1, img = feature_n_draw(list15x15, gray)

    return arr_feat1, img


def matching(arr_feat1, arr_feat2, idx):
    pair_feat = []
    print("----INI WOII----")
    print(tambahan)

    for i in range(arr_feat1.shape[0]):
        # print(len(arr_feat1[i, 2:32]))
        # print(arr_feat1[i, :65])
        # print('---------------------------------------------------------')
        min = distance(arr_feat1[i, 2:], arr_feat2[0, 2:])
        id = 0
        if (abs(arr_feat1.shape[0] - arr_feat2.shape[0]) < 50):
            for j in range(arr_feat2.shape[0]):
                dist_ij = distance(arr_feat1[i, 2:], arr_feat2[j, 2:])
                # print(dist_ij, "ini")
                if (dist_ij < min):
                    min = dist_ij
                    id = j
            #  mengecek distance rata2 (comment bila tidak perlu)
            # print(min)
            # check : end
            # cek gradient
            gradient = int(arr_feat2[id, 1]) - int(arr_feat1[i, 1])
            # cek gradient : end
            if (min < syarat and abs(gradient) < 10):
                # if (min < syarat):
                pair_feat.append(
                    [int(arr_feat1[i, 0]), int(arr_feat1[i, 1]), tambahan + int(arr_feat2[id, 0]),
                     int(arr_feat2[id, 1])])
    if (len(pair_feat) != 0):
        print('------------------------')
        print("kecocokan dengan " + alphabet[idx] + " : " + str(float(len(pair_feat)) * 100 / arr_feat2.shape[0]))
        print("Jum titik    : " + str(arr_feat1.shape[0]) + " & " + str(arr_feat2.shape[0]) + " : " + str(
            arr_feat1.shape[0] - arr_feat2.shape[0]))
    return pair_feat


# def matching_instan(gambar1, gambar2) :
#     arr_feat1, img1 = proses_feature(gambar1)
#     arr_feat2, img2 = proses_feature(gambar2)
#
#     tambahan = img1.shape[1]
#
#     return arr_feat1,arr_feat2,img1,img2,tambahan


# endregion

# region Description : Filter first dan second order [ kernel ]
'''-------------------------------------------------'''
kernel9_h = kernel_H_scale(1)
kernel15_h = kernel_H_scale(9)
kernel21_h = kernel_H_scale(1)
kernel27_h = kernel_H_scale(9)

kernel9_v = np.transpose(kernel9_h)
kernel15_v = np.transpose(kernel15_h)
kernel21_v = np.transpose(kernel21_h)
kernel27_v = np.transpose(kernel27_h)

kernel9_xy = kernel_XY_scale(1)
kernel15_xy = kernel_XY_scale(9)
kernel21_xy = kernel_XY_scale(1)
kernel27_xy = kernel_XY_scale(9)

# FILTER first order : BEGIN
kernel_hor_l = [[1, 1, 1, 1],
                [1, 1, 1, 1],
                [-1, -1, -1, -1],
                [-1, -1, -1, -1]]

kernel_hor = np.array(kernel_hor_l)
# print(kernel_hor.shape)
kernel_ver = np.transpose(kernel_hor)
# print(kernel_ver)
kernel_d = [[0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]]

kernel_d = np.array(kernel_d)
# FILTER first order : END
'''-------------------------------------------------'''
# endregion [ kernel

def select_image():
    global panelA, panelB
    path = tkFileDialog.askopenfilename()

    if len(path) > 0:
        image = cv2.imread(path)
        arr_feat, img = proses_feature(path)
        arr_feat = np.array(arr_feat)
        # print(arr_feat)

        arr_clus = np.zeros((clus))
        my_data2 = arr_feat[:, 2:]
        jwb = kmeans.predict(my_data2)
        for l in jwb:
            arr_clus[l] = arr_clus[l] + 1
        # print(arr_clus)
        jawaban = clf.predict([arr_clus])
        # print(jawaban)
        print('jawaban : '+alphabet[int(jawaban[0])])

        image = cv2.resize(image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)

        edged = cv2.imread("dataset_semua/orang1/" + alphabet[int(jawaban[0])] + " (1).jpg")
        edged = cv2.resize(edged, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        edged = cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)

        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)
        if panelA is None or panelB is None:
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
            panelB = Label(image=edged)
            panelB.image = edged
            panelB.pack(side="right", padx=10, pady=10)
        else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged


kmeans = joblib.load('saved_model.pkl')
clf = joblib.load('saved_mode2.pkl')

# arr_dataset = genfromtxt("data_train_bag_of_feature.csv", delimiter=',')
#
# x_train = arr_dataset[:144*14, :clus]
# x_test  = arr_dataset[ 144*14:144*15, :clus]
# y_train = arr_dataset[:144*14,  clus]
# y_test  = arr_dataset[ 144*14:144*15,  clus]
#
#
# jawaban = clf.predict(x_test)
#
# for i in jawaban :
#     print(alphabet[int(i)])

root = Tk()
panelA = None
panelB = None

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# kick off the GUI
root.mainloop()