from numpy import genfromtxt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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



def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    alp = np.array(alphabet)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(24)
    plt.xticks(tick_marks, alp, rotation=45)
    plt.yticks(tick_marks, alp)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# def region(a):
#     if (a[0] <= 390. / 2 and a[1] <= 390. / 2):
#         return 0
#     if (a[0] > 390. / 2 and a[1] <= 390. / 2):
#         return 1
#     if (a[0] <= 390. / 2 and a[1] > 390. / 2):
#         return 2
#     if (a[0] > 390. / 2 and a[1] > 390. / 2):
#         return 3

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

abrar  = 'bluimg2/csv_abrar/data'
rizki  = 'bluimg2/csv_rizki/data'
kujeng = 'bluimg2/csv_kujeng/data'
limi   = 'bluimg2/csvx2/data'

# my_dat = np.array((0,64))
# print(my_dat)
#

# for i in range(15):
#     for j in range(24):
#         for k in range(3):
#             kode = alphabet[j]+' ('+str(k)+')'
#             print(str(i+1)+' org '+ kode)
#             my_dat2 = genfromtxt('dataset_semua/orang'+str(i+1)+'/csv1/'+ kode + '.csv', delimiter=',')
#             # print(my_dat2[:,2:].shape)
#             print(my_dat2.shape)
#             try :
#                 my_dat = np.concatenate((my_dat,my_dat2[:,2:]), axis = 0)
#                 # print('ok2')
#             except(NameError):
#                 # print('ok')
#                 my_dat = my_dat2[:,2:]
#
# print(my_dat.shape)
# clus = 19
# kmeans = KMeans(n_clusters=clus, random_state=0).fit(my_dat)
# arr_dataset = []
#
# for i in range(15):
#     for j in range(24):
#         for k in range(3):
#
#             arr_clus = np.zeros(((clus*4)+1))
#             kode = alphabet[j]+' ('+str(k)+')'
#             print(str(i+1)+' org '+ kode)
#             my_dat2 = genfromtxt('dataset_semua/orang'+str(i+1)+'/csv1/'+ kode + '.csv', delimiter=',')
#             print(my_dat2[:,2:].shape)
#             my_data2 = my_dat2[:,2:]
#
#             jwb = kmeans.predict(my_data2)
#             # for l in jwb:
#             for ll in range(jwb.shape[0]):
#                 reg = region(my_dat2[ll, :2])
#                 arr = jwb[ll]+(clus*reg)
#                 arr_clus[arr] = arr_clus[arr] + 1
#
#             # persenan
#             # for ii in range(jwb.shape[0]):
#             # for jj in range(clus):
#             #     arr_clus[jj] = (arr_clus[jj]*1.)/400
#             # persenan : end
#             # print(arr_clus)
#             print(arr_clus)
#             print(jwb.shape[0])
#             arr_clus[arr_clus.shape[0]-1] = j
#             arr_dataset.append(arr_clus)
#             # print(arr_clus)
#             # for l in range(my_data2.shape[0]):
#             #     # print(my_data2[l].shape)
#             #     jwb = kmeans.predict()
#
# arr_dataset = np.array(arr_dataset)
# clf = GaussianNB()
#
# x_train = arr_dataset[:72*10, :clus]
# x_test  = arr_dataset[72*10:72*15, :clus]
# y_train = arr_dataset[:72*10,  clus]
# y_test  = arr_dataset[72*10:72*15,  clus]
#
# # sc = StandardScaler()
# #
# # x_train = sc.fit_transform(x_train)
# # x_test  = sc.fit_transform(x_test)
#
#
# # print(x_.shape)
# clf.fit(x_train, y_train)
# print(arr_dataset.shape)
#
# jawaban = clf.predict(x_test)
#
# print('akurasi')
# print(accuracy_score(y_test, jawaban)*100)





for i in range(15):
    for j in range(24):
        for k in range(6):
            kode = alphabet[j]+' ('+str(k)+')'
            print(str(i+1)+' org '+ kode)
            my_dat2 = genfromtxt('dataset_semua/orang'+str(i+1)+'/csv1/'+ kode + '.csv', delimiter=',')
            # print(my_dat2[:,2:].shape)
            # print(my_dat2.shape)
            # for l in range()
            try :
                my_dat = np.concatenate((my_dat,my_dat2[:,2:]), axis = 0)
                # print('ok2')
            except(NameError):
                # print('ok')
                my_dat = my_dat2[:,2:]

print(my_dat.shape)
for a in range(1):
    clus = 36
    kmeans = KMeans(n_clusters=clus, random_state=53).fit(my_dat)
    joblib.dump(kmeans, 'saved_model.pkl')
    # 66
    arr_dataset = []

    for i in range(15):
        for j in range(24):
            for k in range(6):
                #print(i,j,k)

                arr_clus = np.zeros((clus+1))
                kode = alphabet[j]+' ('+str(k)+')'
                #print(str(i+1)+' org '+ kode)
                my_dat2 = genfromtxt('dataset_semua/orang'+str(i+1)+'/csv1/'+ kode + '.csv', delimiter=',')
                #print(my_dat2)
                #rint(my_dat2[:,2:].shape)
                my_data2 = my_dat2[:,2:]
                # reg = region(my_dat2[:,:2])
                jwb = kmeans.predict(my_data2)
                for l in jwb:
                    arr_clus[l] = arr_clus[l] + 1
                # print(arr_clus)
                # persenan
                # for ii in range(jwb.shape[0]):
                # for jj in range(clus):
                #     arr_clus[jj] = (arr_clus[jj]*1.)/400
                # persenan : end
                # print(arr_clus)
                #print(arr_clus)
                #print(jwb.shape[0])
                arr_clus[arr_clus.shape[0]-1] = j
                # -----
                #if(k == 3):
                #    arr_clus[arr_clus.shape[0]-1] = j+24
                # -----
                arr_dataset.append(arr_clus)
                # print(arr_clus)
                # for l in range(my_data2.shape[0]):
                #     # print(my_data2[l].shape)
                #     jwb = kmeans.predict()

    arr_dataset = np.array(arr_dataset)
    clf = SVC(kernel ='linear', C=1, gamma=1)


    np.savetxt("data_train_bag_of_feature.csv", arr_dataset, delimiter=",")


    x_train = arr_dataset[:144*13, :clus]
    y_train = arr_dataset[:144*13,  clus]

    x_test  = arr_dataset[144*13:144*15, :clus]
    y_test  = arr_dataset[144*13:144*15,  clus]

    clf.fit(x_train, y_train)
    joblib.dump(clf, 'saved_mode2.pkl')
    print('ok')
    # print(arr_dataset.shape)
    #
    # jawaban = clf.predict(x_test)
    #
    # cm = confusion_matrix(y_test, jawaban)
    #
    # np.set_printoptions(precision=2)
    # print('Confusion matrix, without normalization')
    # print(cm)
    # plt.figure()
    # plot_confusion_matrix(cm)
    #
    # # Normalize the confusion matrix by row (i.e by the number of samples
    # # in each class)
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print('Normalized confusion matrix')
    # print(cm_normalized)
    # plt.figure()
    # plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    #
    # plt.show()

    
'''
clf.fit(x_train, y_train)
joblib.dump(clf, 'saved_model2.pkl')
print(arr_dataset.shape)
sc = StandardScaler()
jawaban = clf.predict(x_test)
        
print('akurasi')
print(accuracy_score(y_test, jawaban)*100)
'''

# ----------kfold-------------------------
# x_train = sc.fit_transform(x_train)
# x_test  = sc.fit_transform(x_test)
# sc = StandardScaler()
# # X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4, random_state=0)
#
# scores = cross_val_score(clf, x_train, y_train, cv=10)
# print(scores)
# print(np.mean(scores))
# -------------------------kfold
# clf.fit(x_train, y_train)
# print(arr_dataset.shape)
#
# jawaban = clf.predict(x_test)
#
# print('akurasi')
# print(accuracy_score(y_test, jawaban)*100)
#
# for i in range(jawaban.shape[0]):
#     if((i+1)%6 == 0):
#         print(alphabet[int(jawaban[i])])
#
# print(jawaban)
# -------------------------------------------------


# print(kmeans.labels_)

# a = np.array([1,2])
# b = np.array([2,4])
# print(np.concatenate((a,b),axis =0))


#
# my_dat = genfromtxt(kujeng+str(0)+'.csv', delimiter=',')
# for i in range(24):
#     my_data = genfromtxt(kujeng+str(i)+'.csv', delimiter=',')
#     my_dat = np.concatenate((my_data,my_dat), axis = 0)
#
# # data_sign = np.array(data_sign)
#
# data_sign_abrar = []
# for i in range(24):
#     my_data2 = genfromtxt(abrar+str(i)+'.csv', delimiter=',')
#     # print(my_data2.shape)
#     # print(type(my_data[3,3]))
#     my_dat = np.concatenate((my_dat,my_data2), axis =0 )
#
# print(my_dat.shape)
#
# data_sign_abrar = np.array(data_sign_abrar)
#
# data_sign_rizki = []
# for i in range(24):
#     my_data = genfromtxt(limi+str(i)+'.csv', delimiter=',')
#     # print(type(my_data[3,3]))
#     data_sign_rizki.append(my_data)
#
# data_sign_rizki = np.array(data_sign_rizki)
#
# data_sign_kujeng = []
# for i in range(24):
#     my_data = genfromtxt(rizki+str(i)+'.csv', delimiter=',')
#     # print(type(my_data[3,3]))
#     data_sign_kujeng.append(my_data)
#
# data_sign_kujeng = np.array(data_sign_kujeng)
