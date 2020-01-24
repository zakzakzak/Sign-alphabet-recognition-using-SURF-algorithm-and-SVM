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


for i in range(15):
    for j in range(24):
        for k in range(6):
            kode = alphabet[j]+' ('+str(k)+')'
            print(str(i+1)+' org '+ kode)
            my_dat2 = genfromtxt('dataset_semua/orang'+str(i+1)+'/csv1/'+ kode + '.csv', delimiter=',')
            # print(my_dat2[:,2:].shape)
            # print(my_dat2.shape)
            # for l in range()
            if (my_dat2.shape[0]!= 0):
                try :
                    try :
                        my_dat = np.concatenate((my_dat,np.array([my_dat2[2:]])), axis = 0)
                    except:
                        
                        my_dat = np.concatenate((my_dat,my_dat2[:,2:]), axis = 0)
              
                except(NameError):
                    # print('ok')
                    my_dat = my_dat2[:,2:]

print(my_dat.shape)
for lop in range(1):
    clus = 35
    kmeans = KMeans(n_clusters=clus, random_state=53).fit(my_dat)
    joblib.dump(kmeans, 'saved_model40.pkl')
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
                if(my_dat2.shape[0] != 0):
                    #print(my_dat2)
                    try :
                        my_data2 = my_dat2[:,2:]
                    except:
                        my_data2 = np.array([my_dat2[2:]])
                    jwb = kmeans.predict(my_data2)
                    for l in jwb:
                        arr_clus[l] = arr_clus[l] + 1
                    #print(arr_clus)
                    #print(jwb.shape[0])
                    arr_clus[arr_clus.shape[0]-1] = j
                    arr_dataset.append(arr_clus)

                else:
                    kode = alphabet[j]+' ('+str(k)+')'
                    print(str(i+1)+' org '+ kode)
                    arr_clus[arr_clus.shape[0]-1] = 24
                    
              
                    

    arr_dataset = np.array(arr_dataset)

    # normal 1
    arr_dataset_n = []

    for i in range(13,15):
        for j in range(24):
            for k in range(1):
                #print(i,j,k)

                arr_clus = np.zeros((clus+1))
                kode = alphabet[j]+' ('+str(k)+')'
                #print(str(i+1)+' org '+ kode)
                my_dat2 = genfromtxt('dataset_semua/orang'+str(i+1)+'/csv1/'+ kode + '.csv', delimiter=',')
                if(my_dat2.shape[0] != 0):
                    #print(my_dat2)
                    try :
                        my_data2 = my_dat2[:,2:]
                    except:
                        my_data2 = np.array([my_dat2[2:]])
                    jwb = kmeans.predict(my_data2)
                    for l in jwb:
                        arr_clus[l] = arr_clus[l] + 1
                    #print(arr_clus)
                    #print(jwb.shape[0])
                    arr_clus[arr_clus.shape[0]-1] = j
                    arr_dataset_n.append(arr_clus)

                else:
                    kode = alphabet[j]+' ('+str(k)+')'
                    print(str(i+1)+' org '+ kode)
                    arr_clus[arr_clus.shape[0]-1] = 24
                    
              
                    

    arr_dataset_n = np.array(arr_dataset_n)

    # rotasi 2 3

    arr_dataset_r = []

    for i in range(13,15):
        for j in range(24):
            for k in range(1,3):
                #print(i,j,k)

                arr_clus = np.zeros((clus+1))
                kode = alphabet[j]+' ('+str(k)+')'
                #print(str(i+1)+' org '+ kode)
                my_dat2 = genfromtxt('dataset_semua/orang'+str(i+1)+'/csv1/'+ kode + '.csv', delimiter=',')
                if(my_dat2.shape[0] != 0):
                    #print(my_dat2)
                    try :
                        my_data2 = my_dat2[:,2:]
                    except:
                        my_data2 = np.array([my_dat2[2:]])
                    jwb = kmeans.predict(my_data2)
                    for l in jwb:
                        arr_clus[l] = arr_clus[l] + 1
                    #print(arr_clus)
                    #print(jwb.shape[0])
                    arr_clus[arr_clus.shape[0]-1] = j
                    arr_dataset_r.append(arr_clus)

                else:
                    kode = alphabet[j]+' ('+str(k)+')'
                    print(str(i+1)+' org '+ kode)
                    arr_clus[arr_clus.shape[0]-1] = 24

    arr_dataset_r = np.array(arr_dataset_r)

    # cahaya 4 5 6

    arr_dataset_c = []

    for i in range(13,15):
        for j in range(24):
            for k in range(3,6):
                #print(i,j,k)

                arr_clus = np.zeros((clus+1))
                kode = alphabet[j]+' ('+str(k)+')'
                #print(str(i+1)+' org '+ kode)
                my_dat2 = genfromtxt('dataset_semua/orang'+str(i+1)+'/csv1/'+ kode + '.csv', delimiter=',')
                if(my_dat2.shape[0] != 0):
                    #print(my_dat2)
                    try :
                        my_data2 = my_dat2[:,2:]
                    except:
                        my_data2 = np.array([my_dat2[2:]])
                    jwb = kmeans.predict(my_data2)
                    for l in jwb:
                        arr_clus[l] = arr_clus[l] + 1
                    #print(arr_clus)
                    #print(jwb.shape[0])
                    arr_clus[arr_clus.shape[0]-1] = j
                    arr_dataset_c.append(arr_clus)

                else:
                    kode = alphabet[j]+' ('+str(k)+')'
                    print(str(i+1)+' org '+ kode)
                    arr_clus[arr_clus.shape[0]-1] = 24

    arr_dataset_c = np.array(arr_dataset_c)

    # blur 7

    arr_dataset_b = []

    for i in range(13,15):
        for j in range(24):
            for k in range(6,7):
                #print(i,j,k)

                arr_clus = np.zeros((clus+1))
                kode = alphabet[j]+' ('+str(k)+')'
                #print(str(i+1)+' org '+ kode)
                my_dat2 = genfromtxt('dataset_semua/orang'+str(i+1)+'/csv1/'+ kode + '.csv', delimiter=',')
                if(my_dat2.shape[0] != 0):
                    #print(my_dat2)
                    try :
                        my_data2 = my_dat2[:,2:]
                    except:
                        my_data2 = np.array([my_dat2[2:]])
                    jwb = kmeans.predict(my_data2)
                    for l in jwb:
                        arr_clus[l] = arr_clus[l] + 1
                    #print(arr_clus)
                    #print(jwb.shape[0])
                    arr_clus[arr_clus.shape[0]-1] = j
                    arr_dataset_b.append(arr_clus)

                else:
                    kode = alphabet[j]+' ('+str(k)+')'
                    print(str(i+1)+' org '+ kode)
                    arr_clus[arr_clus.shape[0]-1] = 24

    arr_dataset_b = np.array(arr_dataset_b)

    #-------------------------------------------------------------



    
    print(arr_dataset.shape)
    clf = SVC(kernel ='linear', C=1, gamma=1)



    np.savetxt("data_train_bag_of_feature40.csv", arr_dataset, delimiter=",")


    x_train = arr_dataset[:144*13, :clus]
    y_train = arr_dataset[:144*13,  clus]

    x_test_n  = arr_dataset_n[ :, :clus]
    y_test_n  = arr_dataset_n[ :,  clus]

    x_test_r  = arr_dataset_r[ :, :clus]
    y_test_r  = arr_dataset_r[ :,  clus]

    x_test_c  = arr_dataset_c[ :, :clus]
    y_test_c  = arr_dataset_c[ :,  clus]

    x_test_b  = arr_dataset_b[ :, :clus]
    y_test_b  = arr_dataset_b[ :,  clus]

    clf.fit(x_train, y_train)
    #joblib.dump(clf, 'saved_model2.pkl')
    #print(arr_dataset.shape)
    #sc = StandardScaler()
    jawaban = clf.predict(x_test_n)
    print('akurasi : '+ str(accuracy_score(y_test_n, jawaban)*100))

    jawaban = clf.predict(x_test_r)
    print('akurasi : '+ str(accuracy_score(y_test_r, jawaban)*100))

    jawaban = clf.predict(x_test_c)
    print('akurasi : '+ str(accuracy_score(y_test_c, jawaban)*100))

    jawaban = clf.predict(x_test_b)
    print('akurasi : '+ str(accuracy_score(y_test_b, jawaban)*100))

    
    


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



