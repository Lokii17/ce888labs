from keras.datasets import mnist
from keras.datasets import cifar10
import numpy as np
import pandas as pd 

def create_csv(data, name):
    #scale data and reshape to vektors (arrays)
    (x_train, y_train), (x_test, y_test) = data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    #for testing print some values without shuffle
    print(x_train[0][320:323])

    tasks_n = 3    
    mask = np.arange(len(x_train[0]))
    shuffles = []
    #creats n tasks with different shuffeling, the shuffeling is saved in shuffles
    for i in range(tasks_n):
        #print this values after shuffeling
        np.random.shuffle(mask)
        shuffles.append(list(mask))
        print(x_train[0][shuffles[i][320:323]])

        #save all data in for_csv to convert this array to a csv file, the y-value is in the last column
        for_csv = np.zeros(shape=(len(y_train)+len(y_test), len(mask)+1))
        for j in range(len(y_train)):
            for_csv[j][0:len(mask)] = x_train[j][shuffles[i]]
            for_csv[j][len(mask)] = y_train[j]            
            
        for j in range(len(y_test)):
            for_csv[len(y_train)+j][0:len(mask)] = x_test[j][shuffles[i]]
            for_csv[len(y_train)+j][len(mask)] = y_test[j]
        
        #print this values in for_csv (equal to after shuffleing)
        print(for_csv[0][320:323])

        print("shape of for_csv:")
        print(for_csv.shape)

        #now save everything in an csv file
        print("start to save")        
        df = pd.DataFrame(for_csv)
        df.to_csv(name + str(i+1) + ".csv")
        print("file saved")


create_csv(mnist.load_data(), "mnist_")
create_csv(cifar10.load_data(), "cifar10_")
