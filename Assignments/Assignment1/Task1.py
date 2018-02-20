from keras.datasets import mnist
from keras.datasets import cifar10
import numpy as np

def create_csv(data, name):
    #scale data and reshape to vektors (arrays)
    (x_train, y_train), (x_test, y_test) = data
    print(x_train.shape)

    #reshape the array. E.g. MNIST from 60000 x 28 x 28 to 60000x785     (28*28=785)
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    #for testing print some values without shuffle
    print(x_train[0][320:323])

    tasks_n = 3    
    mask = np.arange(len(x_train[0]))
    shuffles = []
    #save all data in for_file to convert this array to a file, the y-value is in the last column.
    for_file = np.zeros(shape=(len(y_train)+len(y_test), len(mask)+1))
    #the loop creats n tasks with different shuffeling, the shuffeling is saved in shuffles
    for i in range(tasks_n):
        #print this values after shuffeling
        np.random.shuffle(mask)
        shuffles.append(list(mask))
        print(x_train[0][shuffles[i][320:323]])

        for j in range(len(y_train)):
            for_file[j][0:len(mask)] = x_train[j][shuffles[i]]
            for_file[j][len(mask)] = y_train[j]            
            
        for j in range(len(y_test)):
            for_file[len(y_train)+j][0:len(mask)] = x_test[j][shuffles[i]]
            for_file[len(y_train)+j][len(mask)] = y_test[j]
        
        #print this values in for_file (equal to after shuffleing)
        print(for_file[0][320:323])

        print("shape of for_file:")
        print(for_file.shape)

        #now save everything in an npy file
        print("start to save")
        np.save(name + str(i+1), for_file)  
        print("file saved")


create_csv(mnist.load_data(), "mnist_")
create_csv(cifar10.load_data(), "cifar10_")
