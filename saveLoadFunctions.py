import pickle



def saveResults(myfilename, results):
    myfile = open(myfilename, 'wb')
    pickle.dump(results, myfile)
    myfile.close()


def loadResults(myfilename):
    f = open(myfilename, 'rb')
    results = pickle.load(f)
    f.close()
    return(results)