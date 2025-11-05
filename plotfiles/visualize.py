import matplotlib
matplotlib.use('Agg')
from numpy import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import sys


def plot(mpath='vectors.bin',wpath='words.txt',savefile = 'plot.png'):
    model = KeyedVectors.load_word2vec_format(mpath, binary=True)
    words = []
    file = open(wpath,'r', encoding='utf-8')
    vectors = None
    
    for line in file:
        lineu = line[:-1].strip()
        
        # Skip if word not in vocabulary
        if lineu not in model:
            print(f"Warning: '{lineu}' not found in vocabulary, skipping...")
            continue
            
        words.append(lineu)
        if vectors is None :
            vectors = model[lineu]
        else:	
            vectors = vstack((vectors,model[lineu]))
    
    if vectors is None or len(words) == 0:
        print("Error: No valid words found in vocabulary!")
        return
    
    pca = PCA(n_components=2)
    pca.fit(vectors)
    
    print(pca.explained_variance_ratio_)
    print('fitted')
    
    result = pca.transform(vectors)
    
    x = result[:,0]
    y = result[:,1]
    
    plt.scatter(x,y)
    j = 0
    for i in words:
        plt.annotate(i,xy=(x[j],y[j]),xytext = (3, 3),textcoords = 'offset points', ha= 'left', va = 'top')
        j = j+1
    
    plt.savefig(savefile)
    print(f'file saved {savefile}')
    try:
        plt.show()
    except:
        print('you have no display')



plot(sys.argv[1],sys.argv[2],sys.argv[3])