import argparse,nltk,os,string,cPickle,math
from nltk.corpus import stopwords

def preprocess(infile,outfile,wordfile):
    with open(infile,'r') as f:
        data=f.readlines()

    docNum = data[0].split('\r\n')[0]
    docList=[]
    for item in data[1:]:
        docList.append(item.split('\r\n')[0])
    #print docList
    docWords=[]
    for doc in docList:
        tokens=nltk.word_tokenize(doc)
        no_stop_words = [w for w in tokens if not w in stopwords.words('english')]
        myStemmer = nltk.stem.lancaster.LancasterStemmer()
        stemmed_words=[myStemmer.stem(word) for word in no_stop_words]
        docWords.append(stemmed_words)
    #print docWords
    totalIndex={}
    index=1
    for doc in docWords:
        for word in doc:
            if word not in totalIndex:
                totalIndex[word]=[index,1]
                index+=1
            else:
                totalIndex[word][1]+=1
    print len(totalIndex)
    cPickle.dump(totalIndex,open(wordfile,'wb'),2)
    featureList=[]
    for doc in docWords:
        wordDict={}
        for word in doc:
            if word not in wordDict:
                wordDict[word]=1
            else:
                wordDict[word]+=1
        featureDict={}
        for word in wordDict:
            featureDict[totalIndex[word][0]]=wordDict[word]
        featureList.append(featureDict)
    #print featureList
    with open(outfile,'w') as outf:
        for featureDict in featureList:
            for feature in featureDict:
                outf.write(str(feature)+':'+str(featureDict[feature])+' ')
            outf.write('\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser("Preprocessing routine.")
    parser.add_argument('-i','--input_file',help='Input data file path',required=True)
    parser.add_argument('-f','--feature_file',help='Output feature file name',required=True)
    parser.add_argument('-w','--word_file',help='Output feature file name',required=True)
    args=parser.parse_args()
    preprocess(args.input_file,args.feature_file,args.word_file)