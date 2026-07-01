from Tokenizer import BPETokenizer


# run test with python -m pytest -s from main directory


def setupWikipediaTestCase() -> tuple[list[int], list[int]]:
    teststr2 = "aaabdaaabac"
    teststrresult2 = "ZabdZabac"
    newList = []
    newResultList = []
    for x in teststr2:
        newList.append(ord(x))
    for x in teststrresult2:
        newResultList.append(ord(x))
    
    return newList,newResultList


def test_fullTokenize():
    teststr1 = "aaabdaaabac"
    tokenizer = BPETokenizer("test name")
    tokenizer.train(teststr1)
    print([chr(x) for x in tokenizer.encodeWord("aaabdaaabac")])
    #visual confirmation of vocabuary being constructed
    #this requires printing to be enabled with the -s tag
    assert True == True
    
def test_MaxPair():

    #testing maxpair -> test returns aa
    newList,newResultList = setupWikipediaTestCase()
    maxpair = BPETokenizer.findMaxPair(newList)
    newList = BPETokenizer.updateTokensRemovePair(newList,maxpair,ord('Z'))
    maxpair2 = BPETokenizer.findMaxPair(newList)
    assert maxpair == tuple([ord('a'),ord('a')])

    # tiebreaker between "ab" token and "Za" token is broken arbitrarily. Max used in this implementation does it by insertion order
    assert maxpair2 == tuple([ord('Z'),ord('a')])

def test_PairReplace():
    #variable initizalization + test case 1
    testCaseLists = [[1,2,3,4,5]] 
    testCaseTargets = [(1,6)]
    testCasenewToken = [7]
    testCaseExpectedResult = [[1,2,3,4,5]]

    #test case 2 sourced from wikipeida
    newList, newResultList = setupWikipediaTestCase()

    testCaseLists.append(newList)
    testCaseTargets.append((ord('a'),ord('a')))
    testCasenewToken.append(ord('Z'))
    testCaseExpectedResult.append(newResultList)

    
    
    for x in range(len(testCaseLists)):
        assert BPETokenizer.updateTokensRemovePair(testCaseLists[x],testCaseTargets[x],testCasenewToken[x]) == testCaseExpectedResult[x]
