from src.Tokenizer import BPETokenizer


# run test with python3 -m pytest -s from main directory


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
    tokenizer.bpe(tokenizer.convertTextBlockToTokens(teststr1))
    print([chr(x) for x in tokenizer.encodeWord("aaabdaaabac")])
    #visual confirmation of vocabuary being constructed
    #this requires printing to be enabled with the -s tag
    assert True == True
    
def test_MaxPair():

    #testing maxpair -> test returns aa
    #findMaxPair now takes a list of chunks; this case is a single chunk, so wrap it.
    #the merge itself is the per-chunk primitive mergePairInChunk.
    newList,newResultList = setupWikipediaTestCase()
    maxpair = BPETokenizer.findMaxPair([newList])
    newList = BPETokenizer.mergePairInChunk(newList,maxpair,ord('Z'))
    maxpair2 = BPETokenizer.findMaxPair([newList])
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

    
    
    #these cases test the per-chunk merge primitive directly (flat list in/out)
    for x in range(len(testCaseLists)):
        assert BPETokenizer.mergePairInChunk(testCaseLists[x],testCaseTargets[x],testCasenewToken[x]) == testCaseExpectedResult[x]


def test_baseVocabIsLatin1():
    # vocab starts as the 256 codepoints 0-255, with id == codepoint
    tokenizer = BPETokenizer("base")
    assert len(tokenizer.vocab) == 256
    assert len(tokenizer.unique_chars) == 256
    assert tokenizer.vocab[65] == "A"
    assert tokenizer.inversevocab["A"] == 65
    assert tokenizer.vocab[0] == chr(0)
    assert tokenizer.vocab[255] == chr(255)


def test_newCharacterRegisteredOnceAcrossCalls():
    # a char outside the base range should be added exactly once and keep the
    # same id across repeated convertTextBlockToTokens calls.
    # regression: unique_chars was checked but not updated, so the same char got
    # a fresh id every call -> duplicate vocab entries.
    tokenizer = BPETokenizer("newchar")
    tokenizer.convertTextBlockToTokens("中")
    firstId = tokenizer.inversevocab["中"]
    tokenizer.convertTextBlockToTokens("中")
    secondId = tokenizer.inversevocab["中"]

    assert firstId == secondId
    # exactly one vocab entry maps back to the char
    assert [tid for tid, ch in tokenizer.vocab.items() if ch == "中"] == [firstId]


def test_registerNewChar():
   # Note this does not hold true for all new characters if you do conversions after merges.
    tokenizer = BPETokenizer("sync")
    assert len(tokenizer.unique_chars) == 256
    tokenizer.convertTextBlockToTokens("界")
    newId = tokenizer.inversevocab["界"]

    assert newId == 256                      # appended right after the base 0-255 range
    assert tokenizer.vocab[newId] == "界"
    assert "界" in tokenizer.unique_chars
    assert len(tokenizer.unique_chars) == len(tokenizer.vocab)


def test_newCharacterAfterTrainingDoesNotCollide():
    # convert run after bpe must give the new char an id past the merge ids.
    # convertTextBlockToTokens uses len(vocab) for the id, so it does not reuse
    # ids that bpe already handed out to merge tokens.
    tokenizer = BPETokenizer("collide")
    tokens = tokenizer.convertTextBlockToTokens("aaaa")
    tokenizer.bpe(tokens)                       # adds merge tokens to vocab only
    mergeIds = set(tokenizer.merges.values())
    assert mergeIds                             # sanity: bpe actually produced a merge

    tokenizer.convertTextBlockToTokens("中")     # new char, id = len(vocab)
    newId = tokenizer.inversevocab["中"]

    # the new char must not overwrite an existing merge token
    assert newId not in mergeIds
    assert tokenizer.vocab[newId] == "中"
