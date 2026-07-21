import argparse
from collections import deque
import pickle



class BPETokenizer:

    def __init__(self,name : str,maxvocab = 10000):
        ## sets up list of all chars theoretically expressable by one byte. (256 ASCII)
        self.name = name
        self.maxvocab = maxvocab
        self.unique_chars = [chr(i) for i in range(256)]

        
        self.vocab: dict[int, str] = {}
        self.vocab = {i: char for i, char in enumerate(self.unique_chars)}

        self.inversevocab = {char: i for i, char in enumerate(self.unique_chars)}
        self.allowedSpecials = []
        self.merges: dict[tuple[int, int], int] = {}
        # maps two token ids to new token id
        self.mergeprio: dict[tuple[int,int],int] = {} 
        #does not map to new token, maps to merge order. 
        #I'm keeping this seperate because I want the ability to independetly change merge priority to prioritize scientific language
        #The process for the above would mean
        #taking word -> use vocab to find byte. inverse search through merges and then reassigning mergeprio.


        self.isFullVocab = False
        
    
    def initSpecialTokens(self, special_tokens:list[str] = None): 
        if special_tokens != None:
            self.allowedSpecials = special_tokens
            for x in special_tokens:
                newId = len(self.unique_chars)
                self.unique_chars.append(x)
                ## update vocab with new chars
                self.vocab[newId] = x

                #consider storign these seperately. if I encounter special tokens in text I might not want to replace them
                self.inversevocab[x] = newId


    def pretokenize(self,text :str) -> list[str]:
        #chunking process
        chunks = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                chunks.append("\n")
            words = line.split()
            for j, word in enumerate(words):
                if j == 0 and i > 0:
                    chunks.append(" " + word)
                elif j == 0:
                    chunks.append(word)
                else:
                    chunks.append(" " + word)

        return chunks
    def convertTextBlockToTokens(self,text :str): 
        processed_text = []
        ## text needs to be normalized here
        
        
        normalizedText = text.strip()
        chunks = self.pretokenize(normalizedText)
        for char in normalizedText:
            processed_text.append(char)

        ## end of text normalization
        ## check for unique characters
        for char in sorted(set(processed_text)):
            if char not in self.unique_chars:
                new_id = len(self.vocab)
                ## update vocab with new chars
                self.vocab[new_id] = char
                self.inversevocab[char] = new_id

        ## converting 
      
        tokens = []
        for word in chunks:
            chunkInTokens = []
            for chr in word:
                chunkInTokens.append(self.inversevocab[chr])
            tokens.append(chunkInTokens)

        return tokens
    
    def train(self, tokens) -> None:
        #general plan:
        #tokenize everything according to default character mapping
        #sliding window pass to count max pairs.
        #update vocab as needed
        ## check for max pairs -> replace -> break if no options
        for i in range(len(self.vocab),self.maxvocab):
            resultPair = self.findMaxPair(tokens)
            if resultPair == None:
                break
            else: 
                #update with result
                newTokenid = len(self.vocab)
                tokens = BPETokenizer.updateTokensRemovePair(tokens,resultPair,newTokenid)

                #change in tokens list

                self.vocab[newTokenid] = self.vocab[resultPair[0]] + self.vocab[resultPair[1]]
                self.inversevocab[self.vocab[resultPair[0]] + self.vocab[resultPair[1]]] = newTokenid

                #append changes to merges list
                #do I need a merges list?
                self.merges[resultPair] = newTokenid
                self.mergeprio[resultPair] = len(self.mergeprio)
 
        if len(self.vocab) > self.maxvocab:
            self.isFullVocab = True
    
             

    
    @staticmethod
    def updateTokensRemovePair(chunks:list[list[int]],pair:tuple[int,int],newToken:int) -> list[list[int]]:
        #apply one merge across every chunk. pairs never span a chunk boundary,
        #so each chunk is merged independently (mirrors findMaxPair's per-chunk counting)
        return [BPETokenizer.mergePairInChunk(chunk,pair,newToken) for chunk in chunks]

    @staticmethod
    def mergePairInChunk(tokens:list[int],pair:tuple[int,int],newToken:int) -> list[int]:
        queue = deque(tokens)
        newTokens = []

        while queue:
            next = queue.popleft()
            if queue and (next,queue[0]) == pair:
                newTokens.append(newToken)
                queue.popleft()
            else:
                newTokens.append(next)

        return newTokens

    @staticmethod
    def updateDictWithPairsFromChunk(tokens:list[int], counts : dict[tuple[int,int],int]):
            for pairStart in range(len(tokens) - 1):
                curpair = (tokens[pairStart], tokens[pairStart + 1])
                counts[curpair] =  counts.get(curpair, 0) + 1

    @staticmethod
    def findMaxPair(tokens: list[list[int]]) -> tuple[int, int] | None:
        paircounts: dict[tuple[int, int], int] = {} #pairs a set of tokens appearing in subsequent order and the times they appear

        for chunk in tokens:
            BPETokenizer.updateDictWithPairsFromChunk(chunk,paircounts)

        if not paircounts:
            return None
        else:
            maxpair = max(paircounts, key=paircounts.get)
            if paircounts[maxpair] <= 1:
                return None
        
        return maxpair


    
    def tokenizeMultipleFiles(self,directory:str):
        # take multiple files of jsons. find text element and all other elements and concantate them together
        # run some function here that converts all json files to text result should be strings
        strings = []



        finalTokens = []
        for string in strings:
            finalTokens.extend(self.encode(strings))
            if "[EOP]" in self.allowedSpecials:
                finalTokens.append(self.inversevocab["[EOP]"])


    def specialTokensFromArticlesHandling(self, final_encoded_tokens ,textSource :str= None, textAuthor : str= None, textTags : list[str] = None):
        final_encoded_tokens = []
        if "[Source]" in self.allowedSpecials:
            if textSource == None:
                raise Exception("Source token allowed but not defined")
            else: 
                
                final_encoded_tokens.extend(self.encodeWord(textSource))
                final_encoded_tokens.append(self.inversevocab["[Source]"])
        
        if "[Author]" in self.allowedSpecials:
            if textAuthor == None:
                raise Exception("Author token allowed but not defined")
            else: 
            
                for x in textAuthor.split(" "):
                    final_encoded_tokens.extend(self.inversevocab(" "))
                    final_encoded_tokens.extend(self.encodeWord(x))
                
                final_encoded_tokens.append(self.inversevocab["[Author]"])

        if "[Tags]" in self.allowedSpecials:
            if textTags == None:
                raise Exception("Tags token allowed but not defined")
            else: 
            
                for x in textTags:
                    final_encoded_tokens.extend(self.encodeWord(" " + x))
                final_encoded_tokens.append(self.inversevocab["[Tags]"])
        return final_encoded_tokens
        
    def encode(self, text: str) -> list[int]:
        # process:
        # split text into words.
        # tokenize words individually
        # return full list of tokens
        final_encoded_tokens = []

        # special tokens:
        
        
        # splittings process
        chunks = self.pretokenize(text)


        
        for i,x in enumerate(chunks):
            final_encoded_tokens.extend(self.encodeWord(x))
    
        #
        #if "[EOP]" in self.allowedSpecials:
        #    final_encoded_tokens.append(self.inversevocab["[EOP]"])
        return final_encoded_tokens

            

    

    def encodeWord(self,word:str) -> list[int]:

        # process:
        # tokenize everything
        # find all possible pairs
        # take smallest rank one. 
        # Run replaces
        # no normalization here: we preserve case (train path also stopped
        # lowercasing), so encode and train agree and casing is not destroyed.
        tokens = []
        tokens = [self.inversevocab.get(char, None) for char in word]

        if None in tokens:
            raise Exception("encoding failed")
        
        
        # this bit was copied from Sebastion Raschka becuase I think it's cool to generate pairs like this
        while True:
            pairs = set(zip(tokens, tokens[1:]))
            if not pairs: break
            #if only one token left. Or if no mergable pairs
            maxOccurenceSoFar = float("inf")
            targetpair = None
            for x in pairs:
                if x in self.mergeprio:
                    if self.mergeprio[x] < maxOccurenceSoFar:
                        targetpair = x
                        maxOccurenceSoFar = self.mergeprio[x]

            if targetpair == None:
                break

            # replacement operation
            
            firstOfPair,secondOfPair = targetpair
            newtokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == firstOfPair and tokens[i+1] == secondOfPair:
                    newtokens.append(self.merges[targetpair])
                    i += 2
                else:
                    newtokens.append(tokens[i])
                    i += 1

            tokens = newtokens
        return  tokens


    def decode(self, tokens: list[int]) -> str:
        # inverse of encode:
        # each token id maps to a string in vocab (a base char, a merged piece,
        # or a special token). concatenate them back into text.
        # note: this is lossy vs the original input because training lowercases,
        # so casing is not recoverable.
        pieces = []
        for token in tokens:
            if token not in self.vocab:
                raise Exception(f"decoding failed: unknown token id {token}")
            pieces.append(self.vocab[token])
        return "".join(pieces)


    def saveModel(self) -> None:
        #simple saving and loading with pickle
        with open(f"tokenizerModels/{self.name}", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def loadModel(name: str) -> "BPETokenizer":
        #simple saving and loading with pickle
        with open(f"tokenizerModels/{name}", "rb") as f:
            return pickle.load(f)

    def display(self, path: str | None = None) -> None:
        #writes vocab, merges and merge priority to a file in a readable table format
        if path is None:
            path = f"{self.name}_display.txt"

        def show(token: int) -> str:
            #make control chars / whitespace visible instead of writing raw
            return repr(self.vocab[token])

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"=== Tokenizer '{self.name}' ===\n")
            f.write(f"vocab size: {len(self.vocab)}   maxvocab: {self.maxvocab}   full: {self.isFullVocab}\n")
            f.write("\n--- Vocab (id -> token) ---\n")
            for token_id in sorted(self.vocab):
                f.write(f"  {token_id:>6}  {show(token_id)}\n")
            f.write("\n--- Merges (pair -> new id) ---\n")
            if not self.merges:
                f.write("  (none)\n")
            else:
                for pair, new_id in self.merges.items():
                    left, right = pair
                    f.write(f"  ({left}, {right})  {show(left)} + {show(right)}  ->  {new_id}  {show(new_id)}\n")

            f.write("\n--- Merge priority (order applied, lower = earlier) ---\n")
            if not self.mergeprio:
                f.write("  (none)\n")
            else:
                for pair, prio in sorted(self.mergeprio.items(), key=lambda item: item[1]):
                    left, right = pair
                    f.write(f"  #{prio:<6} ({left}, {right})  {show(left)} + {show(right)}\n")
        
    
        



