from datetime import datetime

from src.Tokenizer import BPETokenizer



# editable install using pip. filed used for this comes from https://github.com/nathanter/newsarticlescraping/tree/main
import newsarticleapi.collect as nac

## builds tokenizer using scraped substack, hn and news articles.
def main():
    # load recent returns a dict[string,list[dict]]
    data = nac.loadArticles(ArticleLimitPerSource = 200)
    data["substack"] = nac.balanceByTag(data["substack"],TagQuota = 25)

    ## format

    """
    [
        {
            title: "title",
            tags: ["tag1","tag2"]
            source: "link"
            author: "Author Name"
            date: "date"
            text: "LONG TEXT"
        }
    ]
    """
    allEntriesFlat = []
    allTexts = []   
    #flattening
    for x in data.values():
        for entry in x:
            allEntriesFlat.append(entry)
            allTexts.append(entry.get("text",""))
    


    
    # name the model after the run time so each build gets its own files in
    # tokenizerModels instead of overwriting the last one. no colons (awkward in
    # shells) and the format sorts lexicographically = chronologically, same as
    # the scraper's dated json filenames
    tokenizer = BPETokenizer(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    input = tokenizer.tokenizeOnMultipleFilesBeforeTraining(allTexts)
    tokenizer.bpe(input)


    saved = tokenizer.saveModel()
    print(f"saved model to {saved}")
    tokenizer.display()


    
        


if __name__ == "__main__":
    main()
