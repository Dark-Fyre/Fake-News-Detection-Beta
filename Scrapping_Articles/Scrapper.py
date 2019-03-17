from newspaper import Article
from GNews import GNews
import pandas as pd 


#***************************************************** CORE FILE *****************************************************
def Articles_from_Keywords(ls_Keywords):
        """Function takes in a list of keywords and searches 
           on google news for similar articles scrapes them and returns
           a dictionary containing the article text and title."""


        #obtains the link of news articles having similar keywords
        gnews_object = GNews()
        gnews_object.search(ls_Keywords)
        gnews_object.getpage()
        urls = gnews_object.getlink()


        #Scrapes the articles for text and title from obtained links
        article_title=[]
        article_text=[]
        for url in urls:
                article = Article(url,language="en")
                article.download()
                article.parse()
                article.nlp()
                article_title.append(article.title)
                article_text.append(article.text)
        article_info = [article_title,article_text]
        df = pd.DataFrame(article_info)
        df.to_json("scrapped_data.json")