import sys
import xml.etree.ElementTree as etree
from nltk.tokenize import sent_tokenize, casual_tokenize

LEN_LIMIT = 100

def parse_file(filename):
    tree = etree.parse(filename)
    # iterate over documents
    for doc in tree.getroot():
        # iterate over paragraphs (<P> elements)
        for paragraph in doc.find('TEXT'):
            text = paragraph.text
            # Tokenize and split into sentences
            for s in sent_tokenize(text):
                tok = casual_tokenize(s.replace("\n"," "))
                if len(tok) < LEN_LIMIT:
                    print(" ".join(tok))

if __name__=="__main__":
    parse_file(sys.argv[1])
            
