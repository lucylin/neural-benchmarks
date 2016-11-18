import sys
import xml.etree.ElementTree as etree

def parse_file(filename):
    tree = etree.parse(filename)
    # iterate over documents
    for doc in tree.getroot():
        # iterate over paragraphs (<P> elements)
        for paragraph in doc.find('TEXT'):
            text = paragraph.text
            # replace newlines with spaces
            text = ' '.join(text.split())
            print(text)

if __name__=="__main__":
    parse_file(sys.argv[1])
            
