import re

def findWholeWord(w):
    print '{}'.format(w)
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

print findWholeWord('seek')('those who seek shall find')
 

print findWholeWord('word')('swordsmith')   