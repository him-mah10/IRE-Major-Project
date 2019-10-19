from ECT.tokenizer import *
from ECT.tokenizer_wrapper import *
from collections import defaultdict
f= open("temp",mode='r',encoding='utf8')
temp=""
dic=defaultdict(str)
for i in f.readlines():
	i=i.strip()
	if i != "_______________________":
		temp=temp+" "+i.strip()
	else:
		temp=temp.strip()
		temp=temp.split("\t")
		dic[temp[0]]=temp[1]
		temp=""
for i in dic:
	line=dic[i]
	line.encode('utf8')
	s = " ".join(tokenize(line[:-1]))
	s = tokenize_and_join_tweet(s)
	s=re.sub(r'[^\x00-\xf3]', '', s)
	dic[i]=s.strip()

f.close()
file=open("corpus.txt",'w')
f= open("./ECT/corpus.txt",'r')
for i in f.readlines():
	i=i.strip()
	i=i.split('\t')
	i[5]=dic[i[1]]
	i="\t".join(i)
	file.write(i)
	file.write("\n")