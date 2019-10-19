import tweepy
file=open("ids",'r')
ids=file.readlines()
for i in range(len(ids)):
	ids[i]=ids[i].strip()
auth = tweepy.OAuthHandler("")
auth.set_access_token("","")
api=tweepy.API(auth)
corr_count=0
wrong_cout=0
file=open("temp10",'w')
for i in range(len(ids)):
	try:
		tweet=api.get_status(ids[i])
		aa=ids[i]+'\t'+tweet.text
		corr_count+=1
		# print("Correct: "+str(corr_count))
		print(aa)
		file.write(aa)
		file.write("\n")
		file.write("_______________________")
		file.write("\n")
	except Exception as e:
		wrong_cout+=1
		if e.__reduce__()[1][0][0]['message']=="Rate limit exceeded":
			print(e)
			break
		# print("Incorrect: "+str(wrong_cout))
		print(e)
		# print(e)
file.close()
print(i)
print(corr_count)
print(wrong_cout)
print(corr_count+wrong_cout)
