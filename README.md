# IRE-Major-Project

# Selective Expression For Event Coreference Resolution on Twitter

Event coreference is the problem of identifying and connecting mentions of the same events in different contexts. It is a fundamental problem in NLP with wide-spread applications.The given paper is the state-of-the-art for event coreference in Twitter. With the growth in popularity and size of social media, there is an urgent need for systems that can recognize the coreference relation between two event mentions in texts from social media.

Approach till now basically depend upon NLP features which restricts domain scalability and leads to propagation error.In this paper a novel selective expression approach based on event trigger to explore the coreferential relationship in high-volume Twitter texts is proposed.

Firstly a bidirectional Long Short Term Memory (Bi-LSTM) is exploited to extract the sentence level and mention level features. Then, to selectively express the essential parts of generated features, a selective gate is applied on sentence level features. Next, to integrate the time information of event mention pairs, an auxiliary feature is designed based on triggers and time attributes of the two event mentions. Finally, all these features are concatenated and fed into a classifier to
predict the binary coreference relationship between the event mention pair.

They also released a new dataset called EventCoreOnTweet(ECT) dataset on which they evaluated their methods.It annotates the coreferential relationship between event mentions and event trigger of each event mention. The experimental results demonstrate that the approach achieves significant performance in the ECT dataset.


First we need to take the data and make the training and testing set.
For this, run pre_processing.py file.

    python3 pre_processing.py

To train the model, we need model.py which contains our model and train.py to train our model.

For training the model :

    python3 train.py 
    
