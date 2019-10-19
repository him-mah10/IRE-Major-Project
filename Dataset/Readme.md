<h1>This repository contains all the code and the approach which we used to generate dataset.</h1> 
<h2>Steps</h2>
<ul>
  <li>First we extract all the unique tweet ids from their corpus. The script to do this task was created on the fly. 
  <li>Then we dowload all the tweets using tweet ids using get_twitter.py script. We stop if rate limit exceed.
  <li>Then we pre process using tokenizer provided in ECT dataset and then do mapping based on tweet ids.
</ul>
<h2>Results</h2>
We got 1995 tweets out of 2994 tweet ids. Some tweets were missing because either their owners account was deleted or the tweet was deleted or we didn't have the permission to see them.
