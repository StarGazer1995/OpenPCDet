# Development Log

## 29.Jul 	Starting to implement the multi-resolution grouping(MRG)

​	Algorithm:
​					Paper of the PointNet++ indicates that the features in the level $  L_{i}  $ is a concatenation of two vectors.
​					One of the vectors is obtained in the lower level while the another one is the higher level.
​	Some Thought:
​					It is a bottom-up manner. The abstracted set, from bottom to up, would be key-point level, x_conv1, ..., BEV 					level. From the code of MSG, we could assume that the MRG could be coded in a similar way.

​	Challenges:
​					There are serious of **problems**, **what is the meaning of MLP**? It is obvious that the **MLP is for**
​					**grouping in multiple scale**, but **how** it works in multi-resolution? An intuition thought is that the **MLP **
​					**is used for linking the different levels**. But **which points** should be linked? A ball query would be
​					proposed. But the ball radius need to be decided.
