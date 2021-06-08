# probabilistic-matrix-factorization

My implementation of the following paper:
*Mnih, A., & Salakhutdinov, R. (2007). Probabilistic matrix factorization. In Advances in neural information processing systems (pp. 1257-1264).*

MovieLens, the Netflix movie dataset is used. 
If you want, you can use other datasets of the same format. Please check main.py for testing details.
`MovieLens 100k <http://files.grouplens.org/datasets/movielens/ml-100k.zip/>`

For authors' own implementation, please check

`https://www.cs.toronto.edu/~rsalakhu/BPMF.html`

For hyperparameter adjustments, please check the paper itself.
However, I found the followings effective enough:
For PMF: LR=2.5, momentum=0.8, lambda=0.25
For CPMF: LR=50, momentum=0.8, lambda=0.01