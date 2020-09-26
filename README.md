## Python Implementation of Probabilistic Matrix Factorization Algorithm 

The code attempts to implement the following paper:

> *Mnih, A., & Salakhutdinov, R. (2007). Probabilistic matrix factorization. In Advances in neural information processing systems (pp. 1257-1264).*

Probabilistic Matrix Factorization in Python with Ciao dataset


The dataset is a copy of the Epinions and Ciao
dataset in the `<https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm>`

### Train model
run script below, params are default you can change on main.py to setup model for explore model:
```
python main.py
```

### Result
| learning_rate|k_dim | RMSE | MAE|
|----------|:----------:|------:|------:|
| 0.001     |  16 | 1.05318 |0.81547|
| 0.005     |  16 | 1.05399 |0.81527|
| 0.01     |  16 | 1.05903 |0.81657|
| 0.05     |  16 | 1.06514 |0.82039|
| 0.05     |  8 | 1.06159 |0.82062|
| 0.01     |  8 | 1.06556 |0.82129|

### Reference:  
1. Mnih, A., & Salakhutdinov, R. (2007). Probabilistic matrix factorization. In Advances in neural information processing systems (pp. 1257-1264).  
2. Salakhutdinov, R. Probabilistic matrix factorization in Matlab. http://www.utstat.toronto.edu/~rsalakhu/BPMF.html.  
