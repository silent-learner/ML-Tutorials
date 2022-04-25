import numpy as np
import matplotlib.pyplot as plt

b = np.array([0.0,0.0])
alpha = 0.01
m = 40

def prediction(x):
    ans = b[0] + (b[1]*x)
    return ans

def predict():
    global Y_predict
    Y_predict = np.array([])
    for x in X:
        Y_predict = np.append(Y_predict,b[0]+(b[1]*x))

def update_params():
    global b
    S1 = np.sum(Y_predict-Y)
    S2 = np.sum((Y_predict-Y)*X)
    # print(S1)
    b[0] = b[0] - (alpha*(1/m)*(S1))
    b[1] = b[1] - (alpha * (1 / m) * (S2))

def cost():
    j = (1/2*m)*(np.sum((Y_predict-Y)**2))
    return j

X = np.arange(1,20,0.5)
Y = np.array([])

for x in X:
    Y = np.append(Y,2*x + (np.random.random_sample()))

Y_predict = np.array([])
predict()
jold = cost()



while True:
    print("B",b)
    update_params()
    predict()
    jnew = cost()
    if abs(jnew-jold) < 1e-5:
        print(jnew)
        print(jold)
        break
    jold = jnew

val = 13.4
print("Ans : ",prediction(val))
plt.scatter(X, Y, marker="+")
plt.plot(X, Y_predict, 'r--')
plt.show()

