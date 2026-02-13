# This-repository-contains-the-implementation-of-Linear-Regression
Ankitha Ammu 



Problem StatementThe goal of this project is to implement linear regression from scratch using Gradient Descent. The results are compared against the Normal Equation (Closed-form solution) to verify accuracy.+2DatasetTarget Function: $y = 3 + 4x + \epsilon$.Sample Size: 200 samples with $x \in [0,5]$.Noise: Gaussian noise ($\epsilon$) was added to the data generation process.Methods Used1. Closed-form Solution (Normal Equation)I implemented the analytical approach to find the optimal parameters $(\theta)$ using the formula:$$\theta = (X^T X)^{-1} X^T y$$.
This provides a baseline for the intercept and slope to compare against the iterative method.+22. Gradient DescentI implemented an iterative optimization algorithm to minimize the cost function.Parameters: Initialized $\theta = [0,0]$ with a learning rate ($\eta$) of 0.05.Iterations: 1000 total iterations.Gradient Formula: $\nabla_{\theta}J(\theta) = \frac{2}{m}X^T(X\theta - y)$.



import numpy as np
import matplotlib.pyplot as plt

# 1. Dataset Generation 

np.random.seed(42) 
m = 200             
X = 5 * np.random.rand(m, 1)  

# y = 3 + 4x + Gaussian noise 
y = 3 + 4 * X + np.random.randn(m, 1)

# Add a bias column of 1's to X 
X_b = np.c_[np.ones((m, 1)), X] 


# 2. Closed-form solution

# Compute theta = (X^T * X)^-1 * X^T * y 
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Print the estimated intercept and slope 
intercept_cf, slope_cf = theta_best[0][0], theta_best[1][0]
print("--- Closed-form Solution ---")
print(f"Intercept: {intercept_cf:.4f}")
print(f"Slope: {slope_cf:.4f}\n")


# 3. Gradient Descent Implementation 

eta = 0.05          
n_iterations = 1000 
theta = np.array([[0.0], [0.0]])  
loss_history = []  

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients 
    mse = np.mean((X_b.dot(theta) - y)**2)
    loss_history.append(mse)


intercept_gd, slope_gd = theta[0][0], theta[1][0]
print("--- Gradient Descent Solution ---")
print(f"Intercept: {intercept_gd:.4f}")
print(f"Slope: {slope_gd:.4f}")


 

plt.figure(figsize=(14, 6))

 
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='lightgray', label='Raw data points') 
plt.plot(X, intercept_cf + slope_cf * X, color='red', label='Closed-form line', linewidth=2) # [cite: 90]
plt.plot(X, intercept_gd + slope_gd * X, color='blue', linestyle='--', label='GD line', linewidth=2) # [cite: 91]
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Fitted Lines')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(range(n_iterations), loss_history, color='green')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Gradient Descent Loss Curve')

plt.tight_layout()
plt.show()
