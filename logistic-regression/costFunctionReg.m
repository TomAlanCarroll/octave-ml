function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Reuse the existing cost function
[J, grad] = costFunction(theta, X, y);

theta_size = size(theta);

% Incorporate regularization by adding a penalty for thetas
J_penalty = (lambda / (2 * m)) * sum(theta(2:theta_size) .^ 2);
grad_penalty = (lambda / m) * theta(2:theta_size);

% Penalize the cost
J = J + J_penalty;

grad_size = size(grad);

% Calculate the gradient
grad(2:grad_size) = grad(2:grad_size) + grad_penalty;

% =============================================================

end
