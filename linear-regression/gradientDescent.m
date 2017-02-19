function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % Calculate hypothesis
    predictions = X * theta;

    % Calculate these once
    term1 = alpha * (1 / m);
    term2 = predictions - y;

    % Update theta
    theta(1) = theta(1) - (term1 * sum(term2)); % multiplying by X(:, 1) is not necessary since it is always 1
    theta(2) = theta(2) - (term1 * sum(term2 .* X(:, 2)));

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
