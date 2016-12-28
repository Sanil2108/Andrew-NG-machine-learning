function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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
    h=X*theta;
    error_=h-y;
    gradient_=transpose(X)*error_;
    gradient_=gradient_*(alpha/m);
    theta=theta-gradient_;

    % for j=1:2,
    %     pd = 0;
    %     for k=1:m,
    %         pd = pd+(theta(2)*X(k, j)+theta(1)-y(k))*X(k, j);
    %         % h=(X*theta - y);
    %         % pd = pd+(h(k, 1));
    %     endfor
    %     pd=pd/m;

    %     theta(j)=theta(j)-alpha*pd;
    % endfor


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

endfor
theta(1)
theta(2)
J_history(num_iters)
end
