function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

%unvectorized
% finalSum=0;
% for i=1:m,
% 	finalSum+=(-y(i, 1)*log(sigmoid(X(i, :)))+(1-y(i))*log(1-sigmoid(X(i, :))));
% end
% J=finalSum/m;

%vectorized
h=sigmoid(transpose(theta)*transpose(X));
J=(-transpose(y)*transpose(log(h))-transpose(1-y)*transpose(log(1-h)))/m;
% print(size(theta))
% for i=1:size(theta),
% 	pd=0;
% 	for j=1:m,
% 		pd+=(sigmoid(X(j,i))-y(j))*X(j, i);
% 	end
% 	grad(i)=pd;
% end

grad=transpose(X)*(sigmoid(X*theta)-y);
grad=grad./m;

% =============================================================

end
