function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X)(2)
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% unvectorized cost function
first_term=0;
for i=1:m,
	h=sigmoid(transpose(theta)*transpose(X(i, :)));
	first_term=first_term+(-y(i)*log(h)-(1-y(i))*log(1-h));
end
first_term=first_term/m;
second_term=0;
for j=2:n,
	second_term=second_term+theta(j)*theta(j);
end
second_term=second_term*(lambda/(2*m));

J=first_term+second_term;

% for i2=1:1000,

	pd_new=0;
	for i=1:m,
		h=sigmoid(transpose(theta)*transpose(X(i, :)));
		pd_new=pd_new+(h-y(i));
	end
	pd_new=pd_new/m;
	grad(1)=pd_new;

	for j=2:n,
		pd=0;
		for i=1:m,
			h=sigmoid(transpose(theta)*transpose(X(i, :)));
			pd=pd+(h-y(i))*X(i, j);
		end
		pd=pd/m;

		pd=pd+(lambda*theta(j))/m;

		grad(j)=pd;
	end


% end

% grad=theta;

% print(first_term)




% h=sigmoid(transpose(theta)*transpose(X));
% J=(-transpose(y)*transpose(log(h))-transpose(1-y)*transpose(log(1-h)))/m;


% =============================================================

end
