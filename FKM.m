function [U, P, J] = FKM(X, K, m, conv, maxit, stand)
%
% CITATION:
% Maria Brigida Ferraro, Marco Forti, Paolo Giordani
% FKML0: a Matlab routine for sparse fuzzy clustering
% 2024 IEEE International Fuzzy Systems Conference, Yokohama, Japan, 2024
%
% INPUT
% X:        data matrix
% K:        number of clusters
% m:        parameter of fuzziness(e.g. = 2)
% conv:     convergence criterion (eg 1e-6)
% maxit:    maximum number of iterations (e.g. = 1e+3)
% stand:    standardization (if =1, data are standardized)
%
% OUTPUT
% U:        fuzzy membership degree matrix
% P:        centroid matrix
% J:        loss function vector (last element = value at convergence) 
%

[N,S] = size(X); 

%% standardization
if stand == 1
	Jm = eye(N) - (1/N)*ones(N);
	X = Jm*X/diag(std(X,1));
end

%% Initialization
P = zeros(K,S);
D = zeros(N,K);
Y = rand(N,K);
U = zeros(N,K);
for i = 1:N
    U(i,:) = Y(i,:)/sum(Y(i,:));
end

%% Optimization
iter = 0;
J = zeros(maxit, 1);  
while iter < maxit 
    iter = iter+1;
    % Update of the prototypes 
    for r = 1:K
		P(r,:) = ((U(:,r).^m)'*X)/sum((U(:,r).^m));
    end
    % Update of the membership degrees
    for i = 1:N
        for r = 1:K
            D(i,r) = sum((X(i,:)-P(r,:)).^2);
        end
    end
	SUM = sum((1./D).^(1/(m-1)),2);
    for i = 1:N
        for r = 1:K
            U(i,r) = (1/(D(i,r))).^(1/(m-1))/SUM(i);
        end
    end
    J(iter) = sum(sum(U.^m .* D)) + lambda * nnz(U>0);

	% Convergence
	fprintf('Iteration %d: J = %.4f\n', iter, J(iter));	
    if iter > 1 && abs(J(iter) - J(iter-1)) < conv
        J = J(J>0);
		break;
    end
end