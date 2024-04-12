function [U, P, J] = FKML0(X, K, m, lambda, conv, maxit, stand)
%
% CITATION
% Maria Brigida Ferraro, Marco Forti, Paolo Giordani
% FKML0: a Matlab routine for sparse fuzzy clustering
% 2024 IEEE International Fuzzy Systems Conference, Yokohama, Japan, 2024
%
% INPUT
% X:        data matrix
% K:        number of clusters
% m:        parameter of fuzziness (e.g. = 2)
% lambda:   parameter of the L0 regularization term (>= 0)
% conv:     convergence criterion (e.g. = 1e-6)
% maxit:    maximum number of iterations (e.g. = 1e+3)
% stand:    standardization (if = 1, data are standardized)
%
% OUTPUT
% U:        membership degree matrix
% P:        prototype matrix
% J:        loss function vector (last element = value at convergence) 
%

[N, S] = size(X); 

%% standardization
if stand == 1
	Jm = eye(N)-(1/N)*ones(N);
	sd = diag(std(X,1));
	X = Jm*X/sd;
end

%% Initialization
P = zeros(K,S);
D = zeros(N,K);
Y = rand(N,K);
U = zeros(N,K);
for j = 1:N
    U(j,:) = Y(j,:)/sum(Y(j,:));
end

%% Optimization
iter = 0;
J = zeros(maxit, 1);  
while iter < maxit 
	iter = iter + 1;
    % Update of the prototypes 
    for i = 1:K
		P(i,:) = ((U(:,i).^m)'*X)/sum((U(:,i).^m));
    end
    % Update of the membership degrees
    epsilon = 1e-10; 
	for j = 1:N
		for i = 1:K
            D(j,i) = max(sum((X(j,:)-P(i,:)).^2), epsilon);
		end
	end
	SUM = sum((1./D).^(1/(m-1)),2);
    for j = 1:N
        for i = 1:K
            U(j,i) = (1/(D(j,i))).^(1/(m-1))/SUM(j);
        end
        checkrj = 1;
        Uopt = U(j,:);
        U0j = U(j,:);
        p0j = (U0j > 0); 
        while checkrj == 1
            Uaux = U0j;
            Uaux(not(p0j)) = 2;
            [~,mj] = min(Uaux);
            p0j(mj) = 0;
            U0j(mj) = 0;
            SUM0 = sum(p0j.*(1./D(j,:)).^(1/(m-1)),2);
            for i = 1:K 
                U0j(i) = p0j(i)*((1/(D(j,i))).^(1/(m-1))/SUM0);
            end
            if sum(U0j.^m .* D(j,:))+lambda*sum(p0j)>sum(Uopt.^m.*D(j,:))+lambda*nnz(Uopt)
				checkrj = 0; 
            else 
                Uopt = U0j;
                if sum(p0j) == 1 
                    checkrj = 0;
                end
            end
        end
        U(j,:) = Uopt;
    end
    J(iter) = sum(sum(U.^m.*D))+lambda*nnz(U>0);

	% Convergence
	fprintf('Iteration %d: J = %.4f\n', iter, J(iter));
	if iter > 1 && abs(J(iter)-J(iter-1)) < conv
        J = J(J>0);
		break;
	end
	if stand == 1
		meanval = mean(X);
		P=P*sd+meanval;
	end
end