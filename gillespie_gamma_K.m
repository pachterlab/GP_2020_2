function X_mesh = ...
    gillespie_gamma_K(k,t_matrix,S,nCells,gamma_params)
k = repmat(k,[nCells,1]);
k(:,1) = gamrnd(gamma_params(1),1/gamma_params(2),nCells,1);
num_t_pts = size(t_matrix,2);

X_mesh = NaN(nCells,num_t_pts,2); 

t = zeros(nCells,1); 
tindex = ones(nCells,1);

%initialize state: integer unspliced, integer spliced 
X = zeros(nCells,2);

%initialize list of cells that are being simulated
simindices = 1:nCells;
activecells = true(nCells,1);
%define offset time to randomize starting position of deterministic
%initialization rate

while any(activecells)
    mu = zeros(nCells,1);
    
    [t_upd,mu_upd] = rxn_calculator(...
        X(activecells,:),...
        t(activecells),...
        k(activecells,:),...
        sum(activecells));

    t(activecells) = t_upd;
    mu(activecells) = mu_upd;
    
    linindupdate = sub2ind(size(t_matrix),(1:length(tindex(activecells)))',...
        tindex(activecells));
    tvec_time = t_matrix(linindupdate);
    update = false(nCells,1);
    update(activecells) = t(activecells)>tvec_time;
    
    while any(update)
        tobeupdated = find(update);
        for i = 1:length(tobeupdated)
            X_mesh(simindices(tobeupdated(i)),tindex(tobeupdated(i)),1:2) = ...
                X(tobeupdated(i),:);            
        end
        tindex = tindex+update;
        ended_in_update = tindex(update)>num_t_pts;

        if any(ended_in_update)
            ended = tobeupdated(ended_in_update);
            
            activecells(ended) = false;
            mu(ended) = 0;

            if ~any(activecells),break;end
        end
        
        linindupdate = sub2ind(size(t_matrix),(1:length(tindex(activecells)))',...
            tindex(activecells));
        tvec_time = t_matrix(linindupdate);
        update = false(nCells,1);
        update(activecells) = t(activecells)>tvec_time;

    end
    
    z_ = find(activecells);
    X(z_,:) = X(z_,:) + S(mu(z_),:);
end
return


function [t,mu,k] = rxn_calculator(X,t,k,nCells)
nRxn = 3;

a = zeros(nCells,nRxn);

% a is propensity matrix
% reactions:
% production
% conversion
% death

kinit = k(:,1);
beta = k(:,2);
gamma = k(:,3);

a(:,1) = kinit;
a(:,2) = beta .* X(:,1);
a(:,3) = gamma .* X(:,2);

a0 = sum(a,2);
dt = log(1./rand(nCells,1))./a0;
t = t + dt;

r2ao = a0.*rand(nCells,1);
mu = sum(repmat(r2ao,1,nRxn+1) >= cumsum([zeros(nCells,1),a],2),2);
return
