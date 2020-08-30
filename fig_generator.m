function  fig_generator
clear;
clc;
close all;

LINECOL = [0.85,0.33,0.10];
HISTCOL = 0.6*[1 1 1];
rng(2300);

beta = 0.5;
kinit = 0.9;
gamma = 0.4;

alfa = 0.02;

nCells = 10000; %set to 100000 to reproduce figure
bs = 12;
S_burst = [bs 0; -1 1; 0 -1]; %
nT = 3;
kpar = [kinit,beta,gamma];
Tmax=10/min(kpar);
tvec = linspace(0,Tmax,nT);
t_matrix = repmat(tvec,nCells,1);


f=figure(1);
f.Position = [375.4000 297 712 329.6000];

X = gillespie_burst(kpar,t_matrix,S_burst,nCells);

for i = 1:2
    ax(i) = subplot(2,3,i);
    H=histogram(X(:,end,i),'BinMethod','integers','Normalization','pdf',...
        'EdgeColor','none','FaceColor',HISTCOL);hold on;
    x = H.BinEdges(2:end)-0.5;
    if i==1
        y = nbinpdf(x,kinit/beta,1/(1+bs));
        title('a. Nascent intrinsic + \color[rgb]{0.85,0.33,0.10}NB','FontWeight','Normal');
        xlabel('Copy number'); ylabel('Probability');
    elseif i==2
        nbfit = nbinfit(X(:,end,i));
        y = nbinpdf(x,nbfit(1),nbfit(2));
        title('b. Mature intrinsic + \color[rgb]{0.85,0.33,0.10}NB fit','FontWeight','Normal');
        xlabel('Copy number');
        ylim([0 0.034]);
    end
    plot(x,y,'LineWidth',2,'Color',LINECOL);

end

ax(3) = subplot(2,3,3);
scatter(X(:,end,1),X(:,end,2),5,'k','filled','MarkerFaceAlpha',alfa);
title('c. Joint intrinsic','FontWeight','Normal');
xlabel('Nascent'); ylabel('Mature');


%%%%%%%%%%%%%%%%%
S_ext = [1 0; -1 1; 0 -1];
gamma_a = kinit/beta;
gamma_b = 1/(beta*bs);

prob_nas = gamma_b*beta/(gamma_b*beta+1);
prob_mat = gamma_b*gamma/(gamma_b*gamma+1);
gamma_params = [gamma_a,gamma_b];
X = gillespie_gamma_K(kpar,t_matrix,S_ext,nCells,gamma_params);
for i = 1:2
    subplot(2,3,i+3);
    H=histogram(X(:,end,i),'BinMethod','integers','Normalization','pdf',...
        'EdgeColor','none','FaceColor',HISTCOL);hold on;
    x = H.BinEdges(2:end)-0.5;
    if i==1
        y = nbinpdf(x,gamma_a,prob_nas);
        title('d. Nascent extrinsic + \color[rgb]{0.85,0.33,0.10}NB','FontWeight','Normal');
        xlabel('Copy number'); ylabel('Probability');
    elseif i==2
        y = nbinpdf(x,gamma_a,prob_mat);
        title('e. Mature extrinsic + \color[rgb]{0.85,0.33,0.10}NB','FontWeight','Normal');
        xlabel('Copy number');
    end
    subplot(2,3,i+3);
    plot(x,y,'LineWidth',2,'Color',LINECOL);
    axis(axis(ax(i)));
end
subplot(2,3,6)
scatter(X(:,end,1),X(:,end,2),5,'k','filled','MarkerFaceAlpha',alfa);
title('f. Joint extrinsic','FontWeight','Normal');
xlabel('Nascent'); ylabel('Mature');
axis(axis(ax(3)));


return