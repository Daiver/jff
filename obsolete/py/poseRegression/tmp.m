function [ferns,ysPr] = fernsRegTrain( data, ys, varargin )
% Train boosted fern regressor.
%
% Boosted regression using random ferns as the weak regressor. See "Greedy
% function approximation: A gradient boosting machine", Friedman, Annals of
% Statistics 2001, for more details on boosted regression.
%
% A few notes on the parameters: 'type' should in general be set to 'res'
% (the 'ave' version is an undocumented variant that only performs well
% under limited conditions). 'loss' determines the loss function being
% optimized, in general the 'L2' version is the most robust and effective.
% 'reg' is a regularization term for the ferns, a low value such as .01 can
% improve results. Setting the learning rate 'eta' is crucial in order to
% achieve good performance, especially on noisy data. In general, eta
% should decreased as M is increased.
%
% Dimensions:
%  M - number ferns
%  R - number repeats
%  S - fern depth
%  N - number samples
%  F - number features
%
% USAGE
%  [ferns,ysPr] = fernsRegTrain( data, hs, [varargin] )
%
% INPUTS
%  data     - [NxF] N length F feature vectors
%  ys       - [Nx1] target output values
%  varargin - additional params (struct or name/value pairs)
%   .type     - ['res'] options include {'res','ave'}
%   .loss     - ['L2'] options include {'L1','L2','exp'}
%   .S        - [2] fern depth (ferns are exponential in S)
%   .M        - [50] number ferns (same as number phases)
%   .R        - [10] number repetitions per fern
%   .thrr     - [0 1] range for randomly generated thresholds
%   .reg      - [0.01] fern regularization term in [0,1]
%   .eta      - [1] learning rate in [0,1] (not used if type='ave')
%   .verbose  - [0] if true output info to display
%
% OUTPUTS
%  ferns    - learned fern model w the following fields
%   .fids     - [MxS] feature ids for each fern for each depth
%   .thrs     - [MxS] threshold corresponding to each fid
%   .ysFern   - [2^SxM] stored values at fern leaves
%   .loss     - loss(ys,ysGt) computes loss of ys relateive to ysGt
%  ysPr     - [Nx1] predicted output values
%
% EXAMPLE
%  %% generate toy data
%  N=1000; sig=.5; f=@(x) cos(x*pi*4)+(x+1).^2;
%  xs0=rand(N,1); ys0=f(xs0)+randn(N,1)*sig;
%  xs1=rand(N,1); ys1=f(xs1)+randn(N,1)*sig;
%  %% train and apply fern regressor
%  prm=struct('type','res','loss','L2','eta',.05,...
%    'thrr',[-1 1],'reg',.01,'S',2,'M',1000,'R',3,'verbose',0);
%  tic, [ferns,ysPr0] = fernsRegTrain(xs0,ys0,prm); toc
%  tic, ysPr1 = fernsRegApply( xs1, ferns ); toc
%  fprintf('errors train=%f test=%f\n',...
%    ferns.loss(ysPr0,ys0),ferns.loss(ysPr1,ys1));
%  %% visualize results
%  figure(1); clf; hold on; plot(xs0,ys0,'.b'); plot(xs0,ysPr0,'.r');
%  figure(2); clf; hold on; plot(xs1,ys1,'.b'); plot(xs1,ysPr1,'.r');
%
% See also fernsRegApply, fernsInds
%
% Piotr's Computer Vision Matlab Toolbox      Version 2.50
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

% get/check parameters
dfs={'type','res','loss','L2','S',2,'M',50,'R',10,'thrr',[0 1],...
  'reg',0.01,'eta',1,'verbose',0};

type=type(1:3); 
N=length(ys);
% train stagewise regressor (residual or average)
fids=zeros(M,S,'uint32'); thrs=zeros(M,S);
ysSum=zeros(N,1); ysFern=zeros(2^S,M);
for m=1:M
  % train R random ferns using different losses, keep best
  ysTar=ys-ysSum; best={};

    for r=1:R
      [fids1,thrs1,ysFern1,ys1]=trainFern(data,ysTar,S,thrr,reg);
      e1=sum((ysTar-ys1).^2);
      if(e1<=e), e=e1; best={fids1,thrs1,ysFern1,ys1}; end
    end

  % store results and update sums
  assert(~isempty(best)); [fids1,thrs1,ysFern1,ys1]=deal(best{:});
  fids(m,:)=fids1; thrs(m,:)=thrs1;
  ysFern(:,m)=ysFern1*eta; ysSum=ysSum+ys1*eta;
  if(verbose), fprintf('phase=%i  error=%f\n',m,e); end
end
% create output struct
ferns=struct('fids',fids,'thrs',thrs,'ysFern',ysFern); ysPr=ysSum;
  ferns.loss=@(ys,ysGt) mean((ys-ysGt).^2);
end

function [fids,thrs,ysFern,ysPr] = trainFern( data, ys, S, thrr, reg )
% Train single random fern regressor.
[N,F]=size(data); mu=sum(ys)/N; ys=ys-mu;
fids = uint32(floor(rand(1,S)*F+1));
thrs = rand(1,S)*(thrr(2)-thrr(1))+thrr(1);
inds = fernsInds(data,fids,thrs);
ysFern=zeros(2^S,1); cnts=zeros(2^S,1);
for n=1:N, ind=inds(n);
  ysFern(ind)=ysFern(ind)+ys(n);
  cnts(ind)=cnts(ind)+1;
end
ysFern = ysFern ./ max(cnts+reg*N,eps) + mu;
ysPr = ysFern(inds);
end

function m = medianw(x,w)
% Compute weighted median of x.
[x,ord]=sort(x(:)); w=w(ord);
[~,ind]=max(cumsum(w)>=sum(w)/2);
m = x(ind);
end
