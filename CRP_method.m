function [crp_parms, crp_projs]=CRP_method(V,t_win)
% function [crp_parms, crp_projs]=CRP_method(V,t_win)
%
% This function performs canonical response parameterization 
%
%
% Input variables:
%
% V: single-trial stimulation-evoked voltage matrix (Dimensions of V are
%    T×K, with T total timepoints by K total stimulation trial events)
%
% t_win: time matrix, with elements corresponding to time that each sample
%        corresponds to (dimensions of t_win are 1xT)
%
%
% Output variables: 
%
% structure "crp_projs", with fields:
%     proj_tpts: time points that projections were calculated at
%     S_all: full set of projection magnitudes at all durations
%     mean_proj_profile: mean projection magnitude profile
%     var_proj_profile: variance in projection magnitude profile
%     tR_index: index into proj_tpts (projection magnitude times) corresponding to response duration
%     t_value_tR: t-statistic for projection magnitudes at response duration (\tau_R)
%     p_value_tR: p-value at response duration, \tau_R (extraction significance) by t-test
%     t_value_full: t-statistic for projection magnitudes at full time sent in    
%     p_value_full: p-value at  full time sent in (extraction significance) by t-test
%     stat_indices: indices used to calculate statistics (half of distribuition, with comparison pairs non-overlapping) 
%
% structure "crp_parms", with fields:
%     V_tR: Reduced length voltage matrix (to response duration)
%     al: alpha coefficient weights for C into V
%     C: canonical shape, C(t)
%     ep: residual epsilon after removal of form of CCEP  
%     tR: response duration in units seconds 
%     parms_times: times for parameterized data
%     avg_trace_tR: simple average trace, truncated to response duration
%     al_p: alpha-prime - alpha scaled to sqrt(number of samples),
%     epep_root: scalar summarizing residual for each trial
%     Vsnr: "signal to noise" for each trial
%     expl_var: explained variance by parameterization for each trial
%
%
% Sub-functions contained in this function (at end of script):
% 
%    function S0=ccep_proj(V) %% Projections function
%
%    function [E,S]=kt_pca(X) %% Linear Kernel PCA function
% 
%
% Note that the methodology behind this is described in the manuscript:
% "Canonical Response Parameterization: Quantifying the structure of 
%  responses to single-pulse intracranial electrical brain stimulation"
% by Kai J. Miller, et. al., 2022
%
% Copyright (C) 2022 Kai J Miller
% Revised by KJM March 2023 at time of manuscript re-submission 
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
%% Initial housekeeping
    srate=1/mean(diff(t_win)); % get sampling rate from 

%% Calculate sets of normalized single stimulation cross-projection magnitudes
    %
    t_step=5; % timestep between timepoints (in samples - saves memory, but peakfinding problems if too high and result is not smooth enough)
    proj_tpts=10:t_step:size(V,1); % timepoints for calculation of profile of (units samples)
    m=[]; % mean projection magnitudes
    v2=[]; % variance of projection  magnitudes
    %
    for k=proj_tpts % parse through time and perform projections for different data lengths
        %
        S=ccep_proj(V(1:k,:)); % get projection magnitudes for this duration
        S=S/sqrt(srate); % change units from \uV*sqrt(samples) to sqrt(seconds)
        %
        % calculate mean and variance of projections for this duration
        m=[m mean(S)]; v2=[v2 var(S)];        
        % 
        S_all(:,length(m))=S; % store projection weights 
    end
    [~,tt]=max(m); % tt is the sample corresponding to response duration

%% Parameterize trials
    %
    V_tR=V(1:proj_tpts(tt),:); % Reduced length voltage matrix (to response duration)
    [E_tR,~]=kt_pca(V_tR); % Linear kernel trick PCA method to capture structure
    C=E_tR(:,1); % 1st PC, canonical shape, C(t) from paper
%     C=mean(V_tR,2); C=C/norm(C); disp('NOTE - USING RAW AVG INSTEAD OF 1st PC') % mean shape
    
    %
    al=C.'*V_tR; % alpha coefficient weights for C into V
    ep=V_tR-C*al; % residual epsilon (error timeseries) after removal of form of CCEP        
    
%% Package data out 

    % % projections data % %
    crp_projs.proj_tpts=t_win(proj_tpts); % time points that projections were calculated at
    crp_projs.S_all=S_all; % full set of projection magnitudes at all durations
    crp_projs.mean_proj_profile=m; % mean projection magnitude profile
    crp_projs.var_proj_profile=v2; % variance in projection magnitude profile
    %
    crp_projs.tR_index=tt; % index into proj_tpts (projection magnitude times) corresponding to response duration
    %
    crp_projs.avg_trace_input=mean(V,2); % simple average trace, over full time interval input    
    
    % significance statistics - note that have to send in only non-overlapping trials. each trial is represented half of the time as the normalized projected, and half as un-normalized projected-into
    stat_indices=get_stat_indices(size(V,2));
    crp_projs.stat_indices=stat_indices;
    
    % t-statistic at response duration \tau_R
    crp_projs.t_value_tR=mean(S_all(stat_indices,tt))/(std(S_all(stat_indices,tt))/sqrt(length(S_all(stat_indices,tt)))); % calculate t-statistic

    % p-value at response duration \tau_R (extraction significance)
    [~,crp_projs.p_value_tR]=ttest(S_all(stat_indices,tt),0,'tail','right'); % simple t-test
    
    % t-statistic at full time sent in
    crp_projs.t_value_full=mean(S_all(stat_indices,end))/(std(S_all(stat_indices,end))/sqrt(length(S_all(stat_indices,end)))); % calculate t-statistic

    % p-value at  full time sent in (extraction significance)
    [~,crp_projs.p_value_full]=ttest(S_all(stat_indices,end),0,'tail','right'); % simple t-test

        
    % % parameterizations % %
    crp_parms.V_tR=V_tR; % Reduced length voltage matrix (to response duration)
    crp_parms.al=al; % alpha coefficient weights for C into V
    crp_parms.C=C; % 1st PC, canonical shape, C(t) from paper
    crp_parms.ep=ep; % residual epsilon after removal of form of CCEP  
    crp_parms.tR=t_win(proj_tpts(tt)); % response duration in units seconds 
    crp_parms.parms_times=t_win(1:proj_tpts(tt)); % times for parameterized data
    % 
    crp_parms.avg_trace_tR=mean(V_tR,2); % simple average trace, truncated to response duration
    % 
    % extracted single-trial quantities (e.g. Table 1 in manuscript)
    crp_parms.al_p=al/(length(C).^.5); % alpha-prime: alpha scaled to sqrt(number of samples),
    crp_parms.epep_root=sqrt(diag(ep.'*ep)).'; % scalar summarizing residual for each trial
    crp_parms.Vsnr=al./sqrt(diag(ep.'*ep)).'; % "signal to noise" for each trial
    crp_parms.expl_var=1-((diag(ep.'*ep)).')./((diag(V_tR.'*V_tR)).'); % explained variance by parameterization for each trial
  
% % % % % % end of function "CRP_method" % % % % % % 
    
%% Projections function
function S0=ccep_proj(V)
% function S0=ccep_proj(V)
% This function performs the single-trial cross-projections
% kjm, 2022
    
    V0=V./(ones(size(V,1),1)*(sum(V.^2,1).^.5)); % normalize (L2 norm) each trial
    V0(isnan(V0))=0; % this takes care of divide-by-zero situation in normalization -- should only happen if experimenter has done this artifiically somehow, but this takes care of it. 
    P=V0.'*V; % calculate internal projections (semi-normalized - optimal)
%     P=V0.'*V0; % calculate internal projections cross-correlation (fully normalized - doesn't work well - overly favors early transients b/c normalization heavily degrades )
%     P=V.'*V; % calculate internal projections (un-normalized - doesn't work well - too much emphasis on high-amplitude trials and cuts off meaningful structure)
    %
    % get only off-diagonal elements of P (i.e. ignore self-projections)
    p0=P; 
    p0(1:size(P,1)+1:end)=NaN;
    S0=reshape(p0,1,[]); % reshaping projections into single set
    S0(isnan(S0))=[]; % removing diagonal elements (self-projections)

%% Linear Kernel PCA function
function [E,S]=kt_pca(X)
% function [E,S]=kt_pca(X)
% This is an implementation of the linear kernel PCA method ("kernel trick")
% described in "Kernel PCA Pattern Reconstruction via Approximate Pre-Images"
% by Schölkopf et al, ICANN, 1998, pp 147-15
% See also course lectures by K.R. Muller (TU Berlin)
% 
% Inputs:
% X(T,N) - Matrix of data in. Only need this trick if T>>N
% 
% Outputs:
% E(T,N) - Columns of E are estimated Eigenvectors of X, in descending order
% S(1,N) - Corresponding estimated Eigenvalues of X, in descending order
% 
% kjm, 6/2020

%% use the "kernel trick" to estimate eigenvectors of this cluster of pair groups
    [F,S2]=eig(X.'*X); % eigenvector decomposition of (covariance of transpose) - may want to replace this with cov function so mean is subtracted off
    [S2,v_inds]=sort(sort(sum(S2)),'descend'); F=F(:,v_inds); %reshape properly    
    %
    S=S2.^.5; % estimated eigenvalues of both X.'*X and X*X.'
    %
    %     ES=X*F.';
    ES=X*F; % kernel trick
    E=ES./(ones(size(X,1),1)*S); % divide through to obtain unit-normalized eigenvectors
    
    
%% Function to Pick out indices of S to use for statistical comparison (so that projections between the same two trials are only compared once)
function stat_indices=get_stat_indices(N)
% This function picks out the indices of S that can be used for statistical comparison. 
% For each trial, half of normalized projections to other trials are used and the other half trials are the projected into
% ones. No overlapping comparison pairs are used. 
%
% kjm 3/2023
%
% Input: 
% N, scalar - number of trials
%
% Output:
% stat_indices (N^2-N,1) - vector of indices to be used for statistical comparison
%
% kjm, 3,2023

stat_indices=1:2:(N^2-N); % indices used for statistics

if rem(N,2)==1 %odd number of trials - need to offset every other column in original P matrix
    b=0*stat_indices; %initializes what is indexed
    for k=1:N
        if mod(k,2)==0 % offset what would have been every even column in original matrix
           b(((k-1)*((N-1)/2)+1):((k)*((N-1)/2)))=1; 
        end
    end
    %
    stat_indices=stat_indices+b;    
end
    