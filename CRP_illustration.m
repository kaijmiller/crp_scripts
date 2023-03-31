function [crp_projs,crp_parms,bad_trials]=CRP_illustration(fname)
%
% This function reproduces all of the figures, etc, from the manuscript
% "Canonical Response Parameterization: Quantifying the structure of 
%  responses to single-pulse intracranial electrical brain stimulation"
% by Kai J. Miller, et. al., 2022
%
% This function should be explored alongside this manuscript for illustrative learning
%
% The most important purpose of this illustration is to show how the 
% function "CRP_method" is called and how data sent to it are packaged.
%
% Please note that options for desired parameters must be COMMENTED in and out within the function:
%  - starting and ending times for analysis
%  - whether to save plotted figures
%  - whether to reject outlier / artifactual trials, and at what duration to do so
%
% Please note that what sample file will be analyzed / plotted to must be COMMENTED IN
%
% If you send in your own file (in the function call), the string corresponding
% to the filename must also designate the path to it.
%
% Alternate user-input files must contain input data to this illustrative function must contain the variables:
% "data": single-trial voltage matrix (dimensions time x trials)
% "t": temporal indices with respect to stimulus time elements (dimensions 1 x time)
% note that the dimensions of "data" should be larger or the same size as the matrix "V" that is sent to CRP_method
% note that the dimensions of "t" should be larger or the same size as the vector "t_win" that is sent to CRP_method
%
% Sub-functions contained in this function (at end of script):
%
% [m,p]=trial_tests(crp_projs,art_rem,num_trials) %% This function identifies outliers in individual trials 
% kjm_printfig(fname,ppsize) %% This function exports the current figure in a reasonable way, in both eps and png formats 
%
% kjm 7/2022

%% 1 - Set options for illustration 

    % post-stimulation starting time (in seconds)
    t_1=.015;
    
    % post-stimulation end time (in seconds)
    t_2=1;
   

    % COMMENT ONE OPTION IN AND ONE OUT: save figures made?
%         figsave='y';
        figsave='n';

    % COMMENT ONE OPTION IN AND ONE OUT: do removal of artifactual / poorly fit outlier trials?
        art_rem.do='y';
%         art_rem.do='n';

    % COMMENT ONE OPTION IN AND ONE OUT: if doing artifact removal, test for significance at response duration or at full datalength sent?
%     art_rem.interval='tR'; % testing for projection magnitudes at response duration
    art_rem.interval='full'; % testing for projection magnitudes at full datalength sent

        
        

%% 2 - Load data
%  comment lines in and out to run different examples from each figure
if exist('fname')==1 % true if a filename is sent in
    load(['sampledata' filesep fname '.mat'])
else
    %
%     fname='Fig1_sampledata'; % figure 1 data -- random example data to show V matrix
    %
%     fname='Fig2_sampledata'; % figure 2 data --  large response with long TR for projection illustration
    %
%     fname='Fig3_sampledata'; % figure 3 data -- short TR for duration illustration
    %
%     fname='Fig4_sampledata';
    %
%     fname='Fig5_sampledata_1'; % huge effect
%     fname='Fig5_sampledata_2';
%     fname='Fig5_sampledata_3';
%     fname='Fig5_sampledata_4'; % interesting... artifact?
%     fname='Fig5_sampledata_5'; % this is an insignificant CCEP
    %
%     fname='Fig6_a_significant_example'; % simple example of significant structure in extraction
%     fname='Fig6_b_insignificant_example'; % illustration of an insignificant response
%     fname='Fig6_c_earlycomponent'; % dwindle in artifact giving an early significant component ... can send t1 to -.015 to illustrate
%     fname='Fig6_e_20muV_offset'; % false significance resulting from 20\muV baselining error  (can see odd with referencing schema)
%     fname='Fig6_f_offset_corrected'; % as in e, but with baseline corrected, and significance is lost
    %
%     fname='Fig7_artificial_box';
%     fname='Fig7_artificial_box_short';
%     fname='Fig7_artificial_box_split';
%     fname='Fig7_artificial_noise_0p5';
%     fname='Fig7_artificial_noise_2p5';
%     fname='Fig7_artificial_noise_7p5';
%     fname='Fig7_artificial_ramp';
%     fname='Fig7_artificial_ramp_descend';
%     fname='Fig7_artificial_ramp_dual';
%     fname='Fig7_artificial_sine';
%     fname='Fig7_artificial_sine_abs';
%     fname='Fig7_artificial_sine_inverted';
    %
%     fname='Fig8_sampledata';
    %
    fname='Fig9_eeg_data_zeropad'; % EEG data for artifact removal example. Note that it is zero-padded outside of range used.

%     %
%     fname='ramp_dt_50';

% fname='FigX_rightAmygdalaWithSpikes_stim_at_orbitofrontal_WM_0_1';

% fname='FigX_stimRO5-RO6_measRV5_repeats';
% fname='FigX_stimRO5-RO6_measRV5_repeats_1st_10';

% fname='FigX_w_traces_const_3x';
% fname='FigX_b_traces_const_3x';
% fname='FigX_w_traces_var_3x_1x';
% fname='FigX_b_traces_var_3x_1x';

 %
%     fname='ramp_dt_50_200';
    %
    load(['sampledata' filesep fname '.mat'])
    %
end

% % % 

    disp('- - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    disp(['input file: ' fname])

%% 3 - Convert data from original format into format required for CRP function
    % find timepoints 
    if t(1)>=t_1
        error('bad_time_lims 1','error: the first time in the input time vector is greater than the desired starting time')
    elseif t(end)<t_2
        error('bad_time_lims 1','error: the last time in the input time vector is less than the desired ending time')        
    else
        tpts=find(and(t>t_1,t<=t_2));
    end
    %
    V=data(tpts,:); 
    t_win=t(tpts); 
    clear tpts
    %
    srate=1/mean(diff(t));
    
%% 4 - Identify and remove potentially artifactual trials -- can remove and re-run if so
 % Note: one could do this recursively if desired to remove multiple layers of badness, though I do not here

    if art_rem.do=='y'
        disp('- - examining for bad trials - -')            
        [~, crp_projs]=CRP_method(V,t_win); % run CRP method function to get projections
        [m,p]=trial_tests(crp_projs,art_rem,size(V,2)); % calculate means and stats for each projections of each trial
        %       
        % conditions to be potentially artifactual
        % 1 -- p<threshold (p<0.000001 seems reasonable in many cases... choose your own criteria)
%             a=p<0.000001;
%             a=p<0.001;
            a=p<10^(-5);
        % 2 -- represents a decrease in projection magnitude
            a=find(a.*(m<mean(m)));
        %
        if sum(a)>0 % identify if any trials are artifactual
            for k=1:length(a)
                disp(['Trial ' num2str(a(k)) ' appears artifactual, p=' num2str(p(a(k)))])
            end
        disp(['Automatically rejecting bad trials:' num2str(a)])  
        V(:,a)=[];
        %
        else 
            disp('No trials appear to be artifactual')
        %
        end
        %
        bad_trials=a;
        clear a m p
        %
    elseif art_rem.do=='n'
        bad_trials=NaN;        

    end 

%% 5 - Call CRP function
    [crp_parms, crp_projs]=CRP_method(V,t_win);
    
    
%% 6 - Calculate an estimate of uncertainty. In the example way used for the manuscript, 
%      this estimate of uncertainty in \tau_R is chosen for when projection magnitude 
%      exceeds 98% of maximum.
    %
    % use threshold crossing times for current purposes
    m=crp_projs.mean_proj_profile; % trace of averge projection magnitude
    thresh_pct=.98; % percent threshold to consider    
    [mp,c]=max(m);
    %
    %
    % find lower bound on duration
    b=1+find((m(2:end)>(thresh_pct*mp)).*(m(1:(end-1))<=(thresh_pct*mp)));
    b(b>c(1))=[];
    lb=b(end);
    %
    %
    % find higher bound on duration
    b=find((m(1:(end-1))>(thresh_pct*mp)).*(m(2:end)<=(thresh_pct*mp)));
    b(b<c(1))=[];
    %
    if isempty(b), hb=length(m); % in case it goes to end,
    else, hb=b(1);
    end
    %
    err_low_bound_time=crp_projs.proj_tpts(lb);    
    err_high_bound_time=crp_projs.proj_tpts(hb);
    %
    clear m mp b c hb lb
    
    
%% 7 - Plot traces & time-resolved projection magnitudes
figure('Name','Data traces and time-resolved projection magnitudes','NumberTitle','off')
    
    subplot(1,2,1)
    plot(t_win,V,'color',.5*[1 1 1])
    hold on, plot([t_win(1) t_win(end)],[0 0],'color',.3*[1 1 1])
    hold on, plot(crp_parms.parms_times,crp_parms.avg_trace_tR,'color',[1 1 0],'LineWidth',4) % 1st PC, scaling to RMS of classic CCEP   
    hold on, plot(t_win,mean(crp_projs.avg_trace_input,2),'k','LineWidth',2)
    title(['t-value=' num2str(crp_projs.t_value_tR) ', p-value=' num2str(crp_projs.p_value_tR)])
    ylabel('Voltage (\muV)')
    box off
    %
    subplot(1,2,2)
    hold on, plot(crp_projs.proj_tpts,crp_projs.S_all,'.','color',.3*[1 1 1])
%     hold on, plot(max(max(crp_projs.S_all)),'.','color',.3*[1 1 1])
%     hold on, plot(min(min(crp_projs.S_all)),'.','color',.3*[1 1 1])
    hold on, plot([0 t_win(end)],[0 0],'-','color',.6*[1 1 1],'LineWidth',2)    
    % this next one is to plot just the first cluster of projections over time as in Figure 3C (but for 1st cluster)
%    hold on, plot(crp_projs.proj_tpts,crp_projs.S_all(1:(size(V,2)-1),:),'.','color',[28 164 201]/256,'MarkerSize',8) 
    hold on, plot(crp_projs.proj_tpts,crp_projs.mean_proj_profile,'k','LineWidth',2)
    hold on, plot(crp_parms.tR,max(crp_projs.mean_proj_profile),'ro','LineWidth',2)
    a=diff(get(gca,'ylim'))*.03;
    hold on, plot(err_low_bound_time*[1 1],max(crp_projs.mean_proj_profile)+a*[-1 1],'r-','LineWidth',2)
    hold on, plot(err_high_bound_time*[1 1],max(crp_projs.mean_proj_profile)+a*[-1 1],'r-','LineWidth',2)    
    box off, clear a
    xlabel('Time (s)')
    ylabel('Proj Mag (\muV s^{1/2})')
    title(['mean projection (S) ' num2str(max(crp_projs.mean_proj_profile))])
    %
    if figsave=='y'
        kjm_printfig(['figures' filesep fname '_traces_projections'],[16 5])
    end
    
%% 8 - Plot projection magnitudes of full epoch input and at response duration (\tau_R)
figure('Name','Cross-projection magnitudes at response duration and for full input','NumberTitle','off')
    
    subplot(1,2,1)
    plot(crp_projs.S_all(:,end),'.','MarkerSize',15)
    ylabel('Proj Mag (\muV s^{1/2})'),xlabel('sample number'), box off
    title('Projection magnitude of full interval input')
    
    subplot(1,2,2)   
    plot(crp_projs.S_all(:,find(crp_projs.proj_tpts==crp_parms.tR)),'.','MarkerSize',15)
    ylabel('Proj Mag (\muV s^{1/2})'),xlabel('sample number'), box off
    title('Projection magnitude at response duration \tau_R')
    
    if figsave=='y'
        kjm_printfig(['figures' filesep fname '_projmags_tR_and_input'],[10  6])
    end    
    
    
%% 9 - Plot single-trial traces, parameterization, and residuals
figure('Name','Single-trial parameterizations','NumberTitle','off')
    
    for k=1:size(crp_parms.V_tR,2)
        subplot(size(V,2),1,k), 
        plot([0 crp_parms.parms_times(end)],[0 0],'-','color',.5*[1 1 1])
        hold on, plot(crp_parms.parms_times, crp_parms.V_tR(:,k),'k','LineWidth',2),
        hold on, plot(crp_parms.parms_times, crp_parms.al(k)*crp_parms.C,'r','LineWidth',1),
        hold on, plot(crp_parms.parms_times, crp_parms.ep(:,k),'g','LineWidth',1),
        box off
        set(gca,'YLim',[min(min(crp_parms.V_tR)) max(max(crp_parms.V_tR))])
        xticklabels([])
    end
    ylabel('Voltage (\muV)'), xlabel('Time (s)')
    xticklabels('auto')
    clear ha k
    
    if figsave=='y'
        kjm_printfig(['figures' filesep fname '_single_trial_parmsfig'],[7 25])
    end    

%% display parameters
    disp('- - parameterizations - -')
    disp(['mean projection (S) at response duration: ' num2str(max(crp_projs.mean_proj_profile))])
    disp(['response duration, \tau_R: ' num2str(crp_parms.tR)])
    disp(['mean alpha: ' num2str(mean(crp_parms.al))])
    disp(['mean alpha-prime: ' num2str(mean(crp_parms.al)/(length(crp_parms.C).^.5))])
    disp(['mean epep_root: ' num2str(mean(crp_parms.epep_root))])
    disp(['mean SNR: ' num2str(mean(crp_parms.Vsnr))])
    disp(['mean explained variance: ' num2str(mean(crp_parms.expl_var))])
    disp(['Extraction significance t=' num2str(crp_projs.t_value_tR) ', p=' num2str(crp_projs.p_value_tR)])
    [~,tmp_p]=ttest(crp_parms.al);
    disp(['alpha coefficient significance p=' num2str(tmp_p)]), clear tmp_p
    disp('- - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    
% % % % % % END OF FUNCTION "crp_illustration" % % % % % %
    
%%
function [m,p]=trial_tests(crp_projs,art_rem,num_trials)
% [m,p]=trial_tests(crp_projs,art_rem,num_trials)
% this function identifies outliers in individual trials
% note that the third input, "num_trials" is the number of trials
% kjm, 7/2022

    % what interval to do testing of trials at?
    % test for projection distribution at response duration
    if strcmp(art_rem.interval,'tR')
        S_test=crp_projs.S_all(:,crp_projs.tR_index);
        disp('testing for bad trials at response duration tau_R')
    elseif strcmp(art_rem.interval,'full')
        S_test=crp_projs.S_all(:,end);
        disp('testing for bad trials at full time sent for analysis')
    end
    
    % initialize mean for each group and p-values for each group
    m=[]; p=[];
    
    % initialize figure
%     figure('Name','Sorted cross-projections of different trials','NumberTitle','off')
    
    % % % This part accounts for double counting
    stat_indices=1:2:(num_trials^2-num_trials); % indices used for statistics
    if rem(num_trials,2)==1 %odd number of trials - need to offset every other column in original P matrix
        b=0*stat_indices; %initializes what is indexed
        for k=1:num_trials
            if mod(k,2)==0 % offset what would have been every even column in original matrix
               b(((k-1)*((num_trials-1)/2)+1):((k)*((num_trials-1)/2)))=1; 
            end
        end
        %
        stat_indices=stat_indices+b;    
    end
    excl_indices=setdiff([1:length(S_test)], stat_indices);
    % % % End of part for double counting

    for q=1:num_trials
        %
        if q>1
            t_indices1=(q-1):(num_trials-1):(q*(num_trials-1)); % projections of normalized other trials into this trial before this cluster is itself projected 
        else
            t_indices1=[];
        end
        t_indices2=(q*(num_trials)):(num_trials-1):length(S_test); % projections of normalized other trials into this trial after this cluster is itself projected
        t_indices3=(q-1)*(num_trials-1)+[1:(num_trials-1)]; % projections of this trial into other trials
        %
        g_in=unique([t_indices1 t_indices2 t_indices3]); % projections involving this trial
        g_out=setdiff([1:length(S_test)],[g_in excl_indices]); % all other projections
        %
        % get mean of this clusters projection for this trial normalized 
        [~,p(q)]=ttest2(S_test(g_in),S_test(g_out)); % unpaired t-test // note, this includes some c
        m(q)=mean(S_test(t_indices3)); % mean projection for this trial normalized 
        %
        % this plots the distributions for this cluster
%         subplot(num_trials,1,q)
%         plot(S_test,'k.','MarkerSize',15)
%         hold on, plot(t_indices1,S_test(t_indices1),'r.','MarkerSize',15)
%         hold on, plot(t_indices2,S_test(t_indices2),'r.','MarkerSize',15)
%         hold on, plot(t_indices3,S_test(t_indices3),'g.','MarkerSize',15)
%         title(['trial ' num2str(q) ', m=' num2str(m(q)) ', p=' num2str(p(q))]), box off
        %    
        clear t_indices* g_in g_out

    end


%%
function kjm_printfig(fname,ppsize)
% This function exports the current figure in a reasonable way, in both eps
% and png formats 
%
% "fname" - desired filename. include path if desired
% "size" - size of figure in cm
% kjm 05/10

set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperSize', [ppsize]);
set(gcf, 'PaperPosition',[0 0 2*ppsize])

print(gcf,fname,'-depsc2','-r300','-painters')
print(gcf,fname,'-dpng','-r300','-painters')

