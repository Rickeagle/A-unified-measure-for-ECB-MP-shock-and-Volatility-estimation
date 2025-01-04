clc
clear
close all

%% LOAD DATA 
data = readtable('YC_merged_complete.xlsx');
% Main Vars
DataAnnouncement=rmmissing(data(1:200,9));
DR=data(:,[47:52]); 
DRs=data(:,[67:72]); 
DR=rmmissing(DR);  %Delta Returns
DRs=rmmissing(DRs);  %Delta returns star

%DeltaR
DR3mth = DR{:, 5};
DR1 = DR{:, 1};
DR2 = DR{:, 2};
DR5 = DR{:, 6};
DR10 = DR{:, 3};
DR30 = DR{:, 4};
%Delta star
DRs3mth = DRs{:, 5};
DRs1 = DRs{:, 1};
DRs2 = DRs{:, 2};
DRs2negative = -DRs{:, 2};  % Negative delta returns star
DRs5 = DRs{:, 6};
DRs5negative = -DRs{:, 6};  % Negative delta returns star
DRs10 = DRs{:, 3};
DRs30 = DRs{:, 4};


% Building IV matrices
Matrix5=[DR5; DRs5];
Matrix_IV5=[DR5 ; DRs5negative];% Instrument used for the regressor Matrix5
Matrix_IV2=[DR2 ; DRs2negative]; % Instrument used for the regressor Matrix2
IV5=[ones(400,1), Matrix_IV5];
IV2=[ones(400,1), Matrix_IV2];

Matrix2=[DR2;DRs2]; 
Matrix1=[DR1;DRs1];
Matrix10=[DR10;DRs10];
Matrix30=[DR30;DRs30];
Matrix3mth=[DR3mth;DRs3mth];

%% Heteroskedasticity-based IV regression
% First-stage regression: Regress Matrix5 on IV5
BetaFS = inv(IV5'*IV5)*IV5'*Matrix5;
Matrix5_hat =[IV5*BetaFS(2)];% fitted values
% First-stage regression: Regress Matrix2 on IV2
BetaFS2 = inv(IV2'*IV2)*IV2'*Matrix2;
Matrix2_hat =[IV2*BetaFS2(2)];

%Second stage for 5
Beta_2=inv(Matrix5_hat'*Matrix5_hat)*Matrix5_hat'*Matrix2;
BETA_2= Beta_2(2);

% outcome variable to monetary policy

%% Second step
T = size(DR2, 1); 
MP = zeros(T, 1);
for t = 1:T
    y_t = DR2(t, :)';  
    Betahat = [ones(size(BETA_2)), BETA_2]';
    coeffs=inv(Betahat'*Betahat)*Betahat'*DR2(t);
    MP(t) = coeffs(2);
end

% The resulting MP(t) is the estimated monetary policy shock series

%% ALL THE OTHER SERIES
% Matrix1
Beta_1 = inv(Matrix5_hat'*Matrix5_hat)*Matrix5_hat'*Matrix1;
BETA_1 = Beta_1(2);
% Second step for Matrix1
T = size(DR1, 1); 
MP1 = zeros(T, 1);
for t = 1:T
    y_t = Matrix1(t, :)';  
    Betahat1  = [ones(size(BETA_1)), BETA_1]';
    coeffs = inv(Betahat1'*Betahat1)*Betahat1'*DR1(t);
    MP1(t) = coeffs(2);
end

% Robustness using YC 2 year instead of 5
Beta_1_robust = inv(Matrix2_hat'*Matrix2_hat)*Matrix2_hat'*Matrix1;
BETA_1_robust= Beta_1_robust(2);
T = size(DR1, 1); 
MP1_robust= zeros(T, 1);
for t = 1:T
    y_t = Matrix1(t, :)';  
    Betahat1  = [ones(size(BETA_1_robust)), BETA_1_robust]';
    coeffs = inv(Betahat1'*Betahat1)*Betahat1'*DR1(t);
    MP1_robust(t) = coeffs(2);
end

% Matrix10
Beta_10 = inv(Matrix5_hat' * Matrix5_hat) * Matrix5_hat' * Matrix10;
BETA_10 = Beta_10(2);
%Second step for Matrix10
MP10 = zeros(T, 1);
for t = 1:T
    y_t = Matrix10(t, :)';  
    Betahat10 = [ones(size(BETA_10)), BETA_10]';
    coeffs = inv(Betahat10'*Betahat10)*Betahat10'*DR10(t);
    MP10(t) = coeffs(2);
end

% Matrix30
Beta_30 = inv(Matrix5_hat'*Matrix5_hat)*Matrix5_hat'*Matrix30;
BETA_30 = Beta_30(2);
% Second step for Matrix30
MP30 = zeros(T, 1);
for t = 1:T
    y_t = Matrix30(t, :)';  
    Betahat30 = [ones(size(BETA_30)), BETA_30]';
    coeffs = inv(Betahat30' * Betahat30) * Betahat30' * DR30(t);
    MP30(t) = coeffs(2);
end

% Matrix3mth
Beta_3mth = inv(Matrix5_hat' * Matrix5_hat) * Matrix5_hat' * Matrix3mth;
BETA_3mth = Beta_3mth(2);
%Second step for Matrix3mth
MP3mth = zeros(T, 1);
for t = 1:T
    y_t = Matrix3mth(t, :)';  
    Betahat3mth = [ones(size(BETA_3mth)), BETA_3mth]';
    coeffs = inv(Betahat3mth' * Betahat3mth) * Betahat3mth' * DR3mth(t);
    MP3mth(t) = coeffs(2);
end

%% save MP shocks in one file
% Combine into a table
dataTable = table(MP, MP1, MP10, MP30, MP3mth, ...
                  'VariableNames', {'MP', 'MP1', 'MP10', 'MP30', 'MP3mth'});

% Write to an Excel file
writetable(dataTable, 'MPshocks.xlsx');%% I added manually the meetong date
%% MONTHLY SHOCK
% Create the monthly series of MP shocks (MP, MP1, MP10, MP30, MP3mth)
data = readtable('MPshocks.xlsx');
MeetingDate = data.MeetingDate(1:200);
MP     = data.MP(1:200);
MP1    = data.MP1(1:200);
MP10   = data.MP10(1:200);
MP30   = data.MP30(1:200);
MP3mth = data.MP3mth(1:200);

% Convert MeetingDate to datetime 
if ~isdatetime(MeetingDate)
    MeetingDate = datetime(MeetingDate, 'InputFormat', 'dd-MM-yyyy'); 
end
%Create a table and extract YearMonth
T = table(MeetingDate, MP, MP1, MP10, MP30, MP3mth);
T.YearMonth = dateshift(T.MeetingDate, 'start', 'month');
%Group by YearMonth and sum shocks
[G, uniqueMonths] = findgroups(T.YearMonth);

MP_monthly     = splitapply(@sum, T.MP,     G);
MP1_monthly  = splitapply(@sum, T.MP1,    G);
MP10_monthly  = splitapply(@sum, T.MP10,   G);
MP30_monthly   = splitapply(@sum, T.MP30,   G);
MP3mth_monthly = splitapply(@sum, T.MP3mth, G);

% Collect into a grouped table
groupedTable = table(uniqueMonths, MP_monthly, MP1_monthly, MP10_monthly, MP30_monthly, MP3mth_monthly, ...
    'VariableNames', {'YearMonth','MP_month','MP1_month','MP10_month','MP30_month','MP3mth_month'});

%Build a complete list of months from the min to max 
allMonths = (dateshift(min(T.YearMonth), 'start', 'month') : calmonths(1) : ...
             dateshift(max(T.YearMonth), 'start', 'month')).';
allMonthsTable = table(allMonths, 'VariableNames', {'YearMonth'});

%Outer join so that missing months become rows of zeros 
% By doing an outer join on YearMonth, we'll keep every month
monthlyTable = outerjoin(allMonthsTable, groupedTable, ...
    'Keys','YearMonth', 'MergeKeys', true, 'Type','full');

% Fill NaNs with zeros for completeness of the code in case of replication.
% No NaNs were actually revealed in my computation.
varsToFill = {'MP_month','MP1_month','MP10_month','MP30_month','MP3mth_month'};
for i = 1:numel(varsToFill)
    col = varsToFill{i};
    monthlyTable.(col)(isnan(monthlyTable.(col))) = 0;
end

monthlyTable.Time = datestr(monthlyTable.YearMonth, 'mm/yyyy');
monthlyTable = movevars(monthlyTable, 'Time', 'Before', 'YearMonth');
% Save the result
writetable(monthlyTable, 'Monthly_MPshocks.xlsx');

disp('Monthly MP shock variables (including MP1, MP10, MP30, MP3mth) have been created and saved.');

%% Plot the MP series
% Figure
figure('Position', [100, 100, 1000, 550]); 
% Plot MP1
subplot(3, 2, 1);
plot(MP1, 'LineWidth', 1.5);
title('Monetary Policy Shock (MP1)', 'FontSize', 14);
xlabel('Time', 'FontSize', 12);
ylabel('MP1', 'FontSize', 12);
grid on;

% Plot MP2
subplot(3, 2, 2); 
plot(MP, 'LineWidth', 1.5); 
title('Monetary Policy Shock (MP2)', 'FontSize', 14);
xlabel('Time', 'FontSize', 12);
ylabel('MP2', 'FontSize', 12);
grid on;
% Plot MP10
subplot(3, 2, 3); 
plot(MP10, 'LineWidth', 1.5);
title('Monetary Policy Shock (MP10)', 'FontSize', 14);
xlabel('Time', 'FontSize', 12);
ylabel('MP10', 'FontSize', 12);
grid on;
% Plot MP30
subplot(3, 2, 4); 
plot(MP30, 'LineWidth', 1.5);
title('Monetary Policy Shock (MP30)', 'FontSize', 14);
xlabel('Time', 'FontSize', 12);
ylabel('MP30', 'FontSize', 12);
grid on;
% Plot MP3mth
subplot(3, 2, 5); 
plot(MP3mth, 'LineWidth', 1.5);
title('Monetary Policy Shock (MP3mth)', 'FontSize', 14);
xlabel('Time', 'FontSize', 12);
ylabel('MP3mth', 'FontSize', 12);
grid on;

sgtitle('Monetary Policy Shocks', 'FontSize', 18);
tight_layout = @(m, n) set(gcf, 'Renderer', 'painters', 'PaperPositionMode', 'auto');

%% %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%
%% Basic Time Series Properties of MP
figure;
autocorr(MP);
title('Sample ACF of MP Shocks');
% Plot the ACF of Squared MP
figure;
autocorr(MP.^2);
title('Sample ACF of Squared MP Shocks');

%% Let's do the MCLeonard and Li test to check for arch effects
% Let fit my series to the optimal ARIMA specification
maxP = 5; 
maxQ = 5; 
bestAIC = Inf; % Start with a very high AIC
bestP = 0; 
bestQ = 0; 
% Loop over possible combinations of p and q
for p = 0:maxP
    for q = 0:maxQ
        try
            Mdl = arima(p, 0, q);
            EstMdl = estimate(Mdl, MP - mean(MP), 'Display', 'off');           
            % Compute AIC
            [~,~,logL] = infer(EstMdl, MP - mean(MP)); 
            numParams = p + q + 1; % Number of parameters (AR + MA + constant)
            AIC = -2 * logL + 2 * numParams; 
            
            
            if AIC < bestAIC
                bestAIC = AIC;
                bestP = p;
                bestQ = q;
            end
        catch
            fprintf('Model ARMA(%d, %d) failed to estimate.\n', p, q);
        end
    end
end
fprintf('Optimal ARMA Model: AR(%d), MA(%d) with AIC = %.4f\n', bestP, bestQ, bestAIC);

% Fit the optimal ARMA model
OptimalMdl = arima(bestP, 0, bestQ);
OptimalEstMdl = estimate(OptimalMdl, MP - mean(MP));
residuals_ARMA = infer(OptimalEstMdl, MP - mean(MP));

squared_residuals_ARMA = residuals_ARMA.^2;
[h, pValue] = lbqtest(squared_residuals_ARMA, 'Lags', 10);

if h == 0
    fprintf('No significant ARCH effects detected (p-value = %.4f).\n', pValue);
else
    fprintf('Significant ARCH effects detected (p-value = %.4f).\n', pValue);
end

% Plot Squared Residuals Autocorrelation
figure;
autocorr(squared_residuals_ARMA);
title('Autocorrelation of Squared Residuals');
xlabel('Lag');
ylabel('Autocorrelation');
grid on;


%% AR Model for MP with EGARCH Volatility Specification
% AR(2) with EGARCH(2,1) volatility model
model = arima('ARLags', 2, 'Distribution', 'Gaussian', 'Variance', egarch(2, 3));

% Options for the estimation process
options = optimoptions(@fmincon, 'Display', 'off', 'Diagnostics', 'off', 'Algorithm', 'sqp', 'TolCon', 1e-7);
% Fit the model to MP
fit = estimate(model, MP, 'Options', options);
% Infer residuals and variances
[residuals_EGARCH, variances_EGARCH] = infer(fit, MP);
% Plot the residuals and conditional standard deviations
figure;
subplot(2, 1, 1);
plot(residuals_EGARCH, 'LineWidth', 1.5);
grid on;
xlabel('Time');
ylabel('Residual');
title('Filtered Residuals');

subplot(2, 1, 2);
plot(sqrt(variances_EGARCH), 'LineWidth', 1.5);
grid on;
xlabel('Time');
ylabel('Volatility');
title('Filtered Conditional Standard Deviations');
%% Standardize Residuals
standardizedResiduals_EGARCH = residuals_EGARCH ./ sqrt(variances_EGARCH);
figure;
autocorr(standardizedResiduals_EGARCH);
title('Sample ACF of Standardized Residuals');

% Plot ACF of squared standardized residuals
figure;
autocorr(standardizedResiduals_EGARCH.^2);
title('Sample ACF of Squared Standardized Residuals');

% Let's check autocorr in the residuals
[h_EGARCH, pValue_EGARCH] = lbqtest(standardizedResiduals_EGARCH, 'Lags', 10);
fprintf('Ljung-Box p-value (EGARCH-t standardized residuals): %.4f\n', pValue_EGARCH);
% We have that EGARCH (2,3) using a student t-distribution works fine

%%%%%%%%%%%%%%
%% GARCH-t
%  LAG SELECTION
% Variables to start the loop
lags = 1:8; 
bestAIC = Inf; % highest possible value as starting value
bestLag = []; 
AICs = NaN(length(lags), 1); 

% Initializing the loop to estimate diff GARCH specifications
for i = 1:length(lags)
    p = lags(i); 
    q = lags(i); 
    try
        Mdl = garch('GARCHLags', p, 'ARCHLags', q, 'Distribution', 't');
        EstMdl = estimate(Mdl, MP - mean(MP), 'Display', 'off');
        [~, logL] = infer(EstMdl, MP - mean(MP)); 
      
        numParams = numel(EstMdl.Constant) + numel(EstMdl.ARCH) + numel(EstMdl.GARCH) + 1; % +1 for DoF (t-dist)
        n = length(MP); 
        AIC = -2 * sum(logL) + 2 * numParams; % AIC formula
        AICs(i) = AIC;
        if AIC < bestAIC
            bestAIC = AIC;
            bestLag = [p, q];
        end
    catch ME
       % This is simply to guarantee the estimation are completed with no
       % errors
        fprintf('Estimation failed for GARCH(%d, %d): %s\n', p, q, ME.message);
    end
end
disp('AIC values for each model:');
disp(table(lags', AICs, 'VariableNames', {'Lags', 'AIC'}));

if ~isempty(bestLag)
    fprintf('Best model: GARCH(%d, %d) with AIC = %.4f\n', bestLag(1), bestLag(2), bestAIC);
else
    fprintf('No valid model was estimated.\n');
end
% The best Specification is the GARCH(1,1)

%% Estimate GARCH-t for MP

% Specify the GARCH(1,1)-t model
Mdl_GARCH11_t = garch('GARCHLags', 1, 'ARCHLags', 1, 'Distribution', 't');
options = optimoptions(@fmincon,Display="off",Diagnostics="off",Algorithm="sqp",TolCon=1e-7);

fit = estimate(model,MP-mean(MP),Options=options);  % Fit the model
[residuals_GARCH,VolatilityGARCH_t] = infer(fit,MP-mean(MP));

standardizedResiduals_GARCH= residuals_GARCH ./ sqrt(VolatilityGARCH_t);

% Optional: Plot residuals
figure(10);
subplot(2,1,1);
plot(residuals_GARCH, 'LineWidth', 1.5);
title('GARCH-t Residuals');
xlabel('Time');
ylabel('Residuals');
grid on;

subplot(2,1,2);
plot(standardizedResiduals_GARCH, 'LineWidth', 1.5);
title('GARCH-t Standardized Residuals');
xlabel('Time');
ylabel('Standardized Residuals');
grid on;
saveas(figure(10), 'GARCH residuals.png'); 

figure;
autocorr(standardizedResiduals_GARCH.^2);
title('Sample ACF of Squared Standardized Residuals');

% Does the model fit well?
[h_GARCH, pValue_GARCH] = lbqtest(standardizedResiduals_GARCH, 'Lags', 10);
fprintf('Ljung-Box p-value (GARCH-t standardized residuals): %.4f\n', pValue_GARCH);
% The model is fine!
%But let's compare it to a DCS model

%%
% Export VolatilityGARCH_t to add to a dataset
datasetWithVolatility = array2table([MP, VolatilityGARCH_t], ...
    'VariableNames', {'MP', 'GARCHVolatility'});

% Save the dataset as a .csv file (optional)
writetable(datasetWithVolatility, 'GARCH_Estimation.xlsx');
%%
%%%%%%%%%%%%%%%%%%%%%%
% Score and Garch-t comparison
X = ones(size(MP));
start_plot = 1; 
end_plot = size(MP,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
WhichVariable = MP(start_plot:end_plot,1);
WhichDates = [1:size(MP,1)]';

QANT_CUT = 8;
COLORE = 'k';
SPESSORE = 2;
%% Adjusted DCS-t Estimation Code
% Optimization options
optionsIVAN = optimset('Display', 'iter-detailed', ...
                       'LargeScale', 'off', ...
                       'MaxFunEvals', 5000, ...
                       'TolFun', 1e-6, ...
                       'TolX', 1e-6);
% Define bounds for parameters
lb = [-10; -10; -10; -10; min(MP)];  
ub = [10; 10; 10; 10; max(MP)];     

% Improved initial parameter guess
vpars_init = [randn(4, 1) * 0.1; mean(MP)];
% Loss function with error handling for complex outputs
lossT = @(vpar) ScoreFilterStudT_volatility_safe(vpar, MP, X);
% Run optimization using fmincon to apply bounds
[vpars, fval, exitflag, output] = fmincon(lossT, vpars_init, [], [], [], [], lb, ub, [], optionsIVAN);
[LL, Res, Save_Other] = ScoreFilterStudT_volatility_safe(vpars, MP, X);
VolatilityDCS_t = Res.vSig2(1:end-1, 1);

%% Safe ScoreFilterStudT_volatility Function
function [LL, Res, Save_Other] = ScoreFilterStudT_volatility_safe(vpars, MP, X)
    [LL, Res, Save_Other] = ScoreFilterStudT_volatility(vpars, MP, X);
    
    % Handle complex or invalid outputs
    if any(imag(LL) ~= 0) || isnan(LL)
        warning('Complex or NaN log-likelihood detected. Penalizing solution.');
        LL = Inf;  % Penalize invalid solutions
    end
end

%% Plotting Comparison
CUTT = 20;
WhichVariable = [VolatilityDCS_t(CUTT:end, 1), VolatilityGARCH_t(CUTT:end, 1)];
WhichDates = 1:length(WhichVariable);

figure(22);
plot1 = plot(WhichDates, WhichVariable, 'linewidth', 2);
set(plot1(1), 'Color', 'b', 'linestyle', '-');
set(plot1(2), 'Color', 'r', 'linestyle', '--');

xlim([min(WhichDates), max(WhichDates)]);
set(gca, 'Xtick', WhichDates);
title('COMPARISON VOLATILITIES');
legend('DCS-t', 'GARCH-t');
grid('on');
saveas(figure(22), 'ComparisonVolatilities.png'); 


% Figure Plot

figure(23);
StandardizedResid = Save_Other(:, 2) ./ sqrt(VolatilityDCS_t);
WeightsPlot = Save_Other(:, 1);
scatter(StandardizedResid, WeightsPlot);
xlabel('\epsilon / \sigma');
ylabel('weights');
xlim([-max(abs(StandardizedResid)) max(abs(StandardizedResid))]);

saveas(figure(23), 'Weights.residual_relationship.png'); 


%% PROBABILITY OF BEING AN OUTLIER

StandardizedResid = Save_Other(:, 2) ./ sqrt(VolatilityDCS_t); % spe_t
WeightsPlot = Save_Other(:, 1);                                 % w(spe_t)

% Compute the Weight for Prediction Error Equal to 0 (w(0))
w0 = WeightsPlot(StandardizedResid == 0);

% If there is no exact zero in StandardizedResid, interpolate w(0)
if isempty(w0)
    w0 = interp1(StandardizedResid, WeightsPlot, 0, 'linear', 'extrap');
end

% Compute the Probability of Observing Outliers
outlier_prob = 1 - abs(WeightsPlot / w0);

% Display Results
fprintf('Standardized Prediction Error (spe_t):\n');
disp(StandardizedResid);

fprintf('Weights (w(spe_t)):\n');
disp(WeightsPlot);

fprintf('Probability of Observing Outliers:\n');
disp(outlier_prob);

%% Plot the Standardized Residuals and Outlier Probabilities
figure(24);
subplot(2, 1, 1);
plot(StandardizedResid, 'b-', 'LineWidth', 1.5);
title('Standardized Prediction Errors ');
xlabel('Time');
ylabel('Standardized Residual');
grid on;
% Plot Outlier Probabilities
subplot(2, 1, 2);
stem(outlier_prob, 'r', 'filled', 'LineWidth', 1.5);
title('Probability of Observing Outliers');
xlabel('Time');
ylabel('Probability');
grid on;

sgtitle('Standardized Prediction Errors and Outlier Probabilities');
saveas(figure(24), 'Outlier probability.png'); 

figure;
histogram(outlier_prob)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

