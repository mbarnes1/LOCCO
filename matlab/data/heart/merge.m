variablenames = {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'};
% Descriptions:
% age: age in years
% sex: sex (1 = male; 0 = female)
% cp: chest pain type
%        -- Value 1: typical angina
%        -- Value 2: atypical angina
%        -- Value 3: non-anginal pain
%        -- Value 4: asymptomatic
% trestbps: resting blood pressure (in mm Hg on admission to the hospital)
% chol: serum cholestoral in mg/dl
% fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
% restecg: resting electrocardiographic results
%        -- Value 0: normal
%        -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST 
%                    elevation or depression of > 0.05 mV)
%        -- Value 2: showing probable or definite left ventricular hypertrophy
%                    by Estes' criteria
% thalach: maximum heart rate achieved
% exang: exercise induced angina (1 = yes; 0 = no)
% oldpeak = ST depression induced by exercise relative to rest
% slope: the slope of the peak exercise ST segment
%        -- Value 1: upsloping
%        -- Value 2: flat
%        -- Value 3: downsloping
% ca: number of major vessels (0-3) colored by flourosopy
% thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
% num: diagnosis of heart disease (angiographic disease status)
%        -- Value 0: < 50% diameter narrowing
%        -- Value 1: > 50% diameter narrowing

datasets = {'cleveland', 'hungarian', 'switzerland', 'va'};
T = table;
for iData = 1:length(datasets)
    dataset = datasets{iData};
    iT = readtable(['processed.', dataset, '.csv'], 'Delimiter',',', 'TreatAsEmpty', '?');
    iT.Properties.VariableNames = variablenames;
    country = cell2table(repmat({dataset},height(iT),1));
    country.Properties.VariableNames = {'country'};
    iT = horzcat(country, iT);
    T = vertcat(T, iT);
end

% Remove NaN rows/columns or map to categorical
T.sex = categorical(T.sex);
T.sex(T.sex == '0') = 'female';
T.sex(T.sex == '1') = 'male';
T.sex(ismissing(T.sex)) = 'missing';

T.cp = categorical(T.cp);
T.cp(T.cp == '1') = 'typical';
T.cp(T.cp == '2') = 'atypical';
T.cp(T.cp == '3') = 'non';
T.cp(T.cp == '4') = 'asymptomatic';
T.cp(ismissing(T.cp)) = 'missing';

T.restecg = categorical(T.restecg);
T.restecg(T.restecg == '0') = 'normal';
T.restecg(T.restecg == '1') = 'abnormal';
T.restecg(T.restecg == '2') = 'probable';
T.restecg(ismissing(T.restecg)) = 'missing';

T.thal = categorical(T.thal);
T.thal(T.thal == '3') = 'normal';
T.thal(T.thal == '6') = 'fixed';
T.thal(T.thal == '7') = 'reversible';
T.thal(ismissing(T.thal)) = 'missing';

T.slope = categorical(T.slope);
T.slope(T.slope == '1') = 'up';
T.slope(T.slope == '2') = 'flat';
T.slope(T.slope == '3') = 'down';
T.slope(ismissing(T.slope)) = 'missing';

T.ca = [];  % too many nan values

T.num(T.num >= 1) = 1;
T.num = categorical(T.num);
T.num(T.num == '0') = 'neg';
T.num(T.num == '1') = 'pos';

T = rmmissing(T);  % remove missing rows
writetable(T, 'heart_all.csv')