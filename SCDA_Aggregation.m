% The codes are written by Xiu-Shen Wei (weixs.gm@gmail.com). For any problem concerning the code, please feel free to contact Mr. Wei.
% This packages are free for academic usage. You can run them at your own risk. For other purposes, please contact Prof. Jianxin Wu (wujx2001@gmail.com).

% The codes are corresponding to the deep descriptors *aggregation* procedure
% of the proposed SCDA method.

% Load the extracted SCDA features
load('SCDA_flip_plus.mat');

%% Global-avg-pool
train_data_L31a = [];
train_data_L31b = [];
for i = 1 : size(cnnFeat_tr_L31,1)
    if isempty(cnnFeat_tr_L31{i,1})
        train_data_L31a(i,:) = 0;        
    else
        train_data_L31a(i,:) = mean(squeeze(cnnFeat_tr_L31{i,1}));
        train_data_L31a(i,:) = train_data_L31a(i,:) ./ norm(train_data_L31a(i,:));        
    end
    if isempty(cnnFeat_tr_L31{i,2})
        train_data_L31b(i,:) = 0;
    else
        train_data_L31b(i,:) = mean(squeeze(cnnFeat_tr_L31{i,2}));
        train_data_L31b(i,:) = train_data_L31b(i,:) ./ norm(train_data_L31b(i,:));
    end
end
test_data_L31a = [];
test_data_L31b = [];
for i = 1 : size(cnnFeat_te_L31,1)
    if isempty(cnnFeat_te_L31{i,1})
        test_data_L31a(i,:) = 0;        
    else
        test_data_L31a(i,:) = mean(squeeze(cnnFeat_te_L31{i,1}));
        test_data_L31a(i,:) = test_data_L31a(i,:) ./ norm(test_data_L31a(i,:));        
    end
    if isempty(cnnFeat_te_L31{i,2})       
        test_data_L31b(i,:) = 0;
    else       
        test_data_L31b(i,:) = mean(squeeze(cnnFeat_te_L31{i,2}));
        test_data_L31b(i,:) = test_data_L31b(i,:) ./ norm(test_data_L31b(i,:));
    end
end
train_data_L28a = [];
train_data_L28b = [];
for i = 1 : size(cnnFeat_tr_L28,1)
    if isempty(cnnFeat_tr_L28{i,1})
        train_data_L28a(i,:) = 0;
    else
        train_data_L28a(i,:) = mean(squeeze(cnnFeat_tr_L28{i,1}));
        train_data_L28a(i,:) = train_data_L28a(i,:) ./ norm(train_data_L28a(i,:));
    end
    if isempty(cnnFeat_tr_L28{i,2})
        train_data_L28b(i,:) = 0;
    else
        train_data_L28b(i,:) = mean(squeeze(cnnFeat_tr_L28{i,2}));
        train_data_L28b(i,:) = train_data_L28b(i,:) ./ norm(train_data_L28b(i,:));
    end
end
test_data_L28a = [];
test_data_L28b = [];
for i = 1 : size(cnnFeat_te_L28,1)
    if isempty(cnnFeat_te_L28{i,1})
        test_data_L28a(i,:) = 0;
    else
        test_data_L28a(i,:) = mean(squeeze(cnnFeat_te_L28{i,1}));
        test_data_L28a(i,:) = test_data_L28a(i,:) ./ norm(test_data_L28a(i,:));
    end
    if isempty(cnnFeat_te_L28{i,2})
        test_data_L28b(i,:) = 0;
    else
        test_data_L28b(i,:) = mean(squeeze(cnnFeat_te_L28{i,2}));
        test_data_L28b(i,:) = test_data_L28b(i,:) ./ norm(test_data_L28b(i,:));
    end
end

save('SCDA_avgPool.mat','train_data_L31a','test_data_L31a','train_data_L28a','test_data_L28a'...
    ,'train_data_L31b','test_data_L31b','train_data_L28b','test_data_L28b'...
    ,'train_label','test_label','-v7.3');

disp('SCDA avgPool is done ...');

%% Global-max-pool
train_data_L31a = [];
train_data_L31b = [];
for i = 1 : size(cnnFeat_tr_L31,1)
    if isempty(cnnFeat_tr_L31{i,1})
        train_data_L31a(i,:) = 0;
    else
        train_data_L31a(i,:) = max(squeeze(cnnFeat_tr_L31{i,1}));
        train_data_L31a(i,:) = train_data_L31a(i,:) ./ norm(train_data_L31a(i,:));
    end
    if isempty(cnnFeat_tr_L31{i,2})
        train_data_L31b(i,:) = 0;
    else
        train_data_L31b(i,:) = max(squeeze(cnnFeat_tr_L31{i,2}));
        train_data_L31b(i,:) = train_data_L31b(i,:) ./ norm(train_data_L31b(i,:));
    end
end
test_data_L31a = [];
test_data_L31b = [];
for i = 1 : size(cnnFeat_te_L31,1)
    if isempty(cnnFeat_te_L31{i,1})
        test_data_L31a(i,:) = 0;
    else
        test_data_L31a(i,:) = max(squeeze(cnnFeat_te_L31{i,1}));
        test_data_L31a(i,:) = test_data_L31a(i,:) ./ norm(test_data_L31a(i,:));
    end
    if isempty(cnnFeat_te_L31{i,2})
        test_data_L31b(i,:) = 0;
    else
        test_data_L31b(i,:) = max(squeeze(cnnFeat_te_L31{i,2}));
        test_data_L31b(i,:) = test_data_L31b(i,:) ./ norm(test_data_L31b(i,:));
    end
end
train_data_L28a = [];
train_data_L28b = [];
for i = 1 : size(cnnFeat_tr_L28,1)
    if isempty(cnnFeat_tr_L28{i,1})
        train_data_L28a(i,:) = 0;
    else
        train_data_L28a(i,:) = max(squeeze(cnnFeat_tr_L28{i,1}));
        train_data_L28a(i,:) = train_data_L28a(i,:) ./ norm(train_data_L28a(i,:));
    end
    if isempty(cnnFeat_tr_L28{i,2})
        train_data_L28b(i,:) = 0;
    else
        train_data_L28b(i,:) = max(squeeze(cnnFeat_tr_L28{i,2}));
        train_data_L28b(i,:) = train_data_L28b(i,:) ./ norm(train_data_L28b(i,:));
    end
end
test_data_L28a = [];
test_data_L28b = [];
for i = 1 : size(cnnFeat_te_L28,1)
    if isempty(cnnFeat_te_L28{i,1})
        test_data_L28a(i,:) = 0;
    else
        test_data_L28a(i,:) = max(squeeze(cnnFeat_te_L28{i,1}));
        test_data_L28a(i,:) = test_data_L28a(i,:) ./ norm(test_data_L28a(i,:));
    end
    if isempty(cnnFeat_te_L28{i,2})
        test_data_L28b(i,:) = 0;
    else
        test_data_L28b(i,:) = max(squeeze(cnnFeat_te_L28{i,2}));
        test_data_L28b(i,:) = test_data_L28b(i,:) ./ norm(test_data_L28b(i,:));
    end
end

save('SCDA_maxPool.mat','train_data_L31a','test_data_L31a','train_data_L28a','test_data_L28a'...
    ,'train_data_L31b','test_data_L31b','train_data_L28b','test_data_L28b'...
    ,'train_label','test_label','-v7.3');

disp('SCDA maxPool is done ...');

% Concatenation
avg = load('SCDA_avgPool.mat');
maxi = load('SCDA_maxPool.mat');
train_label = avg.train_label;
test_label = avg.test_label;

ratio = 0.5;
train_data = [avg.train_data_L31a maxi.train_data_L31a ratio.*avg.train_data_L28a ratio.*maxi.train_data_L28a ...
    avg.train_data_L31b maxi.train_data_L31b ratio.*avg.train_data_L28b ratio.*maxi.train_data_L28b];
test_data = [avg.test_data_L31a maxi.test_data_L31a ratio.*avg.test_data_L28a ratio.*maxi.test_data_L28a ...
    avg.test_data_L31b maxi.test_data_L31b ratio.*avg.test_data_L28b ratio.*maxi.test_data_L28b];
disp('The final SCDA_flip_plus feature is done ...');
