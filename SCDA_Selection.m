% The codes are written by Xiu-Shen Wei (weixs.gm@gmail.com). For any problem concerning the code, please feel free to contact Mr. Wei.
% This packages are free for academic usage. You can run them at your own risk. For other purposes, please contact Prof. Jianxin Wu (wujx2001@gmail.com).

% The codes are corresponding to the deep descriptors *selection* procedure
% of the proposed SCDA method.

% Setting the GPU device
opt.gpu = 1;
g = gpuDevice(opt.gpu);
reset(g);

% Fine-grained datasets
opt.dataset = 'CUB200';
% CUB200
% Dogs
% Flowers
% Moth
% Pets
% Airplane
% Car

% The pre-trained model--VGG-16
opt.model = 'imagenet-vgg-verydeep-16';

% Our selective threshold
opt.thr = 'mean';

% load CNN model
run('../toolbox/matconvnet-1.0-beta23/matlab/vl_setupnn.m');
net = load(['../model/' opt.model '.mat']);
net.layers(end-5:end)=[]; % Removing the fully connected layers
net = vl_simplenn_move(net, 'gpu') ;
disp('CNN model is ready ...');

% load imdb
path = './feats/';
imdb = load([path opt.dataset '/imdb.mat']);

% Using the RGB average values obtained from ImageNet
net.normalization.averageImage = ones(224,224,3);
net.normalization.averageImage(:,:,1) = net.normalization.averageImage(:,:,1) .* net.meta.normalization.averageImage(1,1);
net.normalization.averageImage(:,:,2) = net.normalization.averageImage(:,:,2) .* net.meta.normalization.averageImage(1,2);
net.normalization.averageImage(:,:,3) = net.normalization.averageImage(:,:,3) .* net.meta.normalization.averageImage(1,3);
imdb.averageImage = net.normalization.averageImage;

num_tr = size(find(imdb.images.set==1),2);
num_te = size(find(imdb.images.set==3),2);
cnnFeat_tr_L31 = cell(num_tr,2);
cnnFeat_te_L31 = cell(num_te,2);
cnnFeat_tr_L28 = cell(num_tr,2);
cnnFeat_te_L28 = cell(num_te,2);
count_tr = 1;
count_te = 1;
ex_time = [];

for i = 1 : size(imdb.images.name,2)
    tic
    %% original image
    im = imread([imdb.imageDir '/' imdb.images.name{1,i} '.jpg']);
    im_ = single(im);
    [h,w,~] = size(im_);
    if min(h,w) > 700
        im_ = imresize(im_, [h*(700/min(h,w)) w*(700/min(h,w))]);
    end
    [h,w,c] = size(im_);
    if  c > 2
        im_ = im_ - imresize(imdb.averageImage,[h,w]) ;
    else    
        im_ = bsxfun(@minus,im_,imresize(imdb.averageImage,[h,w])) ;
    end
    
    res = vl_simplenn(net, gpuArray(im_)) ;
%     res = vl_simplenn(net, im_);
    tmp_1 = gather(res(32).x);
    tmp_2 = gather(res(29).x);
    
    % Pool5
    tmp_featmap = tmp_1;
    tmp_featmap_sum = sum(tmp_featmap, 3);
    tmp_mean = mean(mean(tmp_featmap_sum));
    highlight = zeros(size(tmp_featmap_sum));
    highlight(find(tmp_featmap_sum>tmp_mean)) = 1;
    
    cc = bwconncomp(highlight); % The biggest component
    numPixel = cellfun(@numel,cc.PixelIdxList);
    [~,conn_idx] = max(numPixel);
    highlight_conn_L31 = zeros(size(highlight));
    highlight_conn_L31(cc.PixelIdxList{conn_idx}) = 1;
    
    tmp_sel_feat_L31 = [];
    for sel_i = 1 : size(tmp_featmap,1)
        for sel_j = 1 : size(tmp_featmap,2)
            if highlight_conn_L31(sel_i,sel_j)
                tmp_sel_feat_L31 = [tmp_sel_feat_L31, tmp_featmap(sel_i,sel_j,:)];
            end
        end
    end
    
    % Relu5_2
    tmp_featmap = tmp_2;
    tmp_featmap_sum = sum(tmp_featmap, 3);
    tmp_mean = mean(mean(tmp_featmap_sum));
    highlight = zeros(size(tmp_featmap_sum));
    highlight(find(tmp_featmap_sum>tmp_mean)) = 1;
    highlight28 = highlight;
    highlight = highlight28 & imresize(highlight_conn_L31, size(highlight28), 'nearest');
    tmp_sel_feat_L28 = [];
    for sel_i = 1 : size(tmp_featmap,1)
        for sel_j = 1 : size(tmp_featmap,2)
            if highlight(sel_i,sel_j)
                tmp_sel_feat_L28 = [tmp_sel_feat_L28, tmp_featmap(sel_i,sel_j,:)];
            end
        end
    end
    
    if imdb.images.set(1,i) == 1
        % train data
        cnnFeat_tr_L31{count_tr,1} = tmp_sel_feat_L31; 
        cnnFeat_tr_L28{count_tr,1} = tmp_sel_feat_L28;
    else
        % test data
        cnnFeat_te_L31{count_te,1} = tmp_sel_feat_L31; 
        cnnFeat_te_L28{count_te,1} = tmp_sel_feat_L28;
    end
    
    %% horizontal flip
    im = fliplr(im);
    im_ = single(im);
    [h,w,~] = size(im_);
    if min(h,w) > 700
        im_ = imresize(im_, [h*(700/min(h,w)) w*(700/min(h,w))]);
    end
    [h,w,c] = size(im_);
    if  c > 2
        im_ = im_ - imresize(imdb.averageImage,[h,w]) ;
    else    
        im_ = bsxfun(@minus,im_,imresize(imdb.averageImage,[h,w])) ;
    end
    
    res = vl_simplenn(net, gpuArray(im_)) ;
%     res = vl_simplenn(net, im_);
    tmp_1 = gather(res(32).x);
    tmp_2 = gather(res(29).x);
    
    % Pool5
    tmp_featmap = tmp_1;
    tmp_featmap_sum = sum(tmp_featmap, 3);
    tmp_mean = mean(mean(tmp_featmap_sum));
    highlight = zeros(size(tmp_featmap_sum));
    highlight(find(tmp_featmap_sum>tmp_mean)) = 1;

    cc = bwconncomp(highlight);
    numPixel = cellfun(@numel,cc.PixelIdxList);
    [~,conn_idx] = max(numPixel);
    highlight_conn_L31 = zeros(size(highlight));
    highlight_conn_L31(cc.PixelIdxList{conn_idx}) = 1;
    
    tmp_sel_feat_L31 = [];
    for sel_i = 1 : size(tmp_featmap,1)
        for sel_j = 1 : size(tmp_featmap,2)
            if highlight_conn_L31(sel_i,sel_j)
                tmp_sel_feat_L31 = [tmp_sel_feat_L31, tmp_featmap(sel_i,sel_j,:)];
            end
        end
    end
    
    % Relu5_2
    tmp_featmap = tmp_2;
    tmp_featmap_sum = sum(tmp_featmap, 3);
    tmp_mean = mean(mean(tmp_featmap_sum));
    highlight = zeros(size(tmp_featmap_sum));
    highlight(find(tmp_featmap_sum>tmp_mean)) = 1;
    highlight28 = highlight;
    highlight = highlight28 & imresize(highlight_conn_L31, size(highlight28), 'nearest');
    tmp_sel_feat_L28 = [];
    for sel_i = 1 : size(tmp_featmap,1)
        for sel_j = 1 : size(tmp_featmap,2)
            if highlight(sel_i,sel_j)
                tmp_sel_feat_L28 = [tmp_sel_feat_L28, tmp_featmap(sel_i,sel_j,:)];
            end
        end
    end
    
    if imdb.images.set(1,i) == 1
        % train data
        cnnFeat_tr_L31{count_tr,2} = tmp_sel_feat_L31; 
        cnnFeat_tr_L28{count_tr,2} = tmp_sel_feat_L28;
        count_tr = count_tr + 1;
    else
        % test data
        cnnFeat_te_L31{count_te,2} = tmp_sel_feat_L31; 
        cnnFeat_te_L28{count_te,2} = tmp_sel_feat_L28;
        count_te = count_te + 1;
    end
    
    ex_time(i,1) = toc;
    disp(['Extracing ' opt.dataset ': ' num2str(i) 'th image (' num2str(i*100/size(imdb.images.name,2)) '%) used ' num2str(ex_time(i,1)) 's ...']);
end
train_label = imdb.images.class(find(imdb.images.set==1))';
test_label = imdb.images.class(find(imdb.images.set==3))';

save([path opt.dataset '/SCDA_flip_plus.mat'],'cnnFeat_tr_L31','cnnFeat_te_L31','cnnFeat_tr_L28','cnnFeat_te_L28',...
    'train_label','test_label','ex_time','-v7.3');

disp(['Feature extracting of ' opt.dataset ' is finished ...']);
