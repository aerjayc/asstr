cd '/path/to/gt.mat'

gt = load('gt.mat');
% charBB
% wordBB
% imnames
% txt

save('gt_v7.3.mat', 'charBB', 'wordBB', 'imnames', 'txt', '-v7.3')


%{
len = 858750;

N = 50000;
i = 15;
end_ = (i+1)*N;
while end_ < len
    fprintf('iter: %i\n', i);
    fprintf('end: %i\n', end_);
    start = (i*N) + 1;
    end_ = (i+1)*N;
    if end_ > len
        end_ = len;
    end
    
    charBB = gt.charBB(start:end_);
    wordBB = gt.wordBB(start:end_);
    imnames = gt.imnames(start:end_);
    txt = gt.txt(start:end_);
    
    filename = sprintf('parts1/gt_%d.mat', i);
    save(filename, 'charBB', 'wordBB', 'imnames', 'txt', '-v7.3')
    
    i = i + 1;
end
%}

%filename = sprintf('parts1/gt_%d.mat', i);