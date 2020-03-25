clear;close all;

folder = '291';

%savepath = 'train.h5';
size_input = 41;
size_label = 41;
stride = 41;
saveInputPath = 'Train_sub_bic';
saveLabelPath = 'Train_sub';
%% scale factors
scale = [2,4];
%% downsizing
downsizes = [1,0.7,0.5];

%% initialization
%data = zeros(size_input, size_input, 3, 10);
%label = zeros(size_label, size_label, 3, 10);
if ~exist(saveInputPath, 'dir')
    mkdir(saveInputPath);
end
if ~exist(saveLabelPath, 'dir')
    mkdir(saveLabelPath);
end

margain = 0;
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

%% generate data


for i = 1 : length(filepaths)
    count = 0;
    for flip = 1: 3
        for degree = 1 : 4
            for s = 1 : length(scale)
                for downsize = 1 : length(downsizes)
                    image = imread(fullfile(folder,filepaths(i).name));

                    if flip == 1
                        image = flipdim(image ,1);
                    end
                    if flip == 2
                        image = flipdim(image ,2);
                    end
                    
                    image = imrotate(image, 90 * (degree - 1));

                    image = imresize(image,downsizes(downsize),'bicubic');
                    
                    if size(image,3)==3            
                        % image = rgb2ycbcr(image);
                        % image = im2double(image(:, :, 1));
                        image = im2double(image);
                        im_label = modcrop(image, scale(s));
                        im_size = size(im_label);
                        hei = im_size(1);
                        wid = im_size(2);
                        im_input = imresize(imresize(im_label,1/scale(s),'bicubic'),[hei,wid],'bicubic');
                        
                        %filepaths(i).name
                        for x = 1 : stride : hei-size_input+1
                            for y = 1 :stride : wid-size_input+1

                                subim_input = im_input(x : x+size_input-1, y : y+size_input-1,:);
                                subim_label = im_label(x : x+size_label-1, y : y+size_label-1,:);
                                imwrite(subim_input, fullfile(saveInputPath, [filepaths(i).name(1:end-4)  '_' num2str(count) '.bmp']));  
                                imwrite(subim_label, fullfile(saveLabelPath, [filepaths(i).name(1:end-4)  '_' num2str(count) '.bmp']));  
                                count=count+1;

                                %data(:, :, :, count) = subim_input;
                                %label(:, :, :, count) = subim_label;
                            end
                        end
                    end
                end    
            end
        end
    end
end

%order = randperm(count);
%data = data(:, :, 1, order);
%label = label(:, :, 1, order); 

%% writing to HDF5
%chunksz = 64;
%created_flag = false;
%totalct = 0;

%for batchno = 1:floor(count/chunksz)

 %   last_read=(batchno-1)*chunksz;
  %  batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
  %  batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

   % startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
   % curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
   % created_flag = true;
   % totalct = curr_dat_sz(end);
%end

%h5disp(savepath);
