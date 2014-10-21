caffe('reset')
model_def = '../../models/bvlc_reference_caffenet/bvlc_reference_nosoftmax.prototxt';
matcaffe_init(1, model_def);
im = single(imread('../../examples/images/cat.jpg'));
dim = 227;
im = imresize(im, [dim, dim], 'bilinear');
d = load('ilsvrc_2012_mean');
im_center = im(:, :, [3 2 1]) - d.image_mean(1:dim, 1:dim, :);
im_center = permute(im_center, [2, 1, 3]);

f = caffe('forward', {im_center});
top_data = squeeze(f{1});
[maxV, maxI] = max(top_data);
[sortV, sortI] = sort(top_data, 1, 'descend');
top_diff = eps(1) * ones(1, 1, 4096, 1, 'single');
top_diff(1, 1, sortI(100), 1) = 1;
% b = caffe('backward', f, 0);  
b = caffe('backward', {top_diff}, 0);   

r_im = permute(b{1}, [2, 1, 3]);
r_im = r_im / max(r_im(:)) * 110;
r_im = d.image_mean(1:dim, 1:dim, :) + r_im;
r_im = r_im(:, :, [3 2 1]);
r_im = uint8(r_im);
figure
subplot(1, 3, 1)
imshow(uint8(im));
subplot(1, 3, 2);
imshow(r_im)
subplot(1, 3, 3);
r_im_grey = mean(r_im, 3) / 256;
imshow(r_im_grey);