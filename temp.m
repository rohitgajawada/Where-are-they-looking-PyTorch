img = imread('../data/test/00000000/00000116.jpg');
e = [0.37636364  0.45393939];

filelist = cell(1,3);   
filelist(:,1) = {img};

alpha = 0.3;
w_x = floor(alpha*size(img,2));
w_y = floor(alpha*size(img,1));
if(mod(w_x,2)==0)
    w_x = w_x +1;
end

if(mod(w_y,2)==0)
    w_y = w_y +1;
end

im_face = ones(w_y,w_x,3,'uint8');
im_face(:,:,1) = 123*ones(w_y,w_x,'uint8');
im_face(:,:,2) = 117*ones(w_y,w_x,'uint8');
im_face(:,:,3) = 104*ones(w_y,w_x,'uint8');
center = floor([e(1)*size(img,2) e(2)*size(img,1)]);
d_x = floor((w_x-1)/2);
d_y = floor((w_y-1)/2);

bottom_x = center(1)-d_x;
delta_b_x = 1;
if(bottom_x<1)
    delta_b_x =2-bottom_x;
    bottom_x=1;
end
top_x = center(1)+d_x;
delta_t_x = w_x;
if(top_x>size(img,2))
     delta_t_x = w_x-(top_x-size(img,2));
     top_x = size(img,2);
end
bottom_y = center(2)-d_y;
delta_b_y = 1;
if(bottom_y<1)
    delta_b_y =2-bottom_y;
    bottom_y=1;
end
top_y = center(2)+d_y;
delta_t_y = w_y;
if(top_y>size(img,1))
     delta_t_y = w_y-(top_y-size(img,1));
     top_y = size(img,1);
end

im_face(delta_b_y:delta_t_y,delta_b_x:delta_t_x,:) = img(bottom_y:top_y,bottom_x:top_x,:);
filelist(:,2) = {im_face};

imshow(im_face);