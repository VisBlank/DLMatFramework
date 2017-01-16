function [ rect_out ] = convert_rect_type( rect_in )
%CONVERT_RECT_TYPE Convert rect from [x,w,width,height] to [x1,y1,x2,y2]

x1 = rect_in(:,1);
y1 = rect_in(:,2);
widths = rect_in(:,3);
heights = rect_in(:,4);

% Get the x2,y2 components first convert b_box to he format [x1,y1,x2,y2]
bbox_conv = [rect_in(:,1),rect_in(:,2),rect_in(:,1)+widths,rect_in(:,2)+heights];
x2 = bbox_conv(:,3);
y2 = bbox_conv(:,4);

rect_out = [x1 y1 x2 y2];

end

