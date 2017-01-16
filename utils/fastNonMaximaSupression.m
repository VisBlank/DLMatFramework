function [ top_box ] = fastNonMaximaSupression( b_box, overlapThresh )
% Greedily select high-scoring detections and skip detections
% that are significantly covered by a previously selected
% detection.

% The input expected for the b_box is [x,y,width,height], for calculation
% we need sometimes to convert to [x1,y1,x2,y2]

% Return empty if there is no box given
if isempty(b_box)
    top_box = [];
    return;
end

% Convert to single
b_box = single(b_box);

x1 = b_box(:,1);
y1 = b_box(:,2);
widths = b_box(:,3);
heights = b_box(:,4);

% Get the x2,y2 components first convert b_box to he format [x1,y1,x2,y2]
bbox_conv = [b_box(:,1),b_box(:,2),b_box(:,1)+b_box(:,3),b_box(:,2)+b_box(:,4)];
x2 = bbox_conv(:,3);
y2 = bbox_conv(:,4);

% compute the area of the bounding boxes
areas = widths .* heights;
%area2 = (x2-x1+1) .* (y2-y1+1);

% Sort the bottom-right y-coordinate of the bounding b_box. We need this
% because we will compute the overlap ratio of other bounding b_box
[~, I] = sort(y2);

pick = zeros(size(b_box,1),1);
counter = 1;

while ~isempty(I)
    last = length(I);
    i = I(last);
    pick(counter) = i;
    counter = counter + 1;
    
    % Get the jaccard index (IoU) between the current box and all the
    % others
    xx1 = max(x1(i), x1(I(1:last-1)));
    yy1 = max(y1(i), y1(I(1:last-1)));
    xx2 = min(x2(i), x2(I(1:last-1)));
    yy2 = min(y2(i), y2(I(1:last-1)));
    
    % compute the width and height of the bounding box
    w = max(0.0, xx2-xx1+1);
    h = max(0.0, yy2-yy1+1);
    
    % Compute the overlap
    overlap = (w.*h) ./ areas(I(1:last-1));
    
    % Delete items from the list
    I([last; find(overlap>overlapThresh(1:last-1))]) = [];
end

pick = pick(1:(counter-1));
top_box = b_box(pick,:);


end

