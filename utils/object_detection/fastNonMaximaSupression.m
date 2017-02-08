function [ top_box ] = fastNonMaximaSupression( b_box, IoUThresh, scores )
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

% Choose the first box to start
if isempty(scores)
    [~, sortedBoxesIdx] = sort(y2);
else
    [~, sortedBoxesIdx] = sort(scores);
end

% Pick will have the indexes all the remaining boxes
pick = zeros(size(b_box,1),1);
counter = 1;

while ~isempty(sortedBoxesIdx)
    % Select the last element of the sorted list
    last = length(sortedBoxesIdx);
    curBoxIdx = sortedBoxesIdx(last);
    otherBoxesIdx = sortedBoxesIdx(1:last-1);
    pick(counter) = curBoxIdx;
    counter = counter + 1;
    
    % Get the jaccard index (IoU) between the current box and all the
    % others
    xx1 = max(x1(curBoxIdx), x1(otherBoxesIdx));
    yy1 = max(y1(curBoxIdx), y1(otherBoxesIdx));
    xx2 = min(x2(curBoxIdx), x2(otherBoxesIdx));
    yy2 = min(y2(curBoxIdx), y2(otherBoxesIdx));    
    % compute the width and height of the bounding box
    w = max(0.0, xx2-xx1+1);
    h = max(0.0, yy2-yy1+1);
    
    % Compute the overlap (is not exactly the IoU formula but works)
    IoU = (w.*h) ./ areas(otherBoxesIdx);
    
    % Keep only elements with a IoU < IoUThresh (Delete elements with high
    % overlap)
    sortedBoxesIdx([last; find(IoU>IoUThresh)]) = [];
end

pick = pick(1:(counter-1));
top_box = b_box(pick,:);


end

