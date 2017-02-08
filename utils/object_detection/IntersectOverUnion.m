function [ IoU ] = IntersectOverUnion( bboxA, bboxB )

%% Get the (x,y) coordinate of the intersection rectangle between A,B
% Box format [x,y,width,height]

% Get (x,y,width,height) coordinates of intersection
xInter = max(bboxA(1), bboxB(1));
yInter = max(bboxA(2), bboxB(2));

% Convert rect format to (x1,y1,x2,y2), just to compute the width/height intersection
bboxA_conv = [bboxA(1),bboxA(2),bboxA(1)+bboxA(3),bboxA(2)+bboxA(4)];
bboxB_conv = [bboxB(1),bboxA(2),bboxB(1)+bboxB(3),bboxB(2)+bboxB(4)];
xB_inter = min(bboxA_conv(3),bboxB_conv(3));
yB_inter = min(bboxA_conv(4),bboxB_conv(4));

% Intersection width and height, now we convert our [x1,y1,x2,y2] format
% back to [x,y,w,h]
wInter = xB_inter - xInter;
hInter = yB_inter - yInter;

% Get intersection Area
AreaOverlap = wInter*hInter;

% Get the union Area
AreaUnion = ((bboxA(3)*bboxA(4)) + (bboxB(3)*bboxB(4))) - AreaOverlap;
IoU = AreaOverlap / single(AreaUnion);

end

