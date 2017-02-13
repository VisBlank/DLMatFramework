function [ image_out ] = addTextImage( image, text_str, varargin )
% Add some text on RGB image, and return an image with the added text
% References:
% https://uk.mathworks.com/videos/varargin-and-nargin-variable-inputs-to-a-function-97389.html
% https://uk.mathworks.com/help/vision/ref/inserttext.html
% Example
% catColor = imread('datasets/imgs/catColor.jpg');
% imshow(addTextImage(catColor,'Grey Cat','red'));
% imshow(addTextImage(catColor,'Grey Cat: 98%'));
imgHeight = size(image,1);
imgWidth = size(image,2);

posX = 0.25*(imgWidth/10);
posY = 0.5*(imgHeight/10);
position = [posX posY];
%position = [5 35];

numInputParameters = nargin;
if (numInputParameters > 2)
   color = varargin{1};
else
    color = 'green';
end

%image_out = insertText(image,position,text_str,'FontSize',18,'BoxColor',...
%    'green','BoxOpacity',0.4,'TextColor','white','AnchorPoint','LeftBottom');

image_out = insertText(image,position,text_str,'FontSize',18,'BoxColor',...
    color,'BoxOpacity',0.4,'TextColor','white');
end

