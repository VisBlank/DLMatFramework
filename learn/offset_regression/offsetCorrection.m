function [correctedRects] = offsetCorrection(inputRects, original_image)
correctedRects = inputRects;
options = weboptions('MediaType','application/json');

try
    for ii = 1:length(inputRects)
        if inputRects(ii).strength > 0.4
            % get image to process
            image = uint8(original_image(inputRects(ii).cornerRow:inputRects(ii).cornerRow+inputRects(ii).height,inputRects(ii).cornerCol:inputRects(ii).cornerCol+inputRects(ii).width,:));

            image_single = im2single(image);
            % resize image to 120x120 as model expects
            image_single = imresize(image_single, [120,120]);
            [x,y,z] = size(image_single);
            % send image to python server and get result
            offsets = webwrite('http://127.0.0.1:5000/angle_from_data',jsonencode(struct('rows',x,'cols',y,'depth',z,'data',image_single)), options);

            x_correct = round(-1*(inputRects(ii).width * offsets.output(1)));
            y_correct = round(-1*(inputRects(ii).height * offsets.output(2)));
            %corrected_im = uint8(original_image(inputRects(ii).cornerRow+y_correct:inputRects(ii).cornerRow+y_correct+inputRects(ii).height, inputRects(ii).cornerCol+x_correct:inputRects(ii).cornerCol+x_correct+inputRects(ii).width,:));

            correctedRects(ii).cornerRow = correctedRects(ii).cornerRow + y_correct;
        %     correctedRects(ii).height = correctedRects(ii).height + y_correct;
            correctedRects(ii).cornerCol = correctedRects(ii).cornerCol + x_correct;
        %     correctedRects(ii).width = correctedRects(ii).width + x_correct;
        end
    end
catch
    warning("couldnt correct offsets")
end




end


