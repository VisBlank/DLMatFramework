function calcMean = simpleMean(vecIn)
	total = 0;
	for i=1:size(vecIn,1)
		total = total + vecIn(i);
	end
	calcMean = total / size(vecIn,1);
end
