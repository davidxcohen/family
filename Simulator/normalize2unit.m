function output = normalize2unit(input)
output = input ./ repmat(sqrt(sum(input.^2)),3,1);
end