function [gray, rgb, originalSize] = preprocessing(imagePath)
% Read an image and make a smaller grayscale copy for the detectors.

    [rgb, map] = imread(imagePath);
    if ~isempty(map)
        rgb = ind2rgb(rgb, map);
    end
    originalSize = [size(rgb, 1), size(rgb, 2)];

    % Keep a color copy so the result can be drawn over the input image.
    if ismatrix(rgb)
        rgb = repmat(rgb, 1, 1, 3);
    elseif size(rgb, 3) == 4
        rgb = rgb(:, :, 1:3);
    end
    rgb = im2uint8(rgb);
    gray = im2gray(rgb);

    % Large images create many overlapping blocks. Keep the longest side
    % within 768 pixels, and apply exactly the same size to both copies.
    ratio = 768 / max(originalSize);
    if ratio < 1
        gray = imresize(gray, ratio);
        rgb = imresize(rgb, [size(gray, 1), size(gray, 2)]);
    end
end
