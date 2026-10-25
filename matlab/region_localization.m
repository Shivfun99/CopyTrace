function [cleanMask, overlay, boxes] = region_localization(rgb, dctMask, orbMask)
% Join the two detectors' evidence and highlight the matching regions.

    % Closing joins small gaps; opening removes thin, isolated edges.
    combined = logical(dctMask) | logical(orbMask);
    combined = imclose(combined, strel('disk', 4, 0));
    combined = imopen(combined, strel('disk', 1, 0));
    cleanMask = bwareaopen(combined, 180);

    regions = regionprops(cleanMask, 'BoundingBox');
    boxes = reshape([regions.BoundingBox], 4, [])';

    % Blend red only into the flagged pixels, leaving the rest unchanged.
    overlay = rgb;
    red = [255, 30, 30];
    for channel = 1:3
        layer = overlay(:, :, channel);
        layer(cleanMask) = uint8(0.55 * double(layer(cleanMask)) + ...
            0.45 * red(channel));
        overlay(:, :, channel) = layer;
    end

    % Draw a two-pixel yellow border around each connected region.
    [height, width, ~] = size(overlay);
    yellow = [255, 255, 0];
    for k = 1:size(boxes, 1)
        box = boxes(k, :);
        x1 = max(1, round(box(1)));
        y1 = max(1, round(box(2)));
        x2 = min(width, round(box(1) + box(3) - 1));
        y2 = min(height, round(box(2) + box(4) - 1));

        for thickness = 0:1
            left = x1 + thickness;
            right = x2 - thickness;
            top = y1 + thickness;
            bottom = y2 - thickness;
            if left > right || top > bottom
                break;
            end
            for channel = 1:3
                overlay([top, bottom], left:right, channel) = yellow(channel);
                overlay(top:bottom, [left, right], channel) = yellow(channel);
            end
        end
    end
end
