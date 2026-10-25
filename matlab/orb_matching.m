function [mask, stats, pairs] = orb_matching(gray)
% Match distinctive image points and check their shared geometric movement.
% pairs contains verified point coordinates: [x1 y1 x2 y2].

    mask = false(size(gray));
    pairs = zeros(0, 4);
    stats = struct('Candidates', 0, 'Inliers', 0);
    if min(size(gray)) < 48
        return;
    end

    % ORB describes corners with binary features and can tolerate some
    % rotation and resizing. Smooth regions may have no useful corners.
    points = detectORBFeatures(gray);
    points = points.selectStrongest(900);
    if points.Count < 16
        return;
    end
    [descriptors, points] = extractFeatures(gray, points);
    raw = descriptors.Features;
    locations = double(points.Location);
    count = size(raw, 1);
    if count < 16
        return;
    end

    % Hamming distance counts different bits in two binary descriptors.
    popcount = zeros(256, 1);
    for value = 0:255
        popcount(value+1) = sum(bitget(uint8(value), 1:8));
    end
    bestNeighbor = zeros(count, 1);
    bestDistance = inf(count, 1);
    for i = 1:count
        differentBits = bitxor(raw, raw(i, :));
        distance = sum(popcount(double(differentBits)+1), 2);
        separation = sqrt(sum((locations - locations(i, :)).^2, 2));
        distance(separation < 48) = inf; % Ignore self and nearby corners.
        [nearest, indices] = sort(distance);
        if nearest(1) <= 48 && nearest(1) < 0.8 * nearest(2)
            bestNeighbor(i) = indices(1);
            bestDistance(i) = nearest(1);
        end
    end

    % Both points must select each other. Order each pair by its position,
    % rather than by feature number, so its movement has a consistent sign.
    candidates = zeros(count, 5);
    candidateCount = 0;
    for i = 1:count
        j = bestNeighbor(i);
        if j <= i || bestNeighbor(j) ~= i
            continue;
        end
        a = locations(i, :);
        b = locations(j, :);
        if a(1) > b(1) || (a(1) == b(1) && a(2) > b(2))
            temporary = a;
            a = b;
            b = temporary;
        end
        candidateCount = candidateCount + 1;
        candidates(candidateCount, :) = [a, b, bestDistance(i)];
    end
    candidates = candidates(1:candidateCount, :);
    if candidateCount == 0
        return;
    end

    % ORB can find the same corner at several scales. Keep only one match
    % per small cell at each end so these duplicates do not inflate support.
    [~, order] = sort(candidates(:, 5));
    candidates = candidates(order, :);
    gridSize = ceil(size(gray) / 8);
    sourceUsed = false(gridSize);
    targetUsed = false(gridSize);
    keep = false(candidateCount, 1);
    for i = 1:candidateCount
        cells = floor((candidates(i, 1:4) - 1) / 8) + 1;
        sourceID = sub2ind(gridSize, cells(2), cells(1));
        targetID = sub2ind(gridSize, cells(4), cells(3));
        if ~sourceUsed(sourceID) && ~targetUsed(targetID)
            keep(i) = true;
            sourceUsed(sourceID) = true;
            targetUsed(targetID) = true;
        end
    end
    candidates = candidates(keep, 1:4);
    stats.Candidates = size(candidates, 1);
    if size(candidates, 1) < 8
        return;
    end

    % RANSAC rejects pairs that do not share one translation, rotation and
    % scale. Asking for status handles an ordinary failed fit explicitly.
    source = candidates(:, 1:2);
    target = candidates(:, 3:4);
    [transform, inliers, status] = estgeotform2d(source, target, 'similarity', ...
        'MaxDistance', 4, 'MaxNumTrials', 3000, 'Confidence', 99.5);
    if status ~= 0 || nnz(inliers) < 8
        return;
    end
    scale = norm(transform.A(1:2, 1));
    source = source(inliers, :);
    target = target(inliers, :);
    if scale < 0.55 || scale > 1.8 || ...
            ~hasEnoughSpread(source) || ~hasEnoughSpread(target)
        return;
    end
    pairs = [source, target];
    stats.Inliers = size(pairs, 1);

    % Mark a small neighborhood around each verified corner. These disks
    % show supported areas; they do not recover an exact object boundary.
    radius = 13;
    [dx, dy] = meshgrid(-radius:radius, -radius:radius);
    disk = (dx.^2 + dy.^2) <= radius^2;
    for i = 1:size(pairs, 1)
        mask = stampDisk(mask, round(source(i, :)), disk, radius);
        mask = stampDisk(mask, round(target(i, :)), disk, radius);
    end
end

function enough = hasEnoughSpread(points)
% A row of corners or one tiny cluster is not enough evidence for a region.
    enough = false;
    span = max(points, [], 1) - min(points, [], 1);
    if any(span < 20) || rank(points - mean(points, 1)) < 2
        return;
    end
    boundary = convhull(points(:, 1), points(:, 2));
    enough = polyarea(points(boundary, 1), points(boundary, 2)) >= 400;
end

function mask = stampDisk(mask, point, disk, radius)
% Clip the disk at the image border before copying it into the mask.
    x1 = max(1, point(1)-radius);
    x2 = min(size(mask, 2), point(1)+radius);
    y1 = max(1, point(2)-radius);
    y2 = min(size(mask, 1), point(2)+radius);
    diskX = (x1:x2) - point(1) + radius + 1;
    diskY = (y1:y2) - point(2) + radius + 1;
    mask(y1:y2, x1:x2) = mask(y1:y2, x1:x2) | disk(diskY, diskX);
end
