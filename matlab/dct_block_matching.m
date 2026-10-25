function [mask, stats, pairs] = dct_block_matching(gray)
% Find textured blocks that repeat with a consistent movement.
% pairs contains verified block centers: [x1 y1 x2 y2].

    block = 16;
    stride = 4;
    maxDistance = 0.30;
    minSeparation = 32;
    minGroupSize = 12;
    binSize = 8;
    neighborsToCheck = 12;

    mask = false(size(gray));
    pairs = zeros(0, 4);
    stats = struct('Candidates', 0, 'ConsistentPairs', 0, 'Groups', 0);
    [height, width] = size(gray);
    if height < block || width < block
        return;
    end

    gridSize = floor(([height, width] - block) / stride) + 1;
    coords = zeros(prod(gridSize), 2);
    vectors = zeros(prod(gridSize), 15, 'single');

    % The first 15 zigzag AC coefficients describe coarse texture. Skip the
    % DC coefficient because it mostly describes the block's brightness.
    frequency = [1 2; 2 1; 3 1; 2 2; 1 3; 1 4; 2 3; 3 2; ...
                 4 1; 5 1; 4 2; 3 3; 2 4; 1 5; 1 6];
    frequencyIndex = sub2ind([block, block], frequency(:, 1), frequency(:, 2));
    count = 0;
    for y = 1:stride:(height - block + 1)
        for x = 1:stride:(width - block + 1)
            patch = double(gray(y:y+block-1, x:x+block-1));
            if std(patch(:)) < 12
                continue; % Flat areas are too ambiguous to match reliably.
            end
            coefficients = dct2(patch);
            descriptor = coefficients(frequencyIndex);
            energy = norm(descriptor);
            if energy < 1
                continue;
            end
            count = count + 1;
            coords(count, :) = [x, y];
            vectors(count, :) = single(descriptor' / energy);
        end
    end
    coords = coords(1:count, :);
    vectors = vectors(1:count, :);
    if count < minGroupSize
        return;
    end

    % Similar descriptors often become neighbors after sorting. Comparing
    % a short window keeps this step manageable without a large search tree.
    [~, order] = sortrows(vectors);
    matches = zeros(count * 3, 5); % The last column stores descriptor distance.
    matchCount = 0;
    for k = 1:count
        i = order(k);
        neighbors = order(max(1, k-neighborsToCheck):min(count, k+neighborsToCheck));
        distances = sqrt(sum((vectors(neighbors, :) - vectors(i, :)).^2, 2));
        [distances, nearest] = sort(distances);
        added = 0;
        for j = 1:numel(nearest)
            if distances(j) >= maxDistance
                break;
            end
            other = neighbors(nearest(j));
            a = coords(i, :);
            b = coords(other, :);
            if norm(a - b) < minSeparation
                continue;
            end
            % Store each pair in one spatial direction, never twice.
            if a(1) > b(1) || (a(1) == b(1) && a(2) > b(2))
                continue;
            end
            matchCount = matchCount + 1;
            matches(matchCount, :) = [a, b, double(distances(j))];
            added = added + 1;
            if added == 3
                break;
            end
        end
    end
    matches = matches(1:matchCount, :);
    stats.Candidates = matchCount;
    if matchCount == 0
        return;
    end

    % A copied region should contain many matches with the same movement.
    movement = matches(:, 3:4) - matches(:, 1:2);
    [~, ~, groupIndex] = unique(round(movement / binSize), 'rows');
    for group = 1:max(groupIndex)
        selected = matches(groupIndex == group, :);
        if size(selected, 1) < minGroupSize
            continue;
        end

        % Count each block at most once in this group. Otherwise a handful
        % of blocks with several similar neighbors could inflate support.
        [~, qualityOrder] = sort(selected(:, 5));
        selected = selected(qualityOrder, :);
        sourceIDs = sub2ind(gridSize, (selected(:, 2)-1)/stride+1, ...
            (selected(:, 1)-1)/stride+1);
        targetIDs = sub2ind(gridSize, (selected(:, 4)-1)/stride+1, ...
            (selected(:, 3)-1)/stride+1);
        sourceUsed = false(gridSize);
        targetUsed = false(gridSize);
        keep = false(size(selected, 1), 1);
        for j = 1:size(selected, 1)
            if ~sourceUsed(sourceIDs(j)) && ~targetUsed(targetIDs(j))
                keep(j) = true;
                sourceUsed(sourceIDs(j)) = true;
                targetUsed(targetIDs(j)) = true;
            end
        end
        selected = selected(keep, 1:4);
        sourceIDs = sourceIDs(keep);

        % Require neighboring block positions, not scattered coincidences.
        labels = bwlabel(sourceUsed, 8);
        componentIDs = labels(sourceIDs);
        acceptedGroup = false;
        for component = 1:max(componentIDs)
            coherent = selected(componentIDs == component, :);
            if size(coherent, 1) < minGroupSize
                continue;
            end
            sourceSpan = max(coherent(:, 1:2), [], 1) - min(coherent(:, 1:2), [], 1);
            targetSpan = max(coherent(:, 3:4), [], 1) - min(coherent(:, 3:4), [], 1);
            if any(sourceSpan < 20) || any(targetSpan < 20)
                continue;
            end
            acceptedGroup = true;
            stats.ConsistentPairs = stats.ConsistentPairs + size(coherent, 1);
            pairs = [pairs; coherent + (block-1)/2]; %#ok<AGROW>
            for j = 1:size(coherent, 1)
                a = coherent(j, 1:2);
                b = coherent(j, 3:4);
                mask(a(2):a(2)+block-1, a(1):a(1)+block-1) = true;
                mask(b(2):b(2)+block-1, b(1):b(1)+block-1) = true;
            end
        end
        stats.Groups = stats.Groups + acceptedGroup;
    end
end
