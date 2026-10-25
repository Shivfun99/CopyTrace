function test_project
% Run small, repeatable checks. These are not dataset accuracy measurements.

    outputDir = tempname;
    create_demo(outputDir);
    copiedPath = fullfile(outputDir, 'example_copy_move.png');
    originalPath = fullfile(outputDir, 'example_original.png');
    truthPath = fullfile(outputDir, 'example_mask.png');

    % A known copy should produce matching blocks in both marked regions.
    [gray, rgb, originalSize] = preprocessing(copiedPath);
    truth = imread(truthPath) ~= 0;
    assert(isequal(originalSize, size(gray)), 'Unexpected demo image size.');
    [dctMask, stats, pairs] = dct_block_matching(gray);
    assert(stats.ConsistentPairs >= 12, 'The known copy was not detected.');
    assert(size(pairs, 1) == stats.ConsistentPairs, 'DCT pair count disagrees.');
    [mask, overlay, boxes] = region_localization(rgb, dctMask, false(size(gray)));
    assert(nnz(mask & truth) > 0.5 * nnz(truth), 'Too little of the copy was found.');
    assert(size(boxes, 1) >= 2, 'Source and copy should form separate regions.');
    yellow = overlay(:, :, 1) == 255 & overlay(:, :, 2) == 255 & ...
        overlay(:, :, 3) == 0;
    assert(any(yellow(:)), 'Bounding boxes were not drawn.');
    imwrite(mask, fullfile(outputDir, 'tested_dct_mask.png'));
    imwrite(overlay, fullfile(outputDir, 'tested_overlay.png'));

    % The same scene before copying should not trigger the DCT branch.
    original = preprocessing(originalPath);
    [authenticMask, authenticStats] = dct_block_matching(original);
    assert(~any(authenticMask(:)), 'The authentic control triggered DCT.');
    assert(authenticStats.ConsistentPairs == 0, 'Unexpected authentic matches.');

    % Empty evidence and images smaller than a block should remain empty.
    [flatMask, flatStats] = dct_block_matching(zeros(64, 64, 'uint8'));
    [tinyMask, ~, tinyPairs] = dct_block_matching(zeros(8, 8, 'uint8'));
    assert(~any(flatMask(:)) && flatStats.Candidates == 0);
    assert(~any(tinyMask(:)) && isempty(tinyPairs));
    blankRGB = zeros(64, 64, 3, 'uint8');
    [clean, blankOverlay, emptyBoxes] = region_localization(blankRGB, flatMask, flatMask);
    assert(~any(clean(:)) && isempty(emptyBoxes) && isequal(blankOverlay, blankRGB));
    [tinyORB, tinyStats] = orb_matching(zeros(8, 8, 'uint8'));
    assert(~any(tinyORB(:)) && tinyStats.Inliers == 0);
    fprintf('PASS: preprocessing, DCT, localization, and empty/tiny images.\n');

    % ORB and complete report generation need Computer Vision Toolbox.
    hasVision = exist('detectORBFeatures', 'file') ~= 0 && ...
        exist('extractFeatures', 'file') ~= 0 && ...
        exist('estgeotform2d', 'file') ~= 0 && ...
        exist('insertShape', 'file') ~= 0;
    if hasVision
        [orbMask, orbStats, orbPairs] = orb_matching(gray);
        assert(orbStats.Inliers >= 8 && any(orbMask(:)), 'ORB missed the known copy.');
        assert(size(orbPairs, 1) == orbStats.Inliers, 'ORB pair count disagrees.');
        reportDir = fullfile(outputDir, 'full_result');
        result = evaluation(copiedPath, truthPath, reportDir, false);
        assert(result.TP > 0 && result.Regions >= 2, 'Full detection failed.');
        assert(isfile(fullfile(reportDir, 'report.txt')), 'Report was not saved.');
        assert(isfile(fullfile(reportDir, '08_dct_matches.png')), 'Matches were not saved.');
        authentic = evaluation(originalPath, '', fullfile(outputDir, 'authentic_result'), false);
        assert(~any(authentic.Mask(:)), 'The authentic control triggered detection.');
        fprintf('PASS: ORB and complete evaluation on copied and authentic images.\n');
    else
        fprintf('SKIP: ORB and full evaluation require Computer Vision Toolbox.\n');
    end
    fprintf('Test images and results are in: %s\n', outputDir);
end
