function create_demo(outputDir)
% Make a repeatable copy-move example and its source+copy mask.
% These synthetic images check the program; they are not an accuracy benchmark.

    if nargin < 1
        outputDir = fullfile(fileparts(mfilename('fullpath')), '..', 'dataset');
    end
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Keep the same texture without changing the caller's random seed.
    previousState = rng;
    restoreState = onCleanup(@() rng(previousState));
    rng(7);
    height = 320;
    width = 384;
    texture = conv2(rand(height, width), ones(3) / 9, 'same');
    texture = (texture - min(texture(:))) / (max(texture(:)) - min(texture(:)));
    original = repmat(uint8(30 + 200 * texture), 1, 1, 3);

    % Colored shapes make the copied area easy to see during the demonstration.
    original(65:120, 57:136, 1) = 220;
    original(65:120, 57:136, 2) = 60;
    original(65:120, 57:136, 3) = 40;
    original(77:108, 69:124, :) = repmat(uint8(230 * texture(77:108, 69:124)), 1, 1, 3);
    original(129:144, 57:100, 2) = 230;

    % Copy one patch to a separate location. Both areas belong in the truth mask.
    sourceRows = 49:160;
    sourceCols = 41:152;
    targetRows = 177:288;
    targetCols = 217:328;
    forged = original;
    forged(targetRows, targetCols, :) = original(sourceRows, sourceCols, :);
    truth = false(height, width);
    truth(sourceRows, sourceCols) = true;
    truth(targetRows, targetCols) = true;

    imwrite(original, fullfile(outputDir, 'example_original.png'));
    imwrite(forged, fullfile(outputDir, 'example_copy_move.png'));
    imwrite(truth, fullfile(outputDir, 'example_mask.png'));
    fprintf('Demo images saved in: %s\n', outputDir);
end
