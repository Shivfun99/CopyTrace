function result = evaluation(imagePath, truthPath, outputDir, showFigure)
% Run detection, save a report, and optionally compare with a known truth mask.
% Example: evaluation('photo.png', 'mask.png', 'results/my_photo', false)

    if isempty(which('dct2')) || isempty(which('regionprops'))
        error('evaluation:MissingImageToolbox', 'This project requires Image Processing Toolbox.');
    end
    if isempty(which('detectORBFeatures')) || isempty(which('estgeotform2d')) || isempty(which('insertShape'))
        error('evaluation:MissingVisionToolbox', ...
            ['The full pipeline requires Computer Vision Toolbox and MATLAB R2022b or later. ' ...
             'You can run test_project for the available stages, or use the C++ program.']);
    end

    base = fileparts(mfilename('fullpath'));
    useDemo = nargin < 1 || isempty(imagePath);
    if useDemo
        imagePath = fullfile(base, '..', 'dataset', 'example_copy_move.png');
        if ~isfile(imagePath)
            create_demo;
        end
    end
    if nargin < 2
        truthPath = '';
        if useDemo
            truthPath = fullfile(base, '..', 'dataset', 'example_mask.png');
        end
    end
    if nargin < 3 || isempty(outputDir)
        [~, name] = fileparts(imagePath);
        outputDir = fullfile(base, '..', 'results', 'matlab', name);
    end
    if nargin < 4
        showFigure = true;
    end

    % Read the image. Large inputs are processed at a maximum side of 768.
    [gray, rgb, originalSize] = preprocessing(imagePath);
    truth = [];
    if ~isempty(truthPath)
        [rawTruth, map] = imread(truthPath);
        if ~isequal([size(rawTruth, 1), size(rawTruth, 2)], originalSize)
            error('The truth mask must have the same dimensions as the original image.');
        end
        if ~isempty(map)
            rawTruth = ind2rgb(rawTruth, map);
        end
        % Nonzero color labels count too; grayscale conversion can lose faint ones.
        truth = any(rawTruth ~= 0, 3);
        truth = imresize(truth, size(gray), 'nearest');
    end

    % Time detection and localization only; reading and saving files are excluded.
    started = tic;
    [dctMask, dctStats, dctPairs] = dct_block_matching(gray);
    [orbMask, orbStats, orbPairs] = orb_matching(gray);
    [finalMask, overlay, boxes] = region_localization(rgb, dctMask, orbMask);
    seconds = toc(started);

    result = struct('OriginalSize', originalSize, 'ProcessedSize', size(gray), ...
        'DCT', dctStats, 'ORB', orbStats, 'Regions', size(boxes, 1), ...
        'Boxes', boxes, 'FlaggedPercentage', 100 * nnz(finalMask) / numel(finalMask), ...
        'TimeSeconds', seconds, 'Mask', finalMask);
    result.Assessment = 'NO STRONG COPY-MOVE INDICATORS';
    if any(finalMask(:))
        result.Assessment = 'SUSPICIOUS MATCHED REGIONS';
    end

    % Truth must mark BOTH the source and pasted area for a fair comparison.
    % Zero denominators give zero; judge authentic controls by their image flag.
    if ~isempty(truth)
        result.TP = nnz(finalMask & truth);
        result.FP = nnz(finalMask & ~truth);
        result.FN = nnz(~finalMask & truth);
        result.Precision = result.TP / max(result.TP + result.FP, 1);
        result.Recall = result.TP / max(result.TP + result.FN, 1);
        result.F1 = 2 * result.TP / max(2 * result.TP + result.FP + result.FN, 1);
        result.IoU = result.TP / max(result.TP + result.FP + result.FN, 1);
    end

    % Save the same stages as C++, plus a structure for further MATLAB analysis.
    names = {'01_input.png', '02_grayscale.png', '03_dct_mask.png', ...
        '04_orb_mask.png', '05_final_mask.png', '06_highlighted.png', ...
        '07_orb_matches.png', '08_dct_matches.png', 'report.txt', 'result.mat'};
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    protect_inputs(imagePath, truthPath, outputDir, names);
    orbView = draw_pairs(rgb, orbPairs);
    dctView = draw_pairs(rgb, dctPairs);
    images = {rgb, gray, dctMask, orbMask, finalMask, overlay, orbView, dctView};
    for k = 1:numel(images)
        imwrite(images{k}, fullfile(outputDir, names{k}));
    end
    save(fullfile(outputDir, 'result.mat'), 'result');
    write_report(fullfile(outputDir, 'report.txt'), imagePath, result);

    fprintf('\n%s\n', result.Assessment);
    fprintf('DCT pairs: %d | ORB inliers: %d | Regions: %d\n', ...
        dctStats.ConsistentPairs, orbStats.Inliers, result.Regions);
    fprintf('Flagged area: %.2f%% | Processing: %.3f s\n', result.FlaggedPercentage, seconds);
    if isfield(result, 'F1')
        fprintf('Precision: %.4f | Recall: %.4f | F1: %.4f | IoU: %.4f\n', ...
            result.Precision, result.Recall, result.F1, result.IoU);
    end
    fprintf('Results saved in: %s\n', outputDir);

    % Disable the figure when running tests or using a machine without a display.
    if showFigure
        figure('Name', 'Copy-Move Forgery Detection');
        panels = {rgb, orbView, dctMask, orbMask, finalMask, overlay};
        titles = {'Input', 'Verified ORB matches', 'DCT mask', ...
            'ORB mask', 'Final mask', 'Suspicious regions'};
        for k = 1:6
            subplot(2, 3, k);
            imshow(panels{k});
            title(titles{k});
        end
    end
end

function view = draw_pairs(rgb, pairs)
% Show at most 60 lines so the underlying image stays readable.
    view = rgb;
    if isempty(pairs)
        return;
    end
    chosen = unique(round(linspace(1, size(pairs, 1), min(60, size(pairs, 1)))));
    view = insertShape(view, 'Line', pairs(chosen, :), 'Color', 'green', 'LineWidth', 1);
end

function protect_inputs(imagePath, truthPath, outputDir, names)
% A chosen output folder must not overwrite the image or its truth mask.
    inputs = {imagePath, truthPath};
    for i = 1:numel(inputs)
        if isempty(inputs{i})
            continue;
        end
        [~, source] = fileattrib(inputs{i});
        for k = 1:numel(names)
            destination = fullfile(outputDir, names{k});
            if isfile(destination)
                [~, target] = fileattrib(destination);
                if strcmp(source.Name, target.Name)
                    error('Choose a different output folder; an output would overwrite an input.');
                end
            end
        end
    end
end

function write_report(path, imagePath, result)
% Keep the report readable enough to use during the project demonstration.
    fid = fopen(path, 'w');
    if fid < 0
        error('Cannot write report: %s', path);
    end
    closeFile = onCleanup(@() fclose(fid));
    fprintf(fid, 'CLASSICAL DIP COPY-MOVE ANALYSIS\n===============================\n');
    fprintf(fid, 'Image: %s\n', imagePath);
    fprintf(fid, 'Original dimensions: %d x %d\n', result.OriginalSize(2), result.OriginalSize(1));
    fprintf(fid, 'Processed dimensions: %d x %d\n', result.ProcessedSize(2), result.ProcessedSize(1));
    fprintf(fid, 'DCT candidate pairs: %d\n', result.DCT.Candidates);
    fprintf(fid, 'DCT consistent pairs: %d\n', result.DCT.ConsistentPairs);
    fprintf(fid, 'DCT displacement groups: %d\n', result.DCT.Groups);
    fprintf(fid, 'ORB candidate matches: %d\n', result.ORB.Candidates);
    fprintf(fid, 'ORB RANSAC inliers: %d\n', result.ORB.Inliers);
    fprintf(fid, 'Suspicious connected regions: %d\n', result.Regions);
    fprintf(fid, 'Flagged area: %.2f%%\nDetection and localization time: %.3f s\n', ...
        result.FlaggedPercentage, result.TimeSeconds);
    fprintf(fid, 'Assessment: %s\n', result.Assessment);
    if isfield(result, 'F1')
        fprintf(fid, 'TP: %d | FP: %d | FN: %d\n', result.TP, result.FP, result.FN);
        fprintf(fid, 'Precision: %.4f\nRecall: %.4f\nF1: %.4f\nIoU: %.4f\n', ...
            result.Precision, result.Recall, result.F1, result.IoU);
        fprintf(fid, 'Metrics use nonzero truth pixels; zero denominators give zero.\n');
    end
    fprintf(fid, 'Boxes use processed-image coordinates: x, y, width, height.\n');
    for k = 1:result.Regions
        fprintf(fid, 'Region %d: %.1f, %.1f, %.1f, %.1f\n', k, result.Boxes(k, :));
    end
    fprintf(fid, 'Match pictures show at most 60 lines; counts include all accepted matches.\n');
    fprintf(fid, 'Both source and copy may be marked; their roles cannot be determined.\n');
    fprintf(fid, 'Repeated natural patterns can trigger matches. Results are not proof of forgery.\n');
    fprintf(fid, 'No strong indicators does not establish that an image is authentic.\n');
    [message, code] = ferror(fid);
    if code ~= 0
        error('Cannot finish report: %s', message);
    end
end
