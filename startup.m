%% This function is called if you open matlab from the project folder...
function startup()
disp('Preparing project');

%% Add folders on the matlab path
curdir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(curdir, 'learn/MatLiveScripts')));
addpath(genpath(fullfile(curdir, 'utils')));
addpath(genpath(fullfile(curdir, 'solver')));
addpath(genpath(fullfile(curdir, 'test')));
addpath(genpath(fullfile(curdir, 'datasets')));
addpath(genpath(fullfile(curdir, 'layers')));
addpath(genpath(fullfile(curdir, 'classifiers')));
addpath(genpath(fullfile(curdir, 'vanilla_examples')));
addpath(genpath(fullfile(curdir, 'loss')));

if ~verLessThan('matlab','8.4')
    % Add python folder
    addpath(genpath(fullfile(curdir, 'python_reference_code')));
    insert(py.sys.path,int32(0),[pwd filesep 'python_reference_code']);
else
    warning('Python not supported');
end

%% Configure Simulink worker directory
projectRoot = pwd;
myCacheFolder = fullfile(projectRoot, 'work');
if ~exist(myCacheFolder, 'dir')
    mkdir(myCacheFolder)
end

%% Call only if simulink is available
if license('test','Simulink')
    disp('Simulink available');
   Simulink.fileGenControl('set', 'CacheFolder', myCacheFolder, 'CodeGenFolder', myCacheFolder); 
else
    disp('Simulink not available');
    % At least add work into the path
    addpath(genpath(fullfile(curdir, 'work')));
end

end