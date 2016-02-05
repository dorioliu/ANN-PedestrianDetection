----------------------------------------------------------------------
-- This script demonstrates how to load the (SVHN) House Numbers 
-- training data, and pre-process it to facilitate learning.
--
-- The SVHN is a typical example of supervised training dataset.
-- The problem to solve is a 10-class classification problem, similar
-- to the quite known MNIST challenge.
--
-- It's a good idea to run this script with the interactive mode:
-- $ th -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('CALTECH Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'small', 'how many samples do we load: small | full | extra')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
train_file = '/home/santhosh/Projects/ANNCourse/DeeperLookAtPedDetection/shufImageNamesTrain3-64_torch.txt'
test_file = '/home/santhosh/Projects/ANNCourse/DeeperLookAtPedDetection/shufImageNamesTest-64_torch.txt'

io.input(train_file)

local count = 0
ones1 = 0
twos1 = 0
while true do
	local line = io.read()
	if line == nil then break end
	local l = line.split(line, " ")
	local fileName = l[1]
	local label = tonumber(l[2])
	if label == 1 then
		ones1 = ones1 + 1
	else
		twos1 = twos1 + 1
	end

	if count == 1 then
		local img = image.load(fileName)
		dims = img:size()
	end
	
	count = count + 1
end

trsize = count

io.input(test_file)

local count = 0
local ones2 = 0
local twos2 = 0

while true do
	local line = io.read()
	if line == nil then break end
	local l = line.split(line, " ")
	local fileName = l[1]
	local label = tonumber(l[2])
	if label == 1 then
		ones2 = ones2 + 1
	else
		twos2 = twos2 + 1
	end


	
	count = count + 1
end

--print(ones1)
--print(twos1)
--print(ones2)
--print(twos2)
tesize = count

---------------------------------------------------------------------
print '==> loading dataset'

-- We load the dataset from disk, and re-arrange it to be compatible
-- with Torch's representation. Matlab uses a column-major representation,
-- Torch is row-major, so we just have to transpose the data.

-- Note: the data, in X, is 4-d: the 1st dim indexes the samples, the 2nd
-- dim indexes the color channels (RGB), and the last two dims index the
-- height and width of the samples.

trainData = {
   data = torch.Tensor(trsize, dims[1], dims[2], dims[3]),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}

-- If extra data is used, we load the extra file, and then
-- concatenate the two training sets.

-- Torch's slicing syntax can be a little bit frightening. I've
-- provided a little tutorial on this, in this same directory:
-- A_slicing.lua

-- Finally we load the test data.

testData = {
   data = torch.Tensor(tesize, dims[1], dims[2], dims[3]),
   labels = torch.Tensor(tesize),
   size = function() return tesize end
}

print('==> Loading Training and Testing data')

io.input(train_file)

local iter = 1
while true do
	local line = io.read()
	if line == nil then break end
	local l = line.split(line, " ")
	local fileName = l[1]
	local label = tonumber(l[2])
	--print(iter)
	local img = image.load(fileName)
	--print('Loaded ' .. fileName)
	trainData.data[{iter, {}, {}, {}}] = img
	--image.display(trainData.data[{iter,{},{},{}}])
	trainData.labels[{iter}] = label
	iter = iter + 1
end

io.input(test_file)

iter = 1
while true do
	local line = io.read()
	if line == nil then break end
	local l = line.split(line, " ")
	local fileName = l[1]
	local label = tonumber(l[2])
	--print(iter)
	local img = image.load(fileName)
	--print('Loaded ' .. fileName)
	testData.data[{iter, {}, {}, {}}] = img
	testData.labels[{iter}] = label
	iter = iter + 1
end

print(testData.labels)
----------------------------------------------------------------------
print '==> Preprocessing data'

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch, 
-- in general by doing: dst = src:type('torch.TypeTensor'), 
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):

trainData.data = trainData.data:float()
testData.data = testData.data:float()

-- We now preprocess the data. Preprocessing is crucial
-- when applying pretty much any kind of machine learning algorithm.

-- For natural images, we use several intuitive tricks:
--   + images are mapped into YUV space, to separate luminance information
--     from color information
--   + the luminance channel (Y) is locally normalized, using a contrastive
--     normalization operator: for each neighborhood, defined by a Gaussian
--     kernel, the mean is suppressed, and the standard deviation is normalized
--     to one.
--   + color channels are normalized globally, across the entire dataset;
--     as a result, each color component has 0-mean and 1-norm across the dataset.

-- Convert all images to YUV
print '==> preprocessing data: colorspace RGB -> YUV'
--[[
for i = 1,trainData:size() do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
for i = 1,testData:size() do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end
--]]
-- Name channels for convenience
channels = {'r','g','b'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   print(mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
   print(std[i])
end

print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

local file1 = io.open("MeanData.txt", "w")
local file2 = io.open("StdData.txt", "w")


for i,channel in ipairs(channels) do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()
   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()
   file1:write(string.format("Channel %s\n", channel))
   file2:write(string.format("Channel %s\n", channel))

   file1:write(mean[i])
   file2:write(std[i])
   
   file1:write("\n\n")
   file2:write("\n\n")

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end


----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using itorch.image().

if opt.visualize then
   first256Samples_y = trainData.data[{ {1,256},1 }]
   first256Samples_u = trainData.data[{ {1,256},2 }]
   first256Samples_v = trainData.data[{ {1,256},3 }]

   if itorch then      
      itorch.image(first256Samples_y)
      itorch.image(first256Samples_u)
      itorch.image(first256Samples_v)
   else
      image.display(first256Samples_y)
      image.display(first256Samples_u)
      image.display(first256Samples_v)
      -- print("For visualization, run this script in an itorch notebook")
   end
end

--]]
