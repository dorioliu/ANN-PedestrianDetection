function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

print('=====> Importing the necessary packages')

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nn'
require 'cutorch'
require 'image'

print('======> Loading the model')
model = torch.load('/home/santhosh/Projects/ANNCourse/ANN/Project/results/Train1_1_3_1000/model930.net')
print('======> Loaded the model. The model is shown below\n')
print(model)
model:evaluate()

file1 = io.open('/home/santhosh/Projects/ANNCourse/ANN/Project/results/Train1_1_3_1000/MeanData.txt', "r")
file2 = io.open('/home/santhosh/Projects/ANNCourse/ANN/Project/results/Train1_1_3_1000/StdData.txt', "r")

meanImg = {}
stdImg = {} 

file3 = io.open('/home/santhosh/Projects/ANNCourse/ANN/Project/shufListOfImages.txt', "r")
for i=1,3 do
  meanImg[i] = file1:read()
  stdImg[i] = file2:read()
end

count1 = 0 
while true do
	local line = file3:read()
	count1 = count1 + 1
	if line == nil then break end
	local l = line.split(line, " ")
	local imageName = l[1]

	img = image.load(imageName)
	display = image.toDisplayTensor(img)
	print(string.format('======> Preprocessing image: %d', count1))
	img = img:float()
	image.display(img)
	
	for i=1,3 do
	  img[{{i},{},{}}]:add(-meanImg[i])
	  img[{{i},{},{}}]:div(stdImg[i])
	end

	print('======> Predicting')
	model:float()
	pred = model:forward(img)
	val, idx = torch.max(pred, 1)
	--print(pred[1])
	print(string.format('======> Predicted Answer: %d', idx[1]))

	io.stdin:read'*l'

end

file1:close()
file2:close()
