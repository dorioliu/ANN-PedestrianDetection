function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

if tablelength(arg) < 2 then
  print("Error! The format is \"qlua testing.lua imagePath\"")
  print(arg[1])
  os.exit()
end

print('=====> Importing the necessary packages')

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'nn'
require 'cutorch'
require 'image'

print('======> Loading the model')
model = torch.load('/home/santhosh/Projects/ANNCourse/ANN/Project/results/train_1_5_2300/model.net')
print('======> Loaded the model. The model is shown below\n')
print(model)
model:evaluate()

file1 = io.open('/home/santhosh/Projects/ANNCourse/ANN/Project/results/train_1_5_2300/MeanData.txt', "r")
file2 = io.open('/home/santhosh/Projects/ANNCourse/ANN/Project/results/train_1_5_2300/StdData.txt', "r")

meanImg = {}
stdImg = {} 

print('\n======> Reading image')
img = image.load(arg[1])
display = image.toDisplayTensor(img)
image.display(display)

print('======> Preprocessing image')
--img = image.rgb2yuv(img)
img = img:float()

for i=1,3 do
  meanImg[i] = file1:read()
  stdImg[i] = file2:read()
  img[{{i},{},{}}]:add(-meanImg[i])
  img[{{i},{},{}}]:div(stdImg[i])
end

print('======> Predicting')
model:float()
pred = model:forward(img)
val, idx = torch.max(pred, 1)
print(pred)
print(string.format('======> Predicted Answer: %d', idx[1]))

image.display(model:get(1).output)
image.display(model:get(4).output)
torch.save('Filter1', model:get(1).output)
torch.save('Filter4', model:get(2).output) 





