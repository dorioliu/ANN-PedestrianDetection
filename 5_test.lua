----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()
   local e1 = 0
   local e2 = 0
   local e3 = 0
   local e4 = 0
   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
	  gen1, gen2 = torch.max(pred, 1)
      confusion:add(pred, target)
	  --[[
	  if gen2 == target then
		  if target == "1" then
			  e1 = e1 + 1
		  else
			  e4 = e4 + 1
		  end
	  else
		  if target == "2" then
			  e2 = e2 + 1
		  else
			  e3 = e3 + 1
		  end
	  end
	  --]]
	  if target == 1 then
		  e1 = e1 + 1
	  elseif target == 2 then
		  e2 = e2 + 1
	  else 
		  e3 = e3 + 1
		  print(target .. " " .. type(target))
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   print(" " .. e1 .. " " .. e2 .. "\n " .. e3 .. " " .. e4 )
   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()
end
