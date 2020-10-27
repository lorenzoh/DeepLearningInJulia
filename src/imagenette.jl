# Let's import everything we'll need:
# {cell=a}

using DataLoaders
using DataAugmentation
using DeepLearningTasks
using DeepLearningTasks: encodeinput, encodetarget
using DLDatasets
using MLDataPattern
using Core


# {tests=`Int_`}
# ## Loading data
# [`Int`](#)
# Let's next load the ImageNette datasets using `DLDatasets.jl`:
# {cell=a tests=`Int_`}

ds = DLDatasets.loaddataset(ImageNette, "v2_160px")


# We can load an observation with `getobs`:
# {cell=a}

image, label = getobs(ds, 1)
@show label
image


# ## Data pipeline
# Here is some code I want to execute:
#
# {cell=a}

nclasses = 10
task = ImageClassification(nclasses, sz = (160, 160))


# Let's transform the image:
# {cell=a}

x = encodeinput(task, image)
y = encodetarget(task, 1)
summary.((x, y))
