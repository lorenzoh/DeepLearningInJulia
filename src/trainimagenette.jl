using DataLoaders
using DLPipelines
using DLDatasets
using Flux
using FluxModels
using FluxTraining
using MLDataPattern

# Define the learning task, see https://github.com/lorenzoh/DLPipelines.jl/blob/master/src/tasks/imageclassification.jl
# for the implementation.

task = ImageClassification(10, sz = (224, 224))

## Load the dataset

labeltoclass = metadata(ImageNette).labeltoclass
obsfn((image, label)) = (image, labeltoclass[label])
trainds_, valds_ = DLDatasets.loaddataset(ImageNette, "v2_160px", split = ("train", "val"))

# map data preprocessing for `task` over datasets
trainds = shuffleobs(taskdataset(task, trainds_; obsfn))
valds = taskdataset(task, valds_; obsfn, valid = true)

## DataLoaders

bs = 64
traindl = DataLoader(trainds, bs, partial = false)
valdl = DataLoader(valds, 2bs, partial = false)

model = gpu(Chain(xresnet18(), FluxModels.classificationhead(task.nclasses, 512)))

# use one-cycle schedule
schedule = onecycleschedule(10 * length(traindl), 0.01)

learner = Learner(
    model,
    (traindl, valdl),
    ADAM(),
    Flux.Losses.logitcrossentropy,
    callbacks = [ToGPU()],
    metrics = [Metric(accuracy)],
    schedule = Schedules(schedule)
)

# train for 10 epochs
FluxTraining.fit!(learner, 10)
