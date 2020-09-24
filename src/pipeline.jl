
# ![](../graphs/training.dot.svg)
# {cell=a}
using GraphPlot
using LightGraphs
using ParserCombinator
using GraphIO

g = loadgraph("DeepLearningInJulia/publish/simple.dot", "graphname", GraphIO.DOT.DOTFormat())

gplot(g, nodelabel=1:4)


#

input, target = getobs(dataset, 1)
x, y = encode(input, target)

xs, ys = batch([(x1, y2), ...])

ŷs = model(xs)
loss = lossfn(ŷs, ys)
grads = ...

update!(optim, grads, params(model))

ŷ
