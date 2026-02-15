# Rooted Hypershell Architecture

The Rooted Hypershell Architecture is a novel hierarchical neural network structure for processing knowledge graphs in the OpenCog AtomSpace. It combines rooted tree structures with shell-based hypergraph organization to enable multi-scale cognitive processing.

## Overview

The architecture consists of three main components:

1. **RootedTree**: Tree data structures with support for OEIS A000081 rooted tree enumeration
2. **Hypershell**: Shell-based organization of hypergraph nodes around a root node
3. **RootedHypershell**: Neural network integration combining trees and shells for hierarchical inference

## Key Features

### 1. Rooted Tree Structure (OEIS A000081)

The `RootedTree` module implements rooted tree data structures and includes calculation of the OEIS A000081 sequence, which counts the number of unlabeled rooted trees with n nodes.

```lua
local RootedTree = require('nngraph.rooted_tree')

-- Create a rooted tree
local tree = RootedTree.new("root")
local child = tree:addChild(tree:getRoot(), "child")
local grandchild = tree:addChild(child, "grandchild")

-- Get tree statistics
print("Nodes:", tree:getNodeCount())       -- 3
print("Max depth:", tree:getMaxDepth())    -- 2

-- Calculate A000081 sequence
local sequence = RootedTree.getA000081Sequence(10)
-- sequence = {0, 1, 1, 2, 4, 9, 20, 48, 115, 286, 719}
```

#### Tree Operations

- `RootedTree.new(rootValue)` - Create a new rooted tree
- `tree:addChild(parentNode, childValue)` - Add a child to a node
- `tree:getNodesAtDepth(depth)` - Get all nodes at a specific depth
- `tree:traverseDFS(node, callback)` - Depth-first traversal
- `tree:traverseBFS(callback)` - Breadth-first traversal
- `tree:getLeaves()` - Get all leaf nodes
- `tree:toString()` - Convert to parenthesis notation

#### A000081 Sequence

The module includes implementation of the OEIS A000081 sequence formula:

```
a(n) = (1/(n-1)) * Sum_{k=1..n-1} (Sum_{d|k} d*a(d)) * a(n-k)
```

This sequence counts unlabeled rooted trees and is fundamental to understanding hierarchical structures in knowledge graphs.

### 2. Hypershell Organization

The `Hypershell` module organizes hypergraph nodes into concentric shells around a root node, similar to breadth-first search layers.

```lua
local Hypershell = require('nngraph.hypershell')

-- Create hypershell structure
local atomspace = nngraph.AtomSpace()
local rootHandle = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "root")
local hypershell = Hypershell.new(atomspace, rootHandle)

-- Access shells
local shell0 = hypershell:getShell(0)  -- Root
local shell1 = hypershell:getShell(1)  -- Immediate neighbors
local shell2 = hypershell:getShell(2)  -- Second-order neighbors
```

#### Shell Operations

- `Hypershell.new(atomspace, rootHandle, options)` - Create shell structure
- `hypershell:getShell(depth)` - Get nodes at a specific shell depth
- `hypershell:getNodeDepth(handle)` - Get the depth of a node
- `hypershell:getMaxDepth()` - Get maximum shell depth
- `hypershell:propagateOutward(values, function)` - Propagate values outward
- `hypershell:propagateInward(values, function)` - Propagate values inward
- `hypershell:spreadAttention(attenuation)` - Spread attention through shells

#### Shell-Based Propagation

```lua
-- Outward propagation from root
local values = hypershell:propagateOutward(100, function(value, from, to, depth)
    return value * 0.8  -- Attenuate by 20% per shell
end)

-- Inward propagation toward root
local leafValues = {[leafHandle] = 1.0}
local aggregated = hypershell:propagateInward(leafValues, function(value, from, to, depth)
    return value + 0.1 * depth  -- Accumulate with depth weighting
end)
```

### 3. Rooted Hypershell Neural Network

The `RootedHypershell` class integrates tree and shell structures with neural network processing.

```lua
-- Create rooted hypershell
local rhs = nn.RootedHypershell(atomspace, rootHandle, {
    embeddingDim = 64,
    hiddenDim = 128,
    numShells = 5,
    attenuationFactor = 0.8,
    maxNodesPerShell = 50
})

-- Forward propagation
local output = rhs:forward(nil)

-- Backward propagation
local gradOutput = criterion:backward(output, target)
rhs:backward(nil, gradOutput)
```

#### Neural Network Architecture

Each shell has dedicated neural network components:

- **Shell Processors**: Transform embeddings within each shell
- **Shell Attention**: Attention mechanisms for importance weighting
- **Shell Aggregation**: Combine activations within shells

The architecture processes information hierarchically:

1. **Embedding Layer**: Convert atoms to neural representations
2. **Shell Processing**: Process each shell with specialized networks
3. **Attention Weighting**: Apply attention to focus on relevant nodes
4. **Shell Aggregation**: Combine information across shells
5. **Output Generation**: Produce final representation

#### Key Methods

- `rhs:forward(input)` - Forward propagation through all shells
- `rhs:backward(input, gradOutput)` - Backward propagation
- `rhs:spreadAttention(iterations)` - Spread attention through shells
- `rhs:hierarchicalInference(query)` - Multi-level query processing
- `rhs:getRelevantNodes(query, topK)` - Attention-weighted node ranking
- `rhs:getStats()` - Architecture statistics

## Architecture Design

### Hierarchical Processing

```
Root (Shell 0)
    ├── Shell 1 (immediate neighbors)
    │   ├── Shell 2 (second-order)
    │   │   └── Shell 3 (third-order)
    │   └── Shell 2 (second-order)
    └── Shell 1 (immediate neighbors)
        └── Shell 2 (second-order)
```

### Neural Information Flow

```
Input Atoms → Embeddings → Shell 0 Processing → Shell 1 Processing → ... 
    → Shell N Processing → Aggregation → Output
    
With attention spreading:
Root STI=100 → Shell 1 STI=80 → Shell 2 STI=64 → Shell 3 STI=51.2
```

## Use Cases

### 1. Knowledge Graph Reasoning

Organize taxonomic or ontological knowledge hierarchically:

```lua
-- Biological taxonomy: Fluffy → Cat → Feline → Mammal → Animal
local fluffy = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "Fluffy")
local cat = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "cat")
atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {fluffy, cat})

local rhs = nn.RootedHypershell(atomspace, fluffy)
rhs:forward(nil)  -- Process entire hierarchy
```

### 2. Multi-Scale Attention

Focus cognitive resources on relevant parts of the knowledge graph:

```lua
-- Set high attention on root concept
atomspace:setAttentionValue(rootHandle, 100, 50, false)

-- Spread attention through the hierarchy
rhs:spreadAttention(3)

-- Children now have attention proportional to distance
```

### 3. Hierarchical Query Processing

Process queries at multiple levels of abstraction:

```lua
local query = torch.randn(64)
local results = rhs:hierarchicalInference(query)

-- results[0] = matches at root level
-- results[1] = matches at first shell
-- results[2] = matches at second shell, etc.
```

### 4. Transfer Learning

Use tree structure for knowledge transfer:

```lua
-- Train on specific instance
local rhs_specific = nn.RootedHypershell(atomspace, specificInstance)
rhs_specific:forward(nil)

-- Transfer to general category by re-rooting
local rhs_general = nn.RootedHypershell(atomspace, generalCategory)
-- Embeddings are shared across architectures
```

## Implementation Details

### A000081 Sequence Calculation

The rooted tree enumeration uses the recurrence relation:

```lua
a(n) = (1/(n-1)) * Sum_{k=1..n-1} (Sum_{d|k} d*a(d)) * a(n-k)

with a(0) = 0, a(1) = 1
```

This gives the sequence: 0, 1, 1, 2, 4, 9, 20, 48, 115, 286, 719, ...

### Shell Construction Algorithm

1. Initialize shell 0 with root node
2. For each shell depth up to maxShells:
   - For each node in current shell:
     - Find all neighbors in hypergraph
     - Add unvisited neighbors to next shell
   - If next shell is empty, stop
3. Record depth and parent for each node

### Attention Spreading

Attention spreads from root to leaves with attenuation:

```
STI(child) = STI(parent) * attenuation_factor^depth
```

Default attenuation factor is 0.8, meaning each shell receives 80% of the previous shell's attention.

## Testing

Run the test suite:

```bash
lua -e "require('nngraph.test.test_rooted_hypershell').runAllTests()"
```

Run a specific test:

```bash
lua -e "require('nngraph.test.test_rooted_hypershell').testRootedHypershell()"
```

## Examples

See `examples/rooted_hypershell_example.lua` for a comprehensive demonstration including:

- Creating hierarchical knowledge bases
- Building rooted hypershell architectures
- Shell-based organization
- Attention spreading
- Hierarchical inference
- Neural network training

Run the example:

```bash
lua examples/rooted_hypershell_example.lua
```

## Performance Considerations

### Memory Usage

- Embeddings: O(N × embedding_dim) where N is number of atoms
- Shell structure: O(N) for node-to-depth mapping
- Neural networks: O(num_shells × hidden_dim²)

### Computational Complexity

- Shell construction: O(N × avg_degree)
- Forward pass: O(num_shells × nodes_per_shell × hidden_dim²)
- Backward pass: O(num_shells × nodes_per_shell × hidden_dim²)
- Attention spreading: O(N) per iteration

### Optimization Tips

1. Limit `maxNodesPerShell` for large graphs
2. Use smaller `numShells` for shallow hierarchies
3. Reduce `hiddenDim` for faster processing
4. Cache embeddings between forward passes

## Theoretical Foundation

The rooted hypershell architecture is inspired by:

1. **OEIS A000081**: Mathematical enumeration of rooted trees
2. **Graph Neural Networks**: Message passing on graph structures
3. **Attention Mechanisms**: Cognitive focus on relevant information
4. **OpenCog**: Hypergraph knowledge representation
5. **Hierarchical Reinforcement Learning**: Multi-scale decision making

## Future Extensions

Potential enhancements:

- **Dynamic Shell Resizing**: Adaptive shell boundaries based on attention
- **Multi-Root Architectures**: Multiple focal points for parallel processing
- **Temporal Shells**: Time-based shell organization
- **Probabilistic Shells**: Uncertainty in shell membership
- **Graph Convolutional Layers**: More sophisticated aggregation functions
- **Attention Flow Optimization**: Learn optimal attention spreading patterns

## References

- OEIS A000081: https://oeis.org/A000081
- OpenCog AtomSpace: https://github.com/opencog/atomspace
- Graph Neural Networks: Scarselli et al., 2009
- Attention Mechanisms: Bahdanau et al., 2015
- Rooted Trees: Knuth, The Art of Computer Programming, Vol. 1

## License

Same as parent project (nngraph).
