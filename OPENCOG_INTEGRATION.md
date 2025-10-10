# OpenCog AtomSpace Integration with Neural Networks

This document describes the integration of OpenCog AtomSpace with hypergraph neural networks in the nngraph package. This integration enables cognitive architectures that combine symbolic knowledge representation with neural network learning and inference.

## Overview

The OpenCog AtomSpace integration provides:

- **AtomSpace**: Core hypergraph data structure for knowledge representation
- **AtomNode**: Neural network nodes that represent atoms with embeddings and activations
- **HypergraphModule**: Neural network module for processing hypergraph structures
- **AtomTypes**: Standard OpenCog atom types with neural network properties
- **Attention Allocation**: Spreading activation and attention-based processing
- **Pattern Matching**: Query and unification operations in neural networks

## Core Components

### AtomSpace

The `AtomSpace` class manages a hypergraph of atoms (nodes) and links (hyperedges):

```lua
local atomspace = nngraph.AtomSpace()

-- Add atoms
local cat = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "cat")
local animal = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "animal") 

-- Add links
local inheritance = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {cat, animal})

-- Set truth values
atomspace:setTruthValue(cat, 0.9, 0.8)  -- strength=0.9, confidence=0.8

-- Set attention values  
atomspace:setAttentionValue(cat, 100, 50, false)  -- sti=100, lti=50, vlti=false
```

### AtomTypes

Standard OpenCog atom types with neural network properties:

```lua
local AtomTypes = require('nngraph.atom_types')

-- Node types
AtomTypes.CONCEPT_NODE      -- Embeddable concepts (64-dim default)
AtomTypes.PREDICATE_NODE    -- Predicates for relationships (32-dim default)  
AtomTypes.NUMBER_NODE       -- Numeric values (direct encoding)
AtomTypes.VARIABLE_NODE     -- Variables for pattern matching

-- Link types  
AtomTypes.INHERITANCE_LINK  -- A inherits from B
AtomTypes.SIMILARITY_LINK   -- A is similar to B
AtomTypes.EVALUATION_LINK   -- Predicate evaluation
AtomTypes.AND_LINK          -- Logical AND
AtomTypes.OR_LINK           -- Logical OR

-- Type checking utilities
AtomTypes.isNodeType(atomType)          -- Check if atom type is a node
AtomTypes.canEmbed(atomType)            -- Check if type can be embedded
AtomTypes.getDefaultEmbedDim(atomType)  -- Get embedding dimension
```

### AtomNode

Neural network nodes that represent atoms:

```lua
local atomNode = nngraph.AtomNode(atomspace, atomHandle)

-- Get embedding vector
local embedding = atomNode:getEmbedding()

-- Forward propagation
local activation = atomNode:forward(input)

-- Update truth and attention values
atomNode:updateTruthValue(0.8, 0.9)
atomNode:updateAttentionValue()
```

### HypergraphModule

Neural network module for processing atomspace hypergraphs:

```lua
local hgModule = nn.HypergraphModule(atomspace, {
    embeddingDim = 64,
    hiddenDim = 128, 
    outputDim = 32,
    learningRate = 0.01
})

-- Forward pass
local output = hgModule:forward(atomHandle)         -- Single atom
local output = hgModule:forward({atom1, atom2})     -- Multiple atoms
local output = hgModule:forward(pattern)            -- Pattern matching

-- Backward pass for learning
local gradInput = hgModule:backward(input, gradOutput)
```

## Key Features

### 1. Hypergraph Neural Networks

Process knowledge graphs with arbitrary hypergraph structures:

```lua
-- Create knowledge: "Fluffy is a cat, cats are animals" 
local fluffy = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "fluffy")
local cat = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "cat")
local animal = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "animal")

local link1 = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {fluffy, cat})
local link2 = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {cat, animal})

-- Neural network processes the entire hypergraph
local output = hgModule:forward({fluffy, cat, animal, link1, link2})
```

### 2. Attention Allocation

Attention-based processing focusing on relevant atoms:

```lua
-- Set high attention on important concepts
atomspace:setAttentionValue(fluffy, 100, 50, false)

-- Spread activation through the graph
hgModule:spreadActivation(fluffy, iterations = 3)

-- Attention-based forward pass focuses on high-STI atoms
local output = hgModule:forward(all_atoms)  -- Automatically focuses attention
```

### 3. Truth Value Propagation

Learning truth values through neural networks:

```lua
-- Set initial truth values
atomspace:setTruthValue(cat, 0.8, 0.9)
atomspace:setTruthValue(animal, 0.7, 0.8)

-- Neural network propagates and updates truth values
local output = hgModule:forward({cat, animal})
local gradOutput = compute_gradient(output, target)
hgModule:backward({cat, animal}, gradOutput)

-- Truth values are updated based on gradients
local new_strength, new_confidence = atomspace:getTruthValue(cat)
```

### 4. Pattern Matching

Neural pattern matching and query processing:

```lua
-- Find all concept nodes
local pattern1 = {type = AtomTypes.CONCEPT_NODE}
local matches1 = hgModule:matchPattern(pattern1)

-- Find specific inheritance relationships
local pattern2 = {
    type = AtomTypes.INHERITANCE_LINK,
    outgoing = {
        {type = AtomTypes.CONCEPT_NODE, name = "cat"},
        {type = AtomTypes.CONCEPT_NODE}  -- Any concept
    }
}
local matches2 = hgModule:matchPattern(pattern2)
```

## Neural Network Operations

### Embedding Layers

Different atom types use specialized embeddings:

- **ConceptNode**: 64-dimensional dense embeddings (default)
- **PredicateNode**: 32-dimensional embeddings for relationships  
- **NumberNode**: Direct numeric encoding (no embedding layer)
- **VariableNode**: 16-dimensional embeddings for unification

### Activation Functions

Type-specific activation functions:

- **ConceptNode**: `tanh` (concept similarity in [-1,1])
- **PredicateNode**: `sigmoid` (truth values in [0,1])
- **NumberNode**: `linear` (preserve numeric values)
- **LogicalLinks**: Custom logic functions (`min` for AND, `max` for OR)

### Aggregation Functions

Link-specific aggregation of child activations:

- **InheritanceLink**: `min(premise, conclusion)` for logical strength
- **SimilarityLink**: `1 - |a - b|` for similarity measure
- **AndLink**: `min(...)` for logical conjunction
- **OrLink**: `max(...)` for logical disjunction  
- **ListLink**: `concatenate` to preserve order

## Example Usage

```lua
require 'nngraph'

-- Create atomspace
local atomspace = nngraph.AtomSpace()
local AtomTypes = require('nngraph.atom_types')

-- Build knowledge base
local cat = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "cat")
local animal = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "animal")
local inheritance = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {cat, animal})

-- Set truth values
atomspace:setTruthValue(cat, 0.9, 0.8)
atomspace:setTruthValue(animal, 0.7, 0.9)

-- Create hypergraph neural network
local hgModule = nn.HypergraphModule(atomspace)

-- Forward inference
local output = hgModule:forward({cat, animal, inheritance})
print("Network output:", output)

-- Pattern matching
local pattern = {type = AtomTypes.CONCEPT_NODE}
local matches = hgModule:matchPattern(pattern)
print("Found", #matches, "concept nodes")

-- Learning from examples
local target = torch.Tensor({0.85})
local gradOutput = 2 * (output - target)
hgModule:backward({cat, animal, inheritance}, gradOutput)
```

## Integration with Existing nngraph

The OpenCog integration seamlessly extends existing nngraph functionality:

```lua
-- Use AtomNodes in regular nngraph computations
local atomNode1 = nngraph.AtomNode(atomspace, handle1)
local atomNode2 = nngraph.AtomNode(atomspace, handle2)

-- Create neural network graph
local linear = nn.Linear(64, 32)
local output1 = linear(atomNode1)
local output2 = linear(atomNode2)
local combined = nn.CAddTable()({output1, output2})
local final = nn.Tanh()(combined)

-- Build graph module
local gmod = nn.gModule({atomNode1, atomNode2}, {final})

-- Standard nngraph operations work
gmod:forward({atom1_data, atom2_data})
gmod:backward({atom1_data, atom2_data}, grad_output)
```

## Testing

Run the test suite to verify functionality:

```bash
# Run all atomspace tests
lua -e "require('nngraph.test.test_atomspace').runAllTests()"

# Run specific test
lua -e "require('nngraph.test.test_atomspace').testHypergraphModule()"
```

## Examples

See `examples/atomspace_neural_network_example.lua` for a comprehensive demonstration of:

- Creating knowledge bases with concepts and relationships
- Neural network inference over hypergraph structures
- Attention spreading and focus mechanisms
- Pattern matching and query processing
- Learning truth values through backpropagation

## Applications

This integration enables:

- **Cognitive Architectures**: Symbolic-neural hybrid systems
- **Knowledge Graph Neural Networks**: Learning over structured knowledge
- **Probabilistic Logic Networks**: Neural PLN inference
- **Attention-based Reasoning**: Cognitive attention mechanisms
- **Pattern Mining**: Neural pattern recognition in knowledge graphs
- **Transfer Learning**: Knowledge representation for few-shot learning

## Future Extensions

Planned enhancements:

- **PLN Inference Rules**: Neural implementation of PLN rules
- **Temporal Reasoning**: Time-aware atomspace processing
- **Distributed AtomSpace**: Multi-node hypergraph networks
- **Advanced Pattern Matching**: More sophisticated query languages
- **Optimization**: GPU acceleration for large atomspaces

## References

- OpenCog AtomSpace: https://github.com/opencog/atomspace
- Probabilistic Logic Networks: http://wiki.opencog.org/w/PLN
- Hypergraph Neural Networks: Research on graph neural networks
- Cognitive Architectures: Symbolic-neural integration approaches