--[[
OpenCog AtomSpace Neural Network Example

This example demonstrates how to use the OpenCog atomspace integration
with neural networks for knowledge representation and reasoning.

The example creates a simple knowledge base about animals and their
properties, then uses hypergraph neural networks to perform inference
and learning over the knowledge representation.

Author: Neural Network Graph Package + OpenCog Integration
License: Same as parent project
]]--

require 'nngraph'

print("=" .. string.rep("=", 60))
print("OpenCog AtomSpace Neural Network Integration Example")
print("=" .. string.rep("=", 60))

-- Create atomspace
local atomspace = nngraph.AtomSpace()
local AtomTypes = require('nngraph.atom_types')

print("\n1. Creating Knowledge Base...")
print("-" .. string.rep("-", 30))

-- Create concept nodes for animals
local cat = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "cat")
local dog = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "dog") 
local bird = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "bird")
local animal = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "animal")
local mammal = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "mammal")

-- Create predicate nodes for properties
local has_fur = atomspace:addAtom(AtomTypes.PREDICATE_NODE, "has_fur")
local can_fly = atomspace:addAtom(AtomTypes.PREDICATE_NODE, "can_fly")
local makes_sound = atomspace:addAtom(AtomTypes.PREDICATE_NODE, "makes_sound")

-- Create some specific instances
local fluffy = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "fluffy")  -- A specific cat
local rover = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "rover")    -- A specific dog

print("Created atoms:")
print(string.format("  - %d concept nodes", #atomspace:getAtomsByType(AtomTypes.CONCEPT_NODE)))
print(string.format("  - %d predicate nodes", #atomspace:getAtomsByType(AtomTypes.PREDICATE_NODE)))

-- Create knowledge relationships using inheritance links
local inheritance_links = {
    {cat, mammal},      -- Cat inherits from Mammal
    {dog, mammal},      -- Dog inherits from Mammal  
    {mammal, animal},   -- Mammal inherits from Animal
    {bird, animal},     -- Bird inherits from Animal
    {fluffy, cat},      -- Fluffy is a Cat
    {rover, dog}        -- Rover is a Dog
}

print("\n2. Creating Inheritance Relationships...")
print("-" .. string.rep("-", 40))

for i, link in ipairs(inheritance_links) do
    local inheritance = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, link)
    -- Set truth values for inheritance relationships
    atomspace:setTruthValue(inheritance, 0.9, 0.8)
    print(string.format("  - %s inherits from %s", 
          atomspace:getAtom(link[1]).name, atomspace:getAtom(link[2]).name))
end

-- Create evaluation links for properties
print("\n3. Creating Property Relationships...")
print("-" .. string.rep("-", 40))

local evaluations = {
    {has_fur, {cat}},       -- Cats have fur
    {has_fur, {dog}},       -- Dogs have fur
    {can_fly, {bird}},      -- Birds can fly
    {makes_sound, {cat}},   -- Cats make sound
    {makes_sound, {dog}},   -- Dogs make sound
    {makes_sound, {bird}}   -- Birds make sound
}

for i, eval in ipairs(evaluations) do
    local evaluation = atomspace:addAtom(AtomTypes.EVALUATION_LINK, nil, {eval[1], eval[2][1]})
    atomspace:setTruthValue(evaluation, 0.8, 0.7)
    local predName = atomspace:getAtom(eval[1]).name
    local subjName = atomspace:getAtom(eval[2][1]).name
    print(string.format("  - %s: %s", subjName, predName))
end

-- Set initial truth values for concepts based on specificity
atomspace:setTruthValue(animal, 0.5, 0.9)    -- Very general concept
atomspace:setTruthValue(mammal, 0.6, 0.8)    -- More specific
atomspace:setTruthValue(cat, 0.8, 0.8)       -- Specific concept
atomspace:setTruthValue(dog, 0.8, 0.8)       -- Specific concept
atomspace:setTruthValue(bird, 0.7, 0.8)      -- Specific concept
atomspace:setTruthValue(fluffy, 0.9, 0.9)    -- Very specific instance
atomspace:setTruthValue(rover, 0.9, 0.9)     -- Very specific instance

print("\n4. Atomspace Statistics...")
print("-" .. string.rep("-", 25))
local stats = atomspace:getStatistics()
print(string.format("  - Total atoms: %d", stats.totalAtoms))
print(string.format("  - Total links: %d", stats.totalLinks))
for atomType, count in pairs(stats.typeCount) do
    print(string.format("  - %s: %d", atomType, count))
end

-- Create hypergraph neural network module
print("\n5. Creating Hypergraph Neural Network...")
print("-" .. string.rep("-", 42))

local hgModule = nn.HypergraphModule(atomspace, {
    embeddingDim = 64,
    hiddenDim = 128,
    outputDim = 32,
    learningRate = 0.01,
    attentionThreshold = 30,
    maxAttentionalFocus = 8
})

print("  - Embedding dimension: 64")
print("  - Hidden dimension: 128") 
print("  - Output dimension: 32")
print("  - Learning rate: 0.01")

-- Test forward propagation
print("\n6. Testing Neural Network Forward Pass...")
print("-" .. string.rep("-", 42))

-- Test single atom inference
print("  Single atom activations:")
local atoms_to_test = {fluffy, rover, cat, dog, mammal, animal}
for _, atomHandle in ipairs(atoms_to_test) do
    local atom = atomspace:getAtom(atomHandle)
    local output = hgModule:forward(atomHandle)
    local strength, confidence = atomspace:getTruthValue(atomHandle)
    print(string.format("    - %s: activation=%.4f, tv=(%.2f,%.2f)", 
          atom.name, output[1], strength or 0, confidence or 0))
end

-- Test multi-atom inference
print("\n  Multi-atom inference:")
local animal_group = {cat, dog, bird}
local output = hgModule:forward(animal_group)
print(string.format("    - Animal group activation: %s", 
      table.concat(output:totable(), ", ")))

-- Pattern matching example
print("\n7. Testing Pattern Matching...")
print("-" .. string.rep("-", 31))

-- Find all concept nodes
local pattern1 = {type = AtomTypes.CONCEPT_NODE}
local matches1 = hgModule:matchPattern(pattern1)
print(string.format("  - Found %d concept nodes", #matches1))

-- Find specific named concept
local pattern2 = {type = AtomTypes.CONCEPT_NODE, name = "cat"}
local matches2 = hgModule:matchPattern(pattern2)
print(string.format("  - Found %d atoms named 'cat'", #matches2))

-- Test attention spreading
print("\n8. Testing Attention Spreading...")
print("-" .. string.rep("-", 33))

-- Set high attention on fluffy
atomspace:setAttentionValue(fluffy, 100, 50, false)
print(string.format("  - Set high attention on '%s' (STI=100)", 
      atomspace:getAtom(fluffy).name))

-- Spread activation
hgModule:spreadActivation(fluffy, 2)
print("  - Spread activation for 2 iterations")

-- Check attention values after spreading
print("  - Attention values after spreading:")
for _, atomHandle in ipairs({fluffy, cat, mammal, animal}) do
    local atom = atomspace:getAtom(atomHandle)
    local sti, lti = atomspace:getAttentionValue(atomHandle)
    print(string.format("    - %s: STI=%d, LTI=%d", 
          atom.name, sti or 0, lti or 0))
end

-- Demonstrate learning through backpropagation
print("\n9. Demonstrating Learning...")
print("-" .. string.rep("-", 27))

-- Create target outputs for supervised learning
local learning_pairs = {
    {fluffy, torch.Tensor({0.95})},   -- Fluffy should have high activation
    {rover, torch.Tensor({0.90})},    -- Rover should have high activation  
    {animal, torch.Tensor({0.50})}    -- Animal should have medium activation
}

print("  Learning from examples:")
for epoch = 1, 3 do
    local total_loss = 0
    
    for i, pair in ipairs(learning_pairs) do
        local atomHandle, target = pair[1], pair[2]
        local atom = atomspace:getAtom(atomHandle)
        
        -- Forward pass
        local output = hgModule:forward(atomHandle)
        
        -- Compute loss (MSE)
        local loss = torch.pow(output - target, 2):mean()
        total_loss = total_loss + loss
        
        -- Backward pass
        local gradOutput = 2 * (output - target) / output:nElement()
        hgModule:backward(atomHandle, gradOutput)
    end
    
    print(string.format("    - Epoch %d: Loss = %.6f", epoch, total_loss / #learning_pairs))
end

-- Test learned representations
print("\n  Final activations after learning:")
for _, pair in ipairs(learning_pairs) do
    local atomHandle, target = pair[1], pair[2]
    local atom = atomspace:getAtom(atomHandle)
    local output = hgModule:forward(atomHandle)
    print(string.format("    - %s: output=%.4f, target=%.2f", 
          atom.name, output[1], target[1]))
end

-- Summary
print("\n10. Summary...")
print("-" .. string.rep("-", 11))
print("  ✓ Created atomspace with concepts and relationships")
print("  ✓ Integrated with hypergraph neural networks")
print("  ✓ Demonstrated forward propagation through hypergraph")
print("  ✓ Tested pattern matching and query capabilities")
print("  ✓ Showed attention spreading mechanism")
print("  ✓ Performed learning through backpropagation")

print("\n" .. "=" .. string.rep("=", 60))
print("Example completed successfully!")
print("The OpenCog atomspace is now integrated with neural networks")
print("and ready for advanced cognitive architectures and reasoning.")
print("=" .. string.rep("=", 60))

-- Clean up
atomspace:clear()
hgModule:clearCache()

print("\nAtomspace cleared. Example finished.")