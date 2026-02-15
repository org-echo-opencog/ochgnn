--[[
Rooted Hypershell Architecture Example

This example demonstrates the rooted hypershell architecture for hierarchical
hypergraph neural network processing. It shows how to organize knowledge in
shell-based structures and perform attention-based inference.

Author: Neural Network Graph Package + OpenCog Integration
License: Same as parent project
]]--

require 'nngraph'
local AtomTypes = require('nngraph.atom_types')
local RootedTree = require('nngraph.rooted_tree')

-- Configuration constants
local MAX_TREE_STR_LENGTH = 200

print("==============================================")
print("Rooted Hypershell Architecture Example")
print("==============================================\n")

-- Create an atomspace with hierarchical knowledge
print("1. Creating hierarchical knowledge base...")
local atomspace = nngraph.AtomSpace()

-- Biological taxonomy example
local fluffy = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "Fluffy")
local cat = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "cat")
local feline = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "feline")
local mammal = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "mammal")
local vertebrate = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "vertebrate")
local animal = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "animal")
local living_thing = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "living_thing")

-- Add properties
local furry = atomspace:addAtom(AtomTypes.PREDICATE_NODE, "furry")
local carnivore = atomspace:addAtom(AtomTypes.PREDICATE_NODE, "carnivore")
local warm_blooded = atomspace:addAtom(AtomTypes.PREDICATE_NODE, "warm_blooded")

-- Create inheritance hierarchy
local link1 = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {fluffy, cat})
local link2 = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {cat, feline})
local link3 = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {feline, mammal})
local link4 = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {mammal, vertebrate})
local link5 = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {vertebrate, animal})
local link6 = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {animal, living_thing})

-- Add property relationships
local prop1 = atomspace:addAtom(AtomTypes.EVALUATION_LINK, nil, {furry, cat})
local prop2 = atomspace:addAtom(AtomTypes.EVALUATION_LINK, nil, {carnivore, feline})
local prop3 = atomspace:addAtom(AtomTypes.EVALUATION_LINK, nil, {warm_blooded, mammal})

-- Set truth values with different confidence levels
atomspace:setTruthValue(fluffy, 1.0, 1.0)      -- Fluffy exists with certainty
atomspace:setTruthValue(cat, 0.95, 0.95)       -- Cat is well-defined
atomspace:setTruthValue(feline, 0.90, 0.90)
atomspace:setTruthValue(mammal, 0.85, 0.95)
atomspace:setTruthValue(vertebrate, 0.80, 0.90)
atomspace:setTruthValue(animal, 0.75, 0.85)

-- Set attention values (importance)
atomspace:setAttentionValue(fluffy, 100, 80, false)
atomspace:setAttentionValue(cat, 80, 70, false)
atomspace:setAttentionValue(mammal, 60, 60, false)

print("Created knowledge base with " .. #atomspace:getAllAtoms() .. " atoms")
print()

-- Demonstrate A000081 sequence (rooted tree enumeration)
print("2. Rooted Tree Enumeration (OEIS A000081)...")
print("Number of unlabeled rooted trees with n nodes:")
local sequence = RootedTree.getA000081Sequence(10)
for i = 0, 10 do
    print(string.format("  n=%2d: %d rooted trees", i, sequence[i + 1]))
end
print()

-- Create rooted hypershell with 'fluffy' as root
print("3. Creating Rooted Hypershell Architecture...")
local rootedHypershell = nn.RootedHypershell(atomspace, fluffy, {
    embeddingDim = 64,
    hiddenDim = 128,
    numShells = 5,
    attenuationFactor = 0.8,
    maxNodesPerShell = 20
})

-- Get and display statistics
local stats = rootedHypershell:getStats()
print("Architecture Statistics:")
print("  Rooted Tree:")
print("    - Node count: " .. stats.tree.nodeCount)
print("    - Max depth: " .. stats.tree.maxDepth)
print("    - Root: " .. stats.tree.rootValue)
print("  Hypershell:")
print("    - Max shell depth: " .. stats.shell.maxDepth)
print("    - Total nodes: " .. stats.shell.totalNodes)
print("    - Average shell size: " .. string.format("%.2f", stats.shell.averageShellSize))
print("  Network:")
print("    - Embedding dimension: " .. stats.embeddingDim)
print("    - Hidden dimension: " .. stats.hiddenDim)
print()

-- Display shell structure
print("4. Shell Organization:")
local hypershell = rootedHypershell:getHypershell()
for depth = 0, hypershell:getMaxDepth() do
    local shell = hypershell:getShell(depth)
    print(string.format("  Shell %d: %d nodes", depth, #shell))
    for i, handle in ipairs(shell) do
        local atom = atomspace:getAtom(handle)
        if atom and atom.name and i <= 3 then  -- Show first 3 nodes
            print(string.format("    - %s (%s)", atom.name, atom.type))
        end
    end
    if #shell > 3 then
        print(string.format("    ... and %d more", #shell - 3))
    end
end
print()

-- Perform forward inference
print("5. Forward Inference Through Rooted Hypershell...")
local output = rootedHypershell:forward(nil)
print("Output shape: " .. output:size(1) .. "-dimensional vector")
print("Output norm: " .. string.format("%.4f", output:norm()))
print()

-- Spread attention through the hierarchy
print("6. Spreading Attention Through Shells...")
print("Initial attention values:")
local sti_fluffy_before, _, _ = atomspace:getAttentionValue(fluffy)
local sti_cat_before, _, _ = atomspace:getAttentionValue(cat)
local sti_mammal_before, _, _ = atomspace:getAttentionValue(mammal)
print(string.format("  Fluffy (root):  STI = %.2f", sti_fluffy_before))
print(string.format("  Cat (shell 1):  STI = %.2f", sti_cat_before))
print(string.format("  Mammal (shell 2): STI = %.2f", sti_mammal_before))

rootedHypershell:spreadAttention(3)

local sti_fluffy_after, _, _ = atomspace:getAttentionValue(fluffy)
local sti_cat_after, _, _ = atomspace:getAttentionValue(cat)
local sti_mammal_after, _, _ = atomspace:getAttentionValue(mammal)
print("\nAfter attention spreading:")
print(string.format("  Fluffy (root):  STI = %.2f", sti_fluffy_after))
print(string.format("  Cat (shell 1):  STI = %.2f", sti_cat_after))
print(string.format("  Mammal (shell 2): STI = %.2f", sti_mammal_after))
print()

-- Hierarchical query processing
print("7. Hierarchical Query Processing...")
local query = torch.randn(64):abs()  -- Random query vector
local results = rootedHypershell:hierarchicalInference(query)
print("Query results by shell:")
for shellIdx = 0, math.min(2, hypershell:getMaxDepth()) do
    local shellResults = results[shellIdx] or {}
    print(string.format("  Shell %d: %d results", shellIdx, #shellResults))
    for i = 1, math.min(3, #shellResults) do
        local result = shellResults[i]
        local atom = atomspace:getAtom(result.handle)
        if atom and atom.name then
            print(string.format("    - %s (similarity: %.4f)", 
                  atom.name, result.similarity))
        end
    end
end
print()

-- Get most relevant nodes
print("8. Attention-Weighted Relevance Ranking...")
local relevantNodes = rootedHypershell:getRelevantNodes(query, 5)
print("Top 5 most relevant nodes:")
for i, result in ipairs(relevantNodes) do
    local atom = atomspace:getAtom(result.handle)
    if atom then
        print(string.format("  %d. %s (relevance: %.4f)", 
              i, atom.name or tostring(result.handle), result.relevance))
    end
end
print()

-- Display the tree structure
print("9. Rooted Tree Structure (Parenthesis Notation):")
local tree = rootedHypershell:getTree()
local treeStr = tree:toString()
-- Limit output length for readability
if #treeStr > MAX_TREE_STR_LENGTH then
    print(treeStr:sub(1, MAX_TREE_STR_LENGTH) .. "...")
else
    print(treeStr)
end
print()

-- Demonstrate gradient computation
print("10. Neural Network Learning...")
local target = torch.ones(64) * 0.5
local criterion = nn.MSECriterion()
local loss = criterion:forward(output, target)
print(string.format("Loss: %.4f", loss))

local gradOutput = criterion:backward(output, target)
rootedHypershell:backward(nil, gradOutput)
print("Gradients computed through all shells")
print()

-- Summary
print("==============================================")
print("Example Summary")
print("==============================================")
print("The rooted hypershell architecture provides:")
print("  ✓ Hierarchical organization of hypergraph knowledge")
print("  ✓ Shell-based attention spreading")
print("  ✓ Multi-scale inference through tree structure")
print("  ✓ Attention-weighted relevance ranking")
print("  ✓ Neural network learning with backpropagation")
print("  ✓ Integration with OEIS A000081 rooted tree enumeration")
print()
print("This architecture enables cognitive processing at")
print("multiple levels of abstraction, from specific instances")
print("(Fluffy) to general categories (living_thing).")
print()
