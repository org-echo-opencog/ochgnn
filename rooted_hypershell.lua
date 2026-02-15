--[[
Rooted Hypershell Architecture

This module combines rooted tree structures with hypershell organization to create
a hierarchical neural network architecture for processing hypergraph knowledge.
It enables multi-scale reasoning with attention spreading through tree-like shells.

Author: Neural Network Graph Package + OpenCog Integration
License: Same as parent project
]]--

local RootedTree = require('nngraph.rooted_tree')
local Hypershell = require('nngraph.hypershell')
local AtomTypes = require('nngraph.atom_types')
require 'nn'

local RootedHypershell, parent = torch.class('nn.RootedHypershell', 'nn.Module')

function RootedHypershell:__init(atomspace, rootHandle, options)
    parent.__init(self)
    
    self.atomspace = atomspace
    self.rootHandle = rootHandle
    self.options = options or {}
    
    -- Architecture parameters
    self.embeddingDim = self.options.embeddingDim or 64
    self.hiddenDim = self.options.hiddenDim or 128
    self.numShells = self.options.numShells or 5
    self.attenuationFactor = self.options.attenuationFactor or 0.8
    
    -- Create hypershell structure
    self.hypershell = Hypershell.new(atomspace, rootHandle, {
        maxShells = self.numShells,
        maxNodesPerShell = self.options.maxNodesPerShell or 50
    })
    
    -- Create rooted tree from hypershell structure
    self.tree = self:buildTreeFromShells()
    
    -- Neural network components for each shell
    self.shellProcessors = {}
    self.shellAttention = {}
    self.shellAggregation = {}
    
    -- Initialize neural components
    self:initializeNetworks()
    
    -- Cache for embeddings and activations
    self.embeddings = {}
    self.activations = {}
    self.shellOutputs = {}
end

-- Build a rooted tree structure from the hypershell
function RootedHypershell:buildTreeFromShells()
    local rootAtom = self.atomspace:getAtom(self.rootHandle)
    local rootValue = rootAtom and rootAtom.name or tostring(self.rootHandle)
    
    local tree = RootedTree.new(rootValue)
    
    -- Map atom handles to tree nodes
    local handleToNode = {[self.rootHandle] = tree:getRoot()}
    
    -- Build tree layer by layer from shells
    for depth = 1, self.hypershell:getMaxDepth() do
        local shell = self.hypershell:getShell(depth)
        
        for _, handle in ipairs(shell) do
            local parent = self.hypershell:getNodeParent(handle)
            local parentNode = handleToNode[parent]
            
            if parentNode then
                local atom = self.atomspace:getAtom(handle)
                local value = atom and atom.name or tostring(handle)
                local node = tree:addChild(parentNode, value)
                handleToNode[handle] = node
                
                -- Store handle in node for reference
                node.atomHandle = handle
            end
        end
    end
    
    return tree
end

-- Initialize neural network components for each shell
function RootedHypershell:initializeNetworks()
    for shellIdx = 0, self.hypershell:getMaxDepth() do
        -- Shell-specific processing network
        local processor = nn.Sequential()
        processor:add(nn.Linear(self.embeddingDim, self.hiddenDim))
        processor:add(nn.Tanh())
        processor:add(nn.Linear(self.hiddenDim, self.embeddingDim))
        self.shellProcessors[shellIdx] = processor
        
        -- Shell-specific attention mechanism
        local attention = nn.Sequential()
        attention:add(nn.Linear(self.embeddingDim, self.hiddenDim))
        attention:add(nn.Tanh())
        attention:add(nn.Linear(self.hiddenDim, 1))
        attention:add(nn.Sigmoid())
        self.shellAttention[shellIdx] = attention
        
        -- Shell aggregation network
        local aggregation = nn.Sequential()
        aggregation:add(nn.Linear(self.embeddingDim, self.embeddingDim))
        aggregation:add(nn.ReLU())
        self.shellAggregation[shellIdx] = aggregation
    end
end

-- Get embedding for an atom
function RootedHypershell:getEmbedding(handle)
    if self.embeddings[handle] then
        return self.embeddings[handle]
    end
    
    -- Create embedding based on atom type and properties
    local atom = self.atomspace:getAtom(handle)
    if not atom then
        return torch.zeros(self.embeddingDim)
    end
    
    -- Use deterministic initialization based on atom handle
    torch.manualSeed(handle % 2147483647)  -- Use handle as seed for determinism
    local embedding = torch.randn(self.embeddingDim) * 0.1
    
    -- Incorporate truth value
    local strength, confidence = self.atomspace:getTruthValue(handle)
    if strength and confidence then
        embedding[1] = strength
        embedding[2] = confidence
    end
    
    -- Incorporate attention value
    local sti, lti, _ = self.atomspace:getAttentionValue(handle)
    if sti and lti then
        embedding[3] = math.min(sti / 100.0, 1.0)
        embedding[4] = math.min(lti / 100.0, 1.0)
    end
    
    self.embeddings[handle] = embedding
    return embedding
end

-- Forward propagation through the rooted hypershell
function RootedHypershell:updateOutput(input)
    -- Input can be nil (process entire structure) or a target handle
    local targetHandle = input or self.rootHandle
    
    -- Process shells from root outward
    for shellIdx = 0, self.hypershell:getMaxDepth() do
        local shell = self.hypershell:getShell(shellIdx)
        local shellActivations = {}
        
        for _, handle in ipairs(shell) do
            -- Get embedding
            local embedding = self:getEmbedding(handle)
            
            -- Process through shell-specific network
            local activation = self.shellProcessors[shellIdx]:forward(embedding)
            
            -- Apply attention
            local attentionWeight = self.shellAttention[shellIdx]:forward(activation)
            activation = activation * attentionWeight:expandAs(activation)
            
            -- Store activation
            self.activations[handle] = activation
            table.insert(shellActivations, activation)
        end
        
        -- Aggregate shell outputs
        if #shellActivations > 0 then
            local shellTensor = torch.cat(shellActivations, 1)
            local meanActivation = torch.mean(shellTensor, 1):squeeze()
            self.shellOutputs[shellIdx] = self.shellAggregation[shellIdx]:forward(meanActivation)
        end
    end
    
    -- Combine all shell outputs for final output
    local allShellOutputs = {}
    for shellIdx = 0, self.hypershell:getMaxDepth() do
        if self.shellOutputs[shellIdx] then
            table.insert(allShellOutputs, self.shellOutputs[shellIdx])
        end
    end
    
    if #allShellOutputs > 0 then
        self.output = torch.mean(torch.cat(allShellOutputs, 1), 1):squeeze()
    else
        self.output = torch.zeros(self.embeddingDim)
    end
    
    return self.output
end

-- Backward propagation
function RootedHypershell:updateGradInput(input, gradOutput)
    -- Distribute gradient to all shells
    local shellGradients = {}
    local numShells = self.hypershell:getMaxDepth() + 1
    
    for shellIdx = 0, self.hypershell:getMaxDepth() do
        shellGradients[shellIdx] = gradOutput / numShells
    end
    
    -- Backpropagate through each shell
    for shellIdx = self.hypershell:getMaxDepth(), 0, -1 do
        if self.shellOutputs[shellIdx] then
            -- Backprop through aggregation
            local gradAgg = self.shellAggregation[shellIdx]:backward(
                self.shellOutputs[shellIdx], shellGradients[shellIdx]
            )
            
            -- Distribute to shell nodes
            local shell = self.hypershell:getShell(shellIdx)
            for _, handle in ipairs(shell) do
                if self.activations[handle] then
                    -- Backprop through attention
                    local activation = self.activations[handle]
                    local gradAttn = self.shellAttention[shellIdx]:backward(activation, gradAgg)
                    
                    -- Backprop through processor
                    local embedding = self:getEmbedding(handle)
                    self.shellProcessors[shellIdx]:backward(embedding, gradAttn)
                end
            end
        end
    end
    
    self.gradInput = torch.zeros(self.embeddingDim)
    return self.gradInput
end

-- Spread attention through the rooted hypershell
function RootedHypershell:spreadAttention(iterations)
    iterations = iterations or 3
    
    for _ = 1, iterations do
        self.hypershell:spreadAttention(self.attenuationFactor)
    end
end

-- Get the tree structure
function RootedHypershell:getTree()
    return self.tree
end

-- Get the hypershell structure
function RootedHypershell:getHypershell()
    return self.hypershell
end

-- Get statistics about the architecture
function RootedHypershell:getStats()
    local treeStats = {
        nodeCount = self.tree:getNodeCount(),
        maxDepth = self.tree:getMaxDepth(),
        rootValue = self.tree:getRoot().value
    }
    
    local shellStats = self.hypershell:getStats()
    
    return {
        tree = treeStats,
        shell = shellStats,
        embeddingDim = self.embeddingDim,
        hiddenDim = self.hiddenDim,
        numShells = self.numShells
    }
end

-- Perform hierarchical inference
function RootedHypershell:hierarchicalInference(query)
    -- Process query through each shell level
    local shellResults = {}
    
    for shellIdx = 0, self.hypershell:getMaxDepth() do
        local shell = self.hypershell:getShell(shellIdx)
        local shellResult = {}
        
        for _, handle in ipairs(shell) do
            local activation = self.activations[handle]
            if activation then
                local similarity = torch.dot(query, activation)
                table.insert(shellResult, {
                    handle = handle,
                    similarity = similarity,
                    depth = shellIdx
                })
            end
        end
        
        shellResults[shellIdx] = shellResult
    end
    
    return shellResults
end

-- Get nodes relevant to a query using attention
function RootedHypershell:getRelevantNodes(query, topK)
    topK = topK or 10
    
    -- Compute relevance scores
    local scores = {}
    for handle, activation in pairs(self.activations) do
        if activation then
            local similarity = torch.dot(query, activation)
            local sti, _, _ = self.atomspace:getAttentionValue(handle)
            local relevance = similarity * (sti or 1.0) / 100.0
            table.insert(scores, {handle = handle, relevance = relevance})
        end
    end
    
    -- Sort by relevance
    table.sort(scores, function(a, b) return a.relevance > b.relevance end)
    
    -- Return top K
    local result = {}
    for i = 1, math.min(topK, #scores) do
        table.insert(result, scores[i])
    end
    
    return result
end

-- Export the tree structure as a string
function RootedHypershell:exportTree()
    return self.tree:toString()
end

-- Calculate rooted tree counts for verification
function RootedHypershell.getRootedTreeCounts(n)
    return RootedTree.getA000081Sequence(n)
end

return RootedHypershell
