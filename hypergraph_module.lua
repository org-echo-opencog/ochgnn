--[[
HypergraphModule - Neural Network Module for OpenCog Hypergraph Processing

This module implements neural network operations on OpenCog atomspace
hypergraphs. It supports forward and backward propagation through
hypergraph structures, attention allocation, and pattern matching.

Author: Neural Network Graph Package + OpenCog Integration
License: Same as parent project  
]]--

local utils = require('nngraph.utils')
local AtomTypes = require('nngraph.atom_types')
local AtomNode = require('nngraph.atom_node')
require 'nn'

local HypergraphModule, parent = torch.class('nn.HypergraphModule', 'nn.Module')

function HypergraphModule:__init(atomspace, options)
    parent.__init(self)
    
    self.atomspace = atomspace
    self.options = options or {}
    
    -- Neural network parameters
    self.embeddingDim = self.options.embeddingDim or 64
    self.hiddenDim = self.options.hiddenDim or 128
    self.outputDim = self.options.outputDim or 32
    self.learningRate = self.options.learningRate or 0.01
    
    -- Attention parameters
    self.attentionThreshold = self.options.attentionThreshold or 50
    self.maxAttentionalFocus = self.options.maxAttentionalFocus or 10
    
    -- Network layers
    self.embeddings = {}      -- Embedding layers for different atom types
    self.attention = nn.Sequential()
    self.hypergraphConv = nn.Sequential()
    self.output = nn.Sequential()
    
    -- AtomNode cache
    self.atomNodes = {}
    
    -- Pattern matching
    self.patterns = {}
    self.matches = {}
    
    -- Initialize network layers
    self:initializeLayers()
end

-- Initialize neural network layers
function HypergraphModule:initializeLayers()
    -- Attention mechanism
    self.attention:add(nn.Linear(self.embeddingDim, self.hiddenDim))
    self.attention:add(nn.Tanh())
    self.attention:add(nn.Linear(self.hiddenDim, 1))
    self.attention:add(nn.Sigmoid())
    
    -- Hypergraph convolution layers
    self.hypergraphConv:add(nn.Linear(self.embeddingDim, self.hiddenDim))
    self.hypergraphConv:add(nn.ReLU())
    self.hypergraphConv:add(nn.Linear(self.hiddenDim, self.embeddingDim))
    
    -- Output layers
    self.output:add(nn.Linear(self.embeddingDim, self.outputDim))
    self.output:add(nn.Tanh())
    
    -- Initialize embedding layers for each atom type
    for _, atomType in ipairs(AtomTypes.getNodeTypes()) do
        local embedDim = AtomTypes.getDefaultEmbedDim(atomType)
        self.embeddings[atomType] = nn.LookupTable(10000, embedDim) -- Assume max 10k atoms per type
    end
end

-- Get or create AtomNode for an atom handle
function HypergraphModule:getAtomNode(atomHandle)
    if not self.atomNodes[atomHandle] then
        self.atomNodes[atomHandle] = AtomNode(self.atomspace, atomHandle)
    end
    return self.atomNodes[atomHandle]
end

-- Forward propagation through hypergraph
function HypergraphModule:updateOutput(input)
    -- Input can be:
    -- 1. Single atom handle
    -- 2. List of atom handles  
    -- 3. Query pattern for pattern matching
    
    if type(input) == 'number' then
        -- Single atom
        return self:forwardSingleAtom(input)
    elseif type(input) == 'table' and #input > 0 then
        if type(input[1]) == 'number' then
            -- List of atom handles
            return self:forwardAtomList(input)
        else
            -- Pattern matching query
            return self:forwardPattern(input)
        end
    else
        error("Invalid input type for HypergraphModule")
    end
end

-- Forward pass for single atom
function HypergraphModule:forwardSingleAtom(atomHandle)
    local atomNode = self:getAtomNode(atomHandle)
    local activation = atomNode:forward()
    
    -- Apply attention mechanism
    local embedding = atomNode:getEmbedding()
    if embedding and embedding:nElement() > 0 then
        local attentionWeight = self.attention:forward(embedding)
        activation = activation * attentionWeight[1]
        
        -- Update attention value in atomspace
        atomNode:updateAttentionValue()
    end
    
    -- Apply hypergraph convolution
    if embedding and embedding:nElement() > 0 then
        local convOutput = self.hypergraphConv:forward(embedding)
        atomNode:setEmbedding(convOutput)
    end
    
    self.output = torch.Tensor({activation})
    return self.output
end

-- Forward pass for list of atoms
function HypergraphModule:forwardAtomList(atomHandles)
    local activations = {}
    local embeddings = {}
    
    -- Process each atom
    for _, atomHandle in ipairs(atomHandles) do
        local atomNode = self:getAtomNode(atomHandle)
        local activation = atomNode:forward()
        table.insert(activations, activation)
        
        local embedding = atomNode:getEmbedding()
        if embedding and embedding:nElement() > 0 then
            table.insert(embeddings, embedding)
        end
    end
    
    -- Aggregate activations
    local activationTensor = torch.Tensor(activations)
    
    -- Apply attention to select most important atoms
    local attentionScores = {}
    for _, embedding in ipairs(embeddings) do
        local score = self.attention:forward(embedding)[1]
        table.insert(attentionScores, score)
    end
    
    -- Focus on top-k atoms by attention
    if #attentionScores > self.maxAttentionalFocus then
        local _, indices = torch.Tensor(attentionScores):topk(self.maxAttentionalFocus)
        local focusedActivations = torch.Tensor(self.maxAttentionalFocus)
        for i = 1, self.maxAttentionalFocus do
            focusedActivations[i] = activations[indices[i]]
        end
        activationTensor = focusedActivations
    end
    
    -- Apply hypergraph convolution to embeddings
    for i, embedding in ipairs(embeddings) do
        if i <= #atomHandles then
            local convOutput = self.hypergraphConv:forward(embedding)
            local atomNode = self:getAtomNode(atomHandles[i])
            atomNode:setEmbedding(convOutput)
        end
    end
    
    self.output = activationTensor
    return self.output
end

-- Forward pass for pattern matching
function HypergraphModule:forwardPattern(pattern)
    local matches = self:matchPattern(pattern)
    local matchActivations = {}
    
    for _, match in ipairs(matches) do
        local activation = 1.0
        for _, atomHandle in ipairs(match) do
            local atomNode = self:getAtomNode(atomHandle)
            activation = activation * atomNode:getActivation()
        end
        table.insert(matchActivations, activation)
    end
    
    if #matchActivations > 0 then
        self.output = torch.Tensor(matchActivations)
    else
        self.output = torch.Tensor({0.0})
    end
    
    return self.output
end

-- Pattern matching in atomspace
function HypergraphModule:matchPattern(pattern)
    local matches = {}
    
    -- Simple pattern matching implementation
    -- pattern format: {type=AtomType, name=name, outgoing={...}}
    if pattern.type then
        local candidateAtoms = self.atomspace:getAtomsByType(pattern.type)
        
        for _, atom in ipairs(candidateAtoms) do
            if self:atomMatchesPattern(atom, pattern) then
                table.insert(matches, {atom.handle})
            end
        end
    end
    
    return matches
end

-- Check if atom matches pattern
function HypergraphModule:atomMatchesPattern(atom, pattern)
    -- Check type
    if pattern.type and atom.type ~= pattern.type then
        return false
    end
    
    -- Check name
    if pattern.name and atom.name ~= pattern.name then
        return false
    end
    
    -- Check outgoing atoms (for links)
    if pattern.outgoing then
        if #atom.outgoing ~= #pattern.outgoing then
            return false
        end
        
        for i, outgoingPattern in ipairs(pattern.outgoing) do
            local outgoingAtom = self.atomspace:getAtom(atom.outgoing[i])
            if not outgoingAtom or not self:atomMatchesPattern(outgoingAtom, outgoingPattern) then
                return false
            end
        end
    end
    
    return true
end

-- Backward propagation
function HypergraphModule:updateGradInput(input, gradOutput)
    if not gradOutput then
        gradOutput = torch.ones(self.output:size())
    end
    
    -- Propagate gradients through attention mechanism
    self.attention:backward(self.attention.output, gradOutput)
    
    -- Propagate gradients through hypergraph convolution
    self.hypergraphConv:backward(self.hypergraphConv.output, gradOutput)
    
    -- Update embeddings based on gradients
    if type(input) == 'table' and #input > 0 and type(input[1]) == 'number' then
        for _, atomHandle in ipairs(input) do
            local atomNode = self:getAtomNode(atomHandle)
            -- Update truth values based on gradients (simplified)
            local currentStrength, currentConfidence = self.atomspace:getTruthValue(atomHandle)
            if currentStrength then
                local newStrength = currentStrength + gradOutput:mean() * self.learningRate
                newStrength = math.max(0, math.min(1, newStrength))
                atomNode:updateTruthValue(newStrength, currentConfidence)
            end
        end
    end
    
    self.gradInput = gradOutput
    return self.gradInput
end

-- Update parameters
function HypergraphModule:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    
    -- Update attention parameters
    self.attention:accGradParameters(self.attention.output, gradOutput, scale)
    
    -- Update hypergraph convolution parameters
    self.hypergraphConv:accGradParameters(self.hypergraphConv.output, gradOutput, scale)
    
    -- Update output layer parameters
    self.output:accGradParameters(self.output.output, gradOutput, scale)
end

-- Spread activation through atomspace
function HypergraphModule:spreadActivation(sourceHandle, iterations)
    iterations = iterations or 3
    
    for iter = 1, iterations do
        local activeAtoms = {}
        
        -- Collect atoms above attention threshold
        for handle, atom in pairs(self.atomspace.atoms) do
            local sti = self.atomspace:getAttentionValue(handle)
            if sti and sti >= self.attentionThreshold then
                table.insert(activeAtoms, handle)
            end
        end
        
        -- Spread activation to neighboring atoms
        for _, handle in ipairs(activeAtoms) do
            local atom = self.atomspace:getAtom(handle)
            local atomNode = self:getAtomNode(handle)
            
            -- Spread to outgoing atoms
            for _, outgoingHandle in ipairs(atom.outgoing) do
                local outgoingNode = self:getAtomNode(outgoingHandle)
                local currentActivation = outgoingNode:getActivation()
                local spreadActivation = atomNode:getActivation() * 0.8 -- Decay factor
                outgoingNode:setActivation(currentActivation + spreadActivation)
                outgoingNode:updateAttentionValue()
            end
            
            -- Spread to incoming atoms  
            for _, incomingHandle in ipairs(atom.incoming) do
                local incomingNode = self:getAtomNode(incomingHandle)
                local currentActivation = incomingNode:getActivation()
                local spreadActivation = atomNode:getActivation() * 0.6 -- Lower decay for reverse
                incomingNode:setActivation(currentActivation + spreadActivation)
                incomingNode:updateAttentionValue()
            end
        end
    end
end

-- Get parameters for optimization
function HypergraphModule:parameters()
    local params = {}
    local gradParams = {}
    
    -- Get parameters from all submodules
    local attentionParams, attentionGradParams = self.attention:parameters()
    local convParams, convGradParams = self.hypergraphConv:parameters()
    local outputParams, outputGradParams = self.output:parameters()
    
    -- Combine all parameters
    if attentionParams then
        for _, p in ipairs(attentionParams) do table.insert(params, p) end
        for _, p in ipairs(attentionGradParams) do table.insert(gradParams, p) end
    end
    
    if convParams then
        for _, p in ipairs(convParams) do table.insert(params, p) end
        for _, p in ipairs(convGradParams) do table.insert(gradParams, p) end
    end
    
    if outputParams then
        for _, p in ipairs(outputParams) do table.insert(params, p) end
        for _, p in ipairs(outputGradParams) do table.insert(gradParams, p) end
    end
    
    return params, gradParams
end

-- Clear cached atom nodes
function HypergraphModule:clearCache()
    self.atomNodes = {}
end

-- Export the HypergraphModule class
return HypergraphModule