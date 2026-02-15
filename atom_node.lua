--[[
AtomNode - Neural Network Node for OpenCog Atoms

This module extends nngraph.Node to represent OpenCog atoms in neural network
computations. AtomNodes can represent both individual atoms and hypergraph
structures, supporting forward and backward propagation through hypergraph
neural networks.

Author: Neural Network Graph Package + OpenCog Integration  
License: Same as parent project
]]--

local utils = require('nngraph.utils')
local AtomTypes = require('nngraph.atom_types')
require 'nngraph.node'

local AtomNode, parent = torch.class('nngraph.AtomNode', 'nngraph.Node')

function AtomNode:__init(atomspace, atomHandle, data)
    -- Initialize parent node
    data = data or {}
    data.atomHandle = atomHandle
    data.atomspace = atomspace
    parent.__init(self, data)
    
    -- Get atom from atomspace
    self.atom = atomspace:getAtom(atomHandle)
    if not self.atom then
        error("Invalid atom handle: " .. tostring(atomHandle))
    end
    
    -- Neural network properties
    self.embedding = nil      -- Learned embedding vector
    self.activation = nil     -- Current activation value
    self.lastUpdate = 0       -- Last update timestamp
    
    -- Hypergraph structure
    self.outgoingNodes = {}   -- Child AtomNodes for links
    self.incomingNodes = {}   -- Parent AtomNodes for links
    
    -- Truth value propagation
    self.truthGradient = nil  -- Gradient for truth value learning
    self.attentionGradient = nil -- Gradient for attention learning
    
    -- Initialize embedding if atom type supports it
    self:initializeEmbedding()
end

-- Initialize embedding vector for the atom
function AtomNode:initializeEmbedding()
    if AtomTypes.canEmbed(self.atom.type) then
        local embedDim = AtomTypes.getDefaultEmbedDim(self.atom.type)
        self.embedding = torch.randn(embedDim) * 0.1
        
        -- For numeric nodes, set embedding to the numeric value
        if AtomTypes.isNumericType(self.atom.type) and self.atom.name then
            local numValue = tonumber(self.atom.name)
            if numValue then
                self.embedding:fill(0)
                self.embedding[1] = numValue
            end
        end
    end
end

-- Get the current embedding vector
function AtomNode:getEmbedding()
    if self.embedding then
        return self.embedding:clone()
    elseif AtomTypes.isNumericType(self.atom.type) and self.atom.name then
        -- Return numeric value as scalar tensor
        local numValue = tonumber(self.atom.name) or 0
        return torch.Tensor({numValue})
    else
        -- Return zero embedding for non-embeddable types
        return torch.zeros(AtomTypes.getDefaultEmbedDim(self.atom.type))
    end
end

-- Set the embedding vector
function AtomNode:setEmbedding(embedding)
    if AtomTypes.canEmbed(self.atom.type) then
        self.embedding = embedding:clone()
        return true
    end
    return false
end

-- Get current activation value
function AtomNode:getActivation()
    if self.activation then
        return self.activation
    else
        -- Compute activation from truth value
        local strength, confidence = self.data.atomspace:getTruthValue(self.data.atomHandle)
        if strength and confidence then
            return strength * confidence
        else
            return 0.0
        end
    end
end

-- Set activation value
function AtomNode:setActivation(activation)
    self.activation = activation
    self.lastUpdate = os.time()
end

-- Apply activation function based on atom type
function AtomNode:applyActivation(input)
    local activationFunc = AtomTypes.getActivationFunction(self.atom.type)
    
    if activationFunc == 'sigmoid' then
        return torch.sigmoid(input)
    elseif activationFunc == 'tanh' then
        return torch.tanh(input)
    elseif activationFunc == 'relu' then
        return torch.clamp(input, 0, math.huge)
    elseif activationFunc == 'linear' then
        return input
    else
        return torch.tanh(input) -- default
    end
end

-- Forward propagation for atom node
function AtomNode:forward(input)
    if AtomTypes.isNodeType(self.atom.type) then
        return self:forwardNode(input)
    else
        return self:forwardLink(input)
    end
end

-- Forward propagation for node atoms
function AtomNode:forwardNode(input)
    local embedding = self:getEmbedding()
    
    if input then
        -- Combine input with embedding
        if torch.isTensor(input) and embedding:nElement() == input:nElement() then
            local combined = embedding + input
            self.activation = self:applyActivation(combined):mean()
        else
            self.activation = self:applyActivation(embedding):mean()
        end
    else
        self.activation = self:applyActivation(embedding):mean()
    end
    
    return self.activation
end

-- Forward propagation for link atoms  
function AtomNode:forwardLink(input)
    local aggregationFunc = AtomTypes.getAggregationFunction(self.atom.type)
    local outgoingActivations = {}
    
    -- Collect activations from outgoing atoms
    for _, outgoingHandle in ipairs(self.atom.outgoing) do
        if self.outgoingNodes[outgoingHandle] then
            local activation = self.outgoingNodes[outgoingHandle]:getActivation()
            table.insert(outgoingActivations, activation)
        end
    end
    
    -- Apply aggregation function
    if #outgoingActivations == 0 then
        self.activation = 0.0
    elseif aggregationFunc == 'sum' then
        self.activation = torch.Tensor(outgoingActivations):sum()
    elseif aggregationFunc == 'mean' then
        self.activation = torch.Tensor(outgoingActivations):mean()
    elseif aggregationFunc == 'max' then
        self.activation = torch.Tensor(outgoingActivations):max()
    elseif aggregationFunc == 'min' then
        self.activation = torch.Tensor(outgoingActivations):min()
    elseif aggregationFunc == 'logical_and' then
        self.activation = torch.Tensor(outgoingActivations):min() -- Min for AND
    elseif aggregationFunc == 'logical_or' then
        self.activation = torch.Tensor(outgoingActivations):max() -- Max for OR
    elseif aggregationFunc == 'inheritance' or aggregationFunc == 'implication' then
        -- For inheritance/implication: strength = min(premise, conclusion)
        if #outgoingActivations >= 2 then
            self.activation = math.min(outgoingActivations[1], outgoingActivations[2])
        else
            self.activation = 0.0
        end
    elseif aggregationFunc == 'similarity' then
        -- For similarity: strength = 1 - |a - b|
        if #outgoingActivations >= 2 then
            self.activation = 1.0 - math.abs(outgoingActivations[1] - outgoingActivations[2])
        else
            self.activation = 0.0
        end
    else
        -- Default: sum
        self.activation = torch.Tensor(outgoingActivations):sum()
    end
    
    return self.activation
end

-- Update truth value based on neural network output
function AtomNode:updateTruthValue(strength, confidence)
    if strength then
        self.data.atomspace:setTruthValue(self.data.atomHandle, strength, confidence or 1.0)
    elseif self.activation then
        -- Update truth value from activation
        local newStrength = math.max(0, math.min(1, self.activation))
        self.data.atomspace:setTruthValue(self.data.atomHandle, newStrength, 0.9)
    end
end

-- Update attention value based on activation
function AtomNode:updateAttentionValue()
    if self.activation then
        local sti = math.floor(self.activation * 100) -- Scale to attention units
        local lti = self.data.atomspace:getAttentionValue(self.data.atomHandle)
        if lti == nil then lti = 0 end
        self.data.atomspace:setAttentionValue(self.data.atomHandle, sti, lti, false)
    end
end

-- Add outgoing connection to another AtomNode
function AtomNode:addOutgoing(atomNode)
    if atomNode and atomNode.data.atomHandle then
        self.outgoingNodes[atomNode.data.atomHandle] = atomNode
        atomNode.incomingNodes[self.data.atomHandle] = self
    end
end

-- Remove outgoing connection
function AtomNode:removeOutgoing(atomNode)
    if atomNode and atomNode.data.atomHandle then
        self.outgoingNodes[atomNode.data.atomHandle] = nil
        if atomNode.incomingNodes[self.data.atomHandle] then
            atomNode.incomingNodes[self.data.atomHandle] = nil
        end
    end
end

-- Get string representation for debugging
function AtomNode:toString()
    local atomInfo = string.format("%s(%s)", self.atom.type, self.atom.name or "unnamed")
    local tvInfo = ""
    local strength, confidence = self.data.atomspace:getTruthValue(self.data.atomHandle)
    if strength then
        tvInfo = string.format(" TV:(%.3f,%.3f)", strength, confidence)
    end
    local avInfo = ""
    local sti, lti = self.data.atomspace:getAttentionValue(self.data.atomHandle)
    if sti then
        avInfo = string.format(" AV:(%d,%d)", sti, lti)
    end
    return atomInfo .. tvInfo .. avInfo
end

-- Override parent label method
function AtomNode:label()
    local baseLabel = parent.label(self)
    return self:toString() .. "\\n" .. baseLabel
end

-- Export the AtomNode class
return AtomNode