--[[
Hypershell Module - Shell-based Hypergraph Organization

This module implements a hypershell architecture that organizes hypergraph nodes
into concentric shells around a root node, enabling shell-wise processing and
attention propagation.

Author: Neural Network Graph Package + OpenCog Integration
License: Same as parent project
]]--

local AtomTypes = require('nngraph.atom_types')
require 'nn'

local Hypershell = {}
Hypershell.__index = Hypershell

-- Create a new hypershell structure
function Hypershell.new(atomspace, rootHandle, options)
    local self = setmetatable({}, Hypershell)
    
    self.atomspace = atomspace
    self.rootHandle = rootHandle
    self.options = options or {}
    
    -- Shell parameters
    self.maxShells = self.options.maxShells or 10
    self.maxNodesPerShell = self.options.maxNodesPerShell or 100
    
    -- Shell organization
    self.shells = {}  -- shells[depth] = {handle1, handle2, ...}
    self.nodeDepths = {}  -- nodeDepths[handle] = depth
    self.nodeParents = {}  -- nodeParents[handle] = parentHandle
    
    -- Shell statistics
    self.shellSizes = {}
    self.maxDepth = 0
    
    -- Build the shell structure
    self:buildShells()
    
    return self
end

-- Build shell structure from root using BFS
function Hypershell:buildShells()
    -- Initialize with root at depth 0
    self.shells[0] = {self.rootHandle}
    self.nodeDepths[self.rootHandle] = 0
    self.nodeParents[self.rootHandle] = nil
    self.shellSizes[0] = 1
    
    -- BFS to build shells
    local visited = {[self.rootHandle] = true}
    local currentDepth = 0
    
    while currentDepth < self.maxShells and self.shells[currentDepth] do
        local currentShell = self.shells[currentDepth]
        local nextShell = {}
        
        -- For each node in current shell, find neighbors
        for _, handle in ipairs(currentShell) do
            local neighbors = self:getNeighbors(handle)
            
            for _, neighborHandle in ipairs(neighbors) do
                if not visited[neighborHandle] then
                    visited[neighborHandle] = true
                    table.insert(nextShell, neighborHandle)
                    self.nodeDepths[neighborHandle] = currentDepth + 1
                    self.nodeParents[neighborHandle] = handle
                    
                    -- Limit nodes per shell
                    if #nextShell >= self.maxNodesPerShell then
                        break
                    end
                end
            end
            
            if #nextShell >= self.maxNodesPerShell then
                break
            end
        end
        
        -- Add next shell if it has nodes
        if #nextShell > 0 then
            self.shells[currentDepth + 1] = nextShell
            self.shellSizes[currentDepth + 1] = #nextShell
            self.maxDepth = currentDepth + 1
        else
            break
        end
        
        currentDepth = currentDepth + 1
    end
end

-- Get neighbors of a node in the hypergraph
function Hypershell:getNeighbors(handle)
    local neighbors = {}
    local atom = self.atomspace:getAtom(handle)
    
    if not atom then
        return neighbors
    end
    
    -- If it's a link, its outgoing atoms are neighbors
    if atom.outgoing then
        for _, outgoingHandle in ipairs(atom.outgoing) do
            table.insert(neighbors, outgoingHandle)
        end
    end
    
    -- Find links that contain this atom
    local allAtoms = self.atomspace:getAllAtoms()
    for _, otherHandle in ipairs(allAtoms) do
        local otherAtom = self.atomspace:getAtom(otherHandle)
        if otherAtom and otherAtom.outgoing then
            for _, outgoingHandle in ipairs(otherAtom.outgoing) do
                if outgoingHandle == handle then
                    table.insert(neighbors, otherHandle)
                    -- Also add other atoms in the same link
                    for _, siblingHandle in ipairs(otherAtom.outgoing) do
                        if siblingHandle ~= handle then
                            table.insert(neighbors, siblingHandle)
                        end
                    end
                    break
                end
            end
        end
    end
    
    return neighbors
end

-- Get nodes at a specific shell depth
function Hypershell:getShell(depth)
    return self.shells[depth] or {}
end

-- Get the depth of a node
function Hypershell:getNodeDepth(handle)
    return self.nodeDepths[handle]
end

-- Get the parent of a node in the shell structure
function Hypershell:getNodeParent(handle)
    return self.nodeParents[handle]
end

-- Get the maximum depth
function Hypershell:getMaxDepth()
    return self.maxDepth
end

-- Get all nodes in all shells
function Hypershell:getAllNodes()
    local allNodes = {}
    for depth = 0, self.maxDepth do
        for _, handle in ipairs(self.shells[depth] or {}) do
            table.insert(allNodes, handle)
        end
    end
    return allNodes
end

-- Get shell sizes
function Hypershell:getShellSizes()
    return self.shellSizes
end

-- Get statistics about the hypershell
function Hypershell:getStats()
    return {
        rootHandle = self.rootHandle,
        maxDepth = self.maxDepth,
        totalNodes = #self:getAllNodes(),
        shellSizes = self.shellSizes,
        averageShellSize = self:getAverageShellSize()
    }
end

-- Calculate average shell size
function Hypershell:getAverageShellSize()
    local total = 0
    local count = 0
    for depth = 0, self.maxDepth do
        if self.shellSizes[depth] then
            total = total + self.shellSizes[depth]
            count = count + 1
        end
    end
    return count > 0 and total / count or 0
end

-- Propagate values through shells (outward from root)
function Hypershell:propagateOutward(initialValues, propagationFunction)
    local values = {[self.rootHandle] = initialValues}
    
    for depth = 0, self.maxDepth do
        local shell = self.shells[depth] or {}
        
        for _, handle in ipairs(shell) do
            local currentValue = values[handle]
            
            -- Propagate to neighbors in next shell
            if currentValue then
                local neighbors = self:getNeighbors(handle)
                for _, neighborHandle in ipairs(neighbors) do
                    if self.nodeDepths[neighborHandle] == depth + 1 then
                        -- Apply propagation function
                        local newValue = propagationFunction(currentValue, handle, neighborHandle, depth)
                        
                        -- Aggregate if node already has a value
                        if values[neighborHandle] then
                            values[neighborHandle] = (values[neighborHandle] + newValue) / 2
                        else
                            values[neighborHandle] = newValue
                        end
                    end
                end
            end
        end
    end
    
    return values
end

-- Propagate values through shells (inward toward root)
function Hypershell:propagateInward(leafValues, propagationFunction)
    local values = {}
    
    -- Initialize with leaf values
    for handle, value in pairs(leafValues) do
        values[handle] = value
    end
    
    -- Propagate from max depth to root
    for depth = self.maxDepth, 0, -1 do
        local shell = self.shells[depth] or {}
        
        for _, handle in ipairs(shell) do
            if values[handle] then
                local parent = self.nodeParents[handle]
                if parent then
                    local newValue = propagationFunction(values[handle], handle, parent, depth)
                    
                    -- Aggregate at parent
                    if values[parent] then
                        values[parent] = (values[parent] + newValue) / 2
                    else
                        values[parent] = newValue
                    end
                end
            end
        end
    end
    
    return values
end

-- Apply attention spreading through shells
function Hypershell:spreadAttention(attenuationFactor)
    attenuationFactor = attenuationFactor or 0.8
    
    -- Get initial attention from root
    local rootSTI, rootLTI, rootVLTI = self.atomspace:getAttentionValue(self.rootHandle)
    
    -- Spread attention outward
    local attentionValues = self:propagateOutward(rootSTI, function(value, fromHandle, toHandle, depth)
        return value * math.pow(attenuationFactor, depth + 1)
    end)
    
    -- Update attention values in atomspace
    for handle, sti in pairs(attentionValues) do
        if handle ~= self.rootHandle then
            local _, lti, vlti = self.atomspace:getAttentionValue(handle)
            self.atomspace:setAttentionValue(handle, sti, lti or 0, vlti or false)
        end
    end
    
    return attentionValues
end

-- Get path from root to a node
function Hypershell:getPathFromRoot(handle)
    local path = {}
    local current = handle
    
    while current do
        table.insert(path, 1, current)
        current = self.nodeParents[current]
    end
    
    return path
end

-- Find optimal root node based on centrality
function Hypershell.findOptimalRoot(atomspace, options)
    options = options or {}
    local method = options.method or "degree"  -- "degree", "closeness", or "betweenness"
    
    local allAtoms = atomspace:getAllAtoms()
    
    if method == "degree" then
        -- Use degree centrality (node with most connections)
        local maxDegree = -1
        local bestRoot = nil
        
        for _, handle in ipairs(allAtoms) do
            local tempShell = Hypershell.new(atomspace, handle, {maxShells = 1})
            local degree = #tempShell:getNeighbors(handle)
            
            if degree > maxDegree then
                maxDegree = degree
                bestRoot = handle
            end
        end
        
        return bestRoot
    elseif method == "central" then
        -- Use first node with high attention value
        local maxSTI = -1
        local bestRoot = nil
        
        for _, handle in ipairs(allAtoms) do
            local sti, _, _ = atomspace:getAttentionValue(handle)
            if sti > maxSTI then
                maxSTI = sti
                bestRoot = handle
            end
        end
        
        return bestRoot or allAtoms[1]
    else
        -- Default: return first atom
        return allAtoms[1]
    end
end

return Hypershell
