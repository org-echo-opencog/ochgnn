--[[
OpenCog AtomSpace Implementation for Hypergraph Neural Networks

This module implements the core AtomSpace data structure that serves as the foundation
for hypergraph neural network operations. The AtomSpace manages atoms (nodes) and 
links (hyperedges) with associated truth values and attention values.

Author: Neural Network Graph Package + OpenCog Integration
License: Same as parent project
]]--

local utils = require('nngraph.utils')
local istensor = torch.isTensor
local istable = utils.istable

-- AtomSpace class - manages the hypergraph of atoms and links
local AtomSpace = torch.class('nngraph.AtomSpace')

function AtomSpace:__init()
    -- Core data structures
    self.atoms = {}           -- Table of all atoms by handle (UUID)
    self.atomsByType = {}     -- Atoms organized by type
    self.atomsByName = {}     -- Named atoms for quick lookup
    self.links = {}           -- Links organized by type and outgoing set
    self.nextHandle = 1       -- Handle generator
    
    -- Attention and truth value tracking
    self.attentionBank = {}   -- Attention allocation system
    self.truthValues = {}     -- Truth values for atoms
    
    -- Neural network integration
    self.embeddings = {}      -- Learned embeddings for atoms
    self.activations = {}     -- Current activation levels
    
    -- Statistics
    self.stats = {
        totalAtoms = 0,
        totalLinks = 0,
        typeCount = {}
    }
end

-- Generate unique handle for new atoms
function AtomSpace:generateHandle()
    local handle = self.nextHandle
    self.nextHandle = self.nextHandle + 1
    return handle
end

-- Add an atom to the atomspace
function AtomSpace:addAtom(atomType, name, outgoing)
    local handle = self:generateHandle()
    
    local atom = {
        handle = handle,
        type = atomType,
        name = name or nil,
        outgoing = outgoing or {},  -- For links, list of atoms they connect
        incoming = {},              -- Atoms that link to this atom
        truthValue = {strength = 1.0, confidence = 1.0},  -- Default TV
        attentionValue = {sti = 0, lti = 0, vlti = false}, -- Default AV
        created = os.time(),
        accessed = os.time()
    }
    
    -- Store in main atom table
    self.atoms[handle] = atom
    
    -- Index by type
    if not self.atomsByType[atomType] then
        self.atomsByType[atomType] = {}
    end
    table.insert(self.atomsByType[atomType], atom)
    
    -- Index by name if provided
    if name then
        if not self.atomsByName[name] then
            self.atomsByName[name] = {}
        end
        table.insert(self.atomsByName[name], atom)
    end
    
    -- Handle outgoing connections for links
    if outgoing and #outgoing > 0 then
        for _, targetHandle in ipairs(outgoing) do
            local targetAtom = self.atoms[targetHandle]
            if targetAtom then
                table.insert(targetAtom.incoming, handle)
            end
        end
        self.stats.totalLinks = self.stats.totalLinks + 1
    else
        self.stats.totalAtoms = self.stats.totalAtoms + 1
    end
    
    -- Update statistics
    if not self.stats.typeCount[atomType] then
        self.stats.typeCount[atomType] = 0
    end
    self.stats.typeCount[atomType] = self.stats.typeCount[atomType] + 1
    
    return handle
end

-- Get atom by handle
function AtomSpace:getAtom(handle)
    return self.atoms[handle]
end

-- Get atoms by type
function AtomSpace:getAtomsByType(atomType)
    return self.atomsByType[atomType] or {}
end

-- Get atoms by name
function AtomSpace:getAtomsByName(name)
    return self.atomsByName[name] or {}
end

-- Set truth value for an atom
function AtomSpace:setTruthValue(handle, strength, confidence)
    local atom = self.atoms[handle]
    if atom then
        atom.truthValue = {strength = strength, confidence = confidence}
        return true
    end
    return false
end

-- Get truth value for an atom
function AtomSpace:getTruthValue(handle)
    local atom = self.atoms[handle]
    if atom then
        return atom.truthValue.strength, atom.truthValue.confidence
    end
    return nil, nil
end

-- Set attention value for an atom
function AtomSpace:setAttentionValue(handle, sti, lti, vlti)
    local atom = self.atoms[handle]
    if atom then
        atom.attentionValue = {sti = sti or 0, lti = lti or 0, vlti = vlti or false}
        return true
    end
    return false
end

-- Get attention value for an atom
function AtomSpace:getAttentionValue(handle)
    local atom = self.atoms[handle]
    if atom then
        local av = atom.attentionValue
        return av.sti, av.lti, av.vlti
    end
    return nil, nil, nil
end

-- Remove an atom from the atomspace
function AtomSpace:removeAtom(handle)
    local atom = self.atoms[handle]
    if not atom then
        return false
    end
    
    -- Remove from incoming links of target atoms
    for _, targetHandle in ipairs(atom.outgoing) do
        local targetAtom = self.atoms[targetHandle]
        if targetAtom then
            for i, incomingHandle in ipairs(targetAtom.incoming) do
                if incomingHandle == handle then
                    table.remove(targetAtom.incoming, i)
                    break
                end
            end
        end
    end
    
    -- Remove from type index
    if self.atomsByType[atom.type] then
        for i, indexedAtom in ipairs(self.atomsByType[atom.type]) do
            if indexedAtom.handle == handle then
                table.remove(self.atomsByType[atom.type], i)
                break
            end
        end
    end
    
    -- Remove from name index
    if atom.name and self.atomsByName[atom.name] then
        for i, indexedAtom in ipairs(self.atomsByName[atom.name]) do
            if indexedAtom.handle == handle then
                table.remove(self.atomsByName[atom.name], i)
                break
            end
        end
    end
    
    -- Remove from main table
    self.atoms[handle] = nil
    
    -- Update statistics
    if #atom.outgoing > 0 then
        self.stats.totalLinks = self.stats.totalLinks - 1
    else
        self.stats.totalAtoms = self.stats.totalAtoms - 1
    end
    self.stats.typeCount[atom.type] = self.stats.typeCount[atom.type] - 1
    
    return true
end

-- Get statistics about the atomspace
function AtomSpace:getStatistics()
    return self.stats
end

-- Clear the entire atomspace
function AtomSpace:clear()
    self.atoms = {}
    self.atomsByType = {}
    self.atomsByName = {}
    self.links = {}
    self.attentionBank = {}
    self.truthValues = {}
    self.embeddings = {}
    self.activations = {}
    self.stats = {
        totalAtoms = 0,
        totalLinks = 0,
        typeCount = {}
    }
    self.nextHandle = 1
end

-- Export the AtomSpace class
return AtomSpace