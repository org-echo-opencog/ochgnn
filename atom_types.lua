--[[
OpenCog Atom Types for Hypergraph Neural Networks

This module defines the standard atom types used in OpenCog atomspace,
adapted for neural network operations. Each atom type has specific
properties and behaviors that affect how they participate in hypergraph
neural network computations.

Author: Neural Network Graph Package + OpenCog Integration
License: Same as parent project
]]--

-- Define atom type constants
local AtomTypes = {}

-- Basic Node Types
AtomTypes.NODE = 'Node'
AtomTypes.CONCEPT_NODE = 'ConceptNode'
AtomTypes.PREDICATE_NODE = 'PredicateNode'  
AtomTypes.SCHEMA_NODE = 'SchemaNode'
AtomTypes.NUMBER_NODE = 'NumberNode'
AtomTypes.VARIABLE_NODE = 'VariableNode'
AtomTypes.TYPE_NODE = 'TypeNode'

-- Basic Link Types  
AtomTypes.LINK = 'Link'
AtomTypes.LIST_LINK = 'ListLink'
AtomTypes.SET_LINK = 'SetLink'
AtomTypes.ORDERED_LINK = 'OrderedLink'

-- Logical Links
AtomTypes.INHERITANCE_LINK = 'InheritanceLink'
AtomTypes.SIMILARITY_LINK = 'SimilarityLink'
AtomTypes.IMPLICATION_LINK = 'ImplicationLink'
AtomTypes.EQUIVALENCE_LINK = 'EquivalenceLink'
AtomTypes.AND_LINK = 'AndLink'
AtomTypes.OR_LINK = 'OrLink'
AtomTypes.NOT_LINK = 'NotLink'

-- Evaluation and Execution Links
AtomTypes.EVALUATION_LINK = 'EvaluationLink'
AtomTypes.EXECUTION_LINK = 'ExecutionLink'
AtomTypes.BIND_LINK = 'BindLink'
AtomTypes.LAMBDA_LINK = 'LambdaLink'

-- Quantifier Links
AtomTypes.FORALL_LINK = 'ForAllLink'
AtomTypes.EXISTS_LINK = 'ExistsLink'

-- Attention and Memory Links
AtomTypes.HEBBIAN_LINK = 'HebbianLink'
AtomTypes.ASYMMETRIC_HEBBIAN_LINK = 'AsymmetricHebbianLink'

-- Atom Type Properties
local AtomTypeProperties = {}

-- Node type properties
AtomTypeProperties[AtomTypes.CONCEPT_NODE] = {
    isNode = true,
    hasName = true,
    canEmbed = true,
    defaultEmbedDim = 64,
    activationFunction = 'tanh'
}

AtomTypeProperties[AtomTypes.PREDICATE_NODE] = {
    isNode = true,
    hasName = true,
    canEmbed = true,
    defaultEmbedDim = 32,
    activationFunction = 'sigmoid'
}

AtomTypeProperties[AtomTypes.SCHEMA_NODE] = {
    isNode = true,
    hasName = true,
    canEmbed = true,
    defaultEmbedDim = 48,
    activationFunction = 'relu'
}

AtomTypeProperties[AtomTypes.NUMBER_NODE] = {
    isNode = true,
    hasName = true,
    canEmbed = false,  -- Direct numeric value
    isNumeric = true,
    activationFunction = 'linear'
}

AtomTypeProperties[AtomTypes.VARIABLE_NODE] = {
    isNode = true,
    hasName = true,
    canEmbed = true,
    defaultEmbedDim = 16,
    isVariable = true,
    activationFunction = 'tanh'
}

-- Link type properties
AtomTypeProperties[AtomTypes.LIST_LINK] = {
    isNode = false,
    isLink = true,
    isOrdered = true,
    minArity = 0,
    maxArity = -1,  -- unlimited
    aggregationFunction = 'concatenate'
}

AtomTypeProperties[AtomTypes.SET_LINK] = {
    isNode = false,
    isLink = true,
    isOrdered = false,
    minArity = 0,
    maxArity = -1,
    aggregationFunction = 'sum'
}

AtomTypeProperties[AtomTypes.INHERITANCE_LINK] = {
    isNode = false,
    isLink = true,
    isOrdered = true,
    minArity = 2,
    maxArity = 2,
    aggregationFunction = 'inheritance',
    isLogical = true
}

AtomTypeProperties[AtomTypes.SIMILARITY_LINK] = {
    isNode = false,
    isLink = true,
    isOrdered = false,
    minArity = 2,
    maxArity = 2,
    aggregationFunction = 'similarity',
    isLogical = true
}

AtomTypeProperties[AtomTypes.IMPLICATION_LINK] = {
    isNode = false,
    isLink = true,
    isOrdered = true,
    minArity = 2,
    maxArity = 2,
    aggregationFunction = 'implication',
    isLogical = true
}

AtomTypeProperties[AtomTypes.AND_LINK] = {
    isNode = false,
    isLink = true,
    isOrdered = false,
    minArity = 2,
    maxArity = -1,
    aggregationFunction = 'logical_and',
    isLogical = true
}

AtomTypeProperties[AtomTypes.OR_LINK] = {
    isNode = false,
    isLink = true,
    isOrdered = false,
    minArity = 2,
    maxArity = -1,
    aggregationFunction = 'logical_or',
    isLogical = true
}

AtomTypeProperties[AtomTypes.EVALUATION_LINK] = {
    isNode = false,
    isLink = true,
    isOrdered = true,
    minArity = 2,
    maxArity = 2,
    aggregationFunction = 'evaluation',
    isLogical = true
}

-- Utility functions

-- Check if an atom type is a node
function AtomTypes.isNodeType(atomType)
    local props = AtomTypeProperties[atomType]
    return props and props.isNode == true
end

-- Check if an atom type is a link
function AtomTypes.isLinkType(atomType)
    local props = AtomTypeProperties[atomType]
    return props and props.isLink == true
end

-- Check if an atom type can be embedded
function AtomTypes.canEmbed(atomType)
    local props = AtomTypeProperties[atomType]
    return props and props.canEmbed == true
end

-- Get default embedding dimension for an atom type
function AtomTypes.getDefaultEmbedDim(atomType)
    local props = AtomTypeProperties[atomType]
    return props and props.defaultEmbedDim or 32
end

-- Get activation function for an atom type
function AtomTypes.getActivationFunction(atomType)
    local props = AtomTypeProperties[atomType]
    return props and props.activationFunction or 'tanh'
end

-- Get aggregation function for a link type
function AtomTypes.getAggregationFunction(atomType)
    local props = AtomTypeProperties[atomType]
    return props and props.aggregationFunction or 'sum'
end

-- Validate arity for a link type
function AtomTypes.validateArity(atomType, arity)
    local props = AtomTypeProperties[atomType]
    if not props or not props.isLink then
        return false, "Not a link type"
    end
    
    if props.minArity and arity < props.minArity then
        return false, "Arity too small: " .. arity .. " < " .. props.minArity
    end
    
    if props.maxArity and props.maxArity > 0 and arity > props.maxArity then
        return false, "Arity too large: " .. arity .. " > " .. props.maxArity
    end
    
    return true, "Valid arity"
end

-- Check if an atom type is logical
function AtomTypes.isLogicalType(atomType)
    local props = AtomTypeProperties[atomType]
    return props and props.isLogical == true
end

-- Check if an atom type is numeric
function AtomTypes.isNumericType(atomType)
    local props = AtomTypeProperties[atomType]
    return props and props.isNumeric == true
end

-- Check if an atom type is a variable
function AtomTypes.isVariableType(atomType)
    local props = AtomTypeProperties[atomType]
    return props and props.isVariable == true
end

-- Get all atom types
function AtomTypes.getAllTypes()
    local types = {}
    for k, v in pairs(AtomTypes) do
        if type(v) == 'string' and k:match('_[A-Z]+$') then
            table.insert(types, v)
        end
    end
    return types
end

-- Get node types
function AtomTypes.getNodeTypes()
    local nodeTypes = {}
    for atomType, props in pairs(AtomTypeProperties) do
        if props.isNode then
            table.insert(nodeTypes, atomType)
        end
    end
    return nodeTypes
end

-- Get link types  
function AtomTypes.getLinkTypes()
    local linkTypes = {}
    for atomType, props in pairs(AtomTypeProperties) do
        if props.isLink then
            table.insert(linkTypes, atomType)
        end
    end
    return linkTypes
end

-- Export modules
AtomTypes.Properties = AtomTypeProperties
return AtomTypes