--[[
Rooted Tree Data Structure

This module implements rooted tree structures for hypergraph organization.
Supports tree construction, traversal, and enumeration based on OEIS A000081
(number of unlabeled rooted trees with n nodes).

Author: Neural Network Graph Package + OpenCog Integration
License: Same as parent project
]]--

local RootedTree = {}
RootedTree.__index = RootedTree

-- Create a new rooted tree
function RootedTree.new(rootValue)
    local self = setmetatable({}, RootedTree)
    self.root = {
        value = rootValue,
        children = {},
        parent = nil,
        depth = 0
    }
    self.nodeCount = 1
    self.nodesByDepth = {[0] = {self.root}}
    self.maxDepth = 0
    return self
end

-- Add a child node to a parent node
function RootedTree:addChild(parentNode, childValue)
    local childNode = {
        value = childValue,
        children = {},
        parent = parentNode,
        depth = parentNode.depth + 1
    }
    
    table.insert(parentNode.children, childNode)
    self.nodeCount = self.nodeCount + 1
    
    -- Update depth tracking
    if childNode.depth > self.maxDepth then
        self.maxDepth = childNode.depth
        self.nodesByDepth[childNode.depth] = {}
    end
    
    if not self.nodesByDepth[childNode.depth] then
        self.nodesByDepth[childNode.depth] = {}
    end
    table.insert(self.nodesByDepth[childNode.depth], childNode)
    
    return childNode
end

-- Get all nodes at a specific depth
function RootedTree:getNodesAtDepth(depth)
    return self.nodesByDepth[depth] or {}
end

-- Get the root node
function RootedTree:getRoot()
    return self.root
end

-- Get the maximum depth of the tree
function RootedTree:getMaxDepth()
    return self.maxDepth
end

-- Get the total number of nodes
function RootedTree:getNodeCount()
    return self.nodeCount
end

-- Depth-first traversal (pre-order)
function RootedTree:traverseDFS(node, callback)
    node = node or self.root
    callback(node)
    for _, child in ipairs(node.children) do
        self:traverseDFS(child, callback)
    end
end

-- Breadth-first traversal (level-order)
function RootedTree:traverseBFS(callback)
    local queue = {self.root}
    while #queue > 0 do
        local node = table.remove(queue, 1)
        callback(node)
        for _, child in ipairs(node.children) do
            table.insert(queue, child)
        end
    end
end

-- Get all leaf nodes
function RootedTree:getLeaves()
    local leaves = {}
    self:traverseDFS(nil, function(node)
        if #node.children == 0 then
            table.insert(leaves, node)
        end
    end)
    return leaves
end

-- Get the path from root to a node
function RootedTree:getPathToRoot(node)
    local path = {}
    local current = node
    while current do
        table.insert(path, 1, current)
        current = current.parent
    end
    return path
end

-- Calculate A000081 sequence value (rooted trees with n nodes)
-- This is a recursive formula based on the OEIS definition
-- Uses memoization via cache for efficiency
local a000081_cache = {[0] = 0, [1] = 1}

-- Tolerance for validating that result is close to an integer
local ROUNDING_TOLERANCE = 0.01

local function calculateA000081(n)
    if a000081_cache[n] then
        return a000081_cache[n]
    end
    
    -- Pre-calculate all values up to n using dynamic programming (bottom-up)
    for i = 2, n do
        if not a000081_cache[i] then
            -- Recurrence: a(i) = (1/(i-1)) * Sum_{k=1..i-1} (Sum_{d|k} d*a(d)) * a(i-k)
            local result = 0
            for k = 1, i - 1 do
                -- Calculate divisor sum for k
                local divisorSum = 0
                for d = 1, k do
                    if k % d == 0 then
                        divisorSum = divisorSum + d * calculateA000081(d)
                    end
                end
                result = result + divisorSum * calculateA000081(i - k)
            end
            result = result / (i - 1)
            
            -- Validate that result is close to an integer
            local rounded = math.floor(result + 0.5)
            if math.abs(result - rounded) > ROUNDING_TOLERANCE then
                error(string.format("A000081 calculation error at n=%d: result %f is not close to integer", i, result))
            end
            
            a000081_cache[i] = rounded
        end
    end
    
    return a000081_cache[n]
end

-- Get the A000081 sequence value for n nodes
function RootedTree.countRootedTrees(n)
    return calculateA000081(n)
end

-- Get the A000081 sequence up to n
function RootedTree.getA000081Sequence(n)
    local sequence = {}
    for i = 0, n do
        table.insert(sequence, calculateA000081(i))
    end
    return sequence
end

-- Find a node by value using DFS
function RootedTree:findNode(value)
    local found = nil
    self:traverseDFS(nil, function(node)
        if not found and node.value == value then
            found = node
        end
    end)
    return found
end

-- Get the subtree size (number of nodes in subtree rooted at node)
function RootedTree:getSubtreeSize(node)
    local size = 1
    for _, child in ipairs(node.children) do
        size = size + self:getSubtreeSize(child)
    end
    return size
end

-- Get the height of a subtree (longest path to leaf)
function RootedTree:getSubtreeHeight(node)
    if #node.children == 0 then
        return 0
    end
    local maxHeight = 0
    for _, child in ipairs(node.children) do
        local childHeight = self:getSubtreeHeight(child)
        if childHeight > maxHeight then
            maxHeight = childHeight
        end
    end
    return maxHeight + 1
end

-- Convert tree to string representation (parenthesis notation)
function RootedTree:toString(node)
    node = node or self.root
    if #node.children == 0 then
        return "(" .. tostring(node.value) .. ")"
    end
    
    local result = "(" .. tostring(node.value)
    for _, child in ipairs(node.children) do
        result = result .. self:toString(child)
    end
    result = result .. ")"
    return result
end

-- Create tree from parenthesis notation
function RootedTree.fromString(str)
    -- Simple parser for parenthesis notation
    -- Format: (value(child1)(child2)...)
    local function parseTree(s, pos)
        if s:sub(pos, pos) ~= '(' then
            return nil, pos
        end
        pos = pos + 1
        
        -- Extract value
        local valueEnd = pos
        while valueEnd <= #s and s:sub(valueEnd, valueEnd) ~= '(' and s:sub(valueEnd, valueEnd) ~= ')' do
            valueEnd = valueEnd + 1
        end
        local value = s:sub(pos, valueEnd - 1)
        pos = valueEnd
        
        local tree = RootedTree.new(value)
        
        -- Parse children
        while pos <= #s and s:sub(pos, pos) == '(' do
            local childTree, newPos = parseTree(s, pos)
            if childTree then
                -- Add child tree as a child of root
                local childNode = tree:addChild(tree.root, childTree.root.value)
                -- Copy children from parsed subtree
                for _, grandchild in ipairs(childTree.root.children) do
                    tree:addChild(childNode, grandchild.value)
                end
                pos = newPos
            else
                break
            end
        end
        
        if s:sub(pos, pos) == ')' then
            pos = pos + 1
        end
        
        return tree, pos
    end
    
    local tree, _ = parseTree(str, 1)
    return tree
end

return RootedTree
