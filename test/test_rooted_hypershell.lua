--[[
Test suite for Rooted Hypershell Architecture

This test suite validates the rooted tree, hypershell, and integrated
rooted hypershell architecture components.

Author: Neural Network Graph Package + OpenCog Integration
License: Same as parent project
]]--

require 'nngraph'
local RootedTree = require('nngraph.rooted_tree')
local Hypershell = require('nngraph.hypershell')
local AtomTypes = require('nngraph.atom_types')

-- Test helper functions
local function assertEq(actual, expected, message)
    if actual ~= expected then
        error(string.format("Assertion failed: %s. Expected %s, got %s", 
              message or "values not equal", tostring(expected), tostring(actual)))
    end
end

local function assertNear(actual, expected, tolerance, message)
    tolerance = tolerance or 1e-6
    if math.abs(actual - expected) > tolerance then
        error(string.format("Assertion failed: %s. Expected %s ± %s, got %s", 
              message or "values not near", tostring(expected), tostring(tolerance), tostring(actual)))
    end
end

local function assertTrue(condition, message)
    if not condition then
        error("Assertion failed: " .. (message or "condition is false"))
    end
end

-- Test RootedTree basic operations
local function testRootedTreeBasics()
    print("Testing RootedTree basic operations...")
    
    local tree = RootedTree.new("root")
    assertEq(tree:getNodeCount(), 1, "Tree should start with 1 node")
    assertEq(tree:getMaxDepth(), 0, "Tree should start with depth 0")
    
    -- Add children
    local child1 = tree:addChild(tree:getRoot(), "child1")
    assertEq(tree:getNodeCount(), 2, "Tree should have 2 nodes")
    assertEq(tree:getMaxDepth(), 1, "Tree should have depth 1")
    
    local child2 = tree:addChild(tree:getRoot(), "child2")
    assertEq(tree:getNodeCount(), 3, "Tree should have 3 nodes")
    
    local grandchild = tree:addChild(child1, "grandchild")
    assertEq(tree:getNodeCount(), 4, "Tree should have 4 nodes")
    assertEq(tree:getMaxDepth(), 2, "Tree should have depth 2")
    
    -- Test depth queries
    local nodesAtDepth0 = tree:getNodesAtDepth(0)
    assertEq(#nodesAtDepth0, 1, "Should have 1 node at depth 0")
    
    local nodesAtDepth1 = tree:getNodesAtDepth(1)
    assertEq(#nodesAtDepth1, 2, "Should have 2 nodes at depth 1")
    
    local nodesAtDepth2 = tree:getNodesAtDepth(2)
    assertEq(#nodesAtDepth2, 1, "Should have 1 node at depth 2")
    
    print("✓ RootedTree basic operations test passed")
end

-- Test RootedTree traversal
local function testRootedTreeTraversal()
    print("Testing RootedTree traversal...")
    
    local tree = RootedTree.new("root")
    local child1 = tree:addChild(tree:getRoot(), "child1")
    local child2 = tree:addChild(tree:getRoot(), "child2")
    tree:addChild(child1, "grandchild1")
    tree:addChild(child1, "grandchild2")
    
    -- Test DFS traversal
    local dfsOrder = {}
    tree:traverseDFS(nil, function(node)
        table.insert(dfsOrder, node.value)
    end)
    assertEq(#dfsOrder, 5, "DFS should visit all 5 nodes")
    assertEq(dfsOrder[1], "root", "DFS should start at root")
    
    -- Test BFS traversal
    local bfsOrder = {}
    tree:traverseBFS(function(node)
        table.insert(bfsOrder, node.value)
    end)
    assertEq(#bfsOrder, 5, "BFS should visit all 5 nodes")
    assertEq(bfsOrder[1], "root", "BFS should start at root")
    
    -- Test leaf finding
    local leaves = tree:getLeaves()
    assertEq(#leaves, 3, "Tree should have 3 leaves")
    
    print("✓ RootedTree traversal test passed")
end

-- Test A000081 sequence calculation
local function testA000081Sequence()
    print("Testing A000081 sequence calculation...")
    
    -- Known values from OEIS A000081: https://oeis.org/A000081
    -- Sequence: Number of unlabeled rooted trees with n nodes
    local expected = {0, 1, 1, 2, 4, 9, 20, 48, 115, 286, 719}
    
    for i = 0, #expected - 1 do
        local actual = RootedTree.countRootedTrees(i)
        assertEq(actual, expected[i + 1], 
                 string.format("A000081(%d) should be %d", i, expected[i + 1]))
    end
    
    -- Test sequence generation
    local sequence = RootedTree.getA000081Sequence(10)
    assertEq(#sequence, 11, "Sequence should have 11 elements (0-10)")
    
    for i = 0, 10 do
        assertEq(sequence[i + 1], expected[i + 1],
                 string.format("Sequence[%d] should be %d", i, expected[i + 1]))
    end
    
    print("✓ A000081 sequence test passed")
end

-- Test RootedTree string conversion
local function testRootedTreeString()
    print("Testing RootedTree string conversion...")
    
    local tree = RootedTree.new("A")
    local child1 = tree:addChild(tree:getRoot(), "B")
    local child2 = tree:addChild(tree:getRoot(), "C")
    tree:addChild(child1, "D")
    
    local str = tree:toString()
    assertTrue(str:sub(1, 2) == "(A", "String should start with (A")
    assertTrue(str:sub(-1) == ")", "String should end with )")
    
    print("✓ RootedTree string conversion test passed")
end

-- Test Hypershell creation
local function testHypershellCreation()
    print("Testing Hypershell creation...")
    
    local atomspace = nngraph.AtomSpace()
    
    -- Create a simple knowledge graph
    local cat = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "cat")
    local animal = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "animal")
    local mammal = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "mammal")
    
    local link1 = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {cat, mammal})
    local link2 = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {mammal, animal})
    
    -- Create hypershell with cat as root
    local hypershell = Hypershell.new(atomspace, cat)
    
    assertEq(hypershell:getNodeDepth(cat), 0, "Root should be at depth 0")
    assertTrue(hypershell:getMaxDepth() >= 0, "Should have at least depth 0")
    
    local shell0 = hypershell:getShell(0)
    assertEq(#shell0, 1, "Shell 0 should have 1 node (root)")
    assertEq(shell0[1], cat, "Shell 0 should contain the root")
    
    print("✓ Hypershell creation test passed")
end

-- Test Hypershell propagation
local function testHypershellPropagation()
    print("Testing Hypershell propagation...")
    
    local atomspace = nngraph.AtomSpace()
    
    -- Create a chain: A -> B -> C
    local a = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "A")
    local b = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "B")
    local c = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "C")
    
    local link1 = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {a, b})
    local link2 = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {b, c})
    
    local hypershell = Hypershell.new(atomspace, a)
    
    -- Test outward propagation
    local values = hypershell:propagateOutward(100, function(value, from, to, depth)
        return value * 0.8  -- Attenuate by 0.8
    end)
    
    assertEq(values[a], 100, "Root should have initial value")
    assertTrue(values[b] ~= nil, "Node B should receive propagated value")
    
    print("✓ Hypershell propagation test passed")
end

-- Test RootedHypershell integration
local function testRootedHypershell()
    print("Testing RootedHypershell integration...")
    
    local atomspace = nngraph.AtomSpace()
    
    -- Create knowledge graph
    local cat = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "cat")
    local mammal = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "mammal")
    local animal = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "animal")
    
    atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {cat, mammal})
    atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {mammal, animal})
    
    -- Set truth values
    atomspace:setTruthValue(cat, 0.9, 0.8)
    atomspace:setTruthValue(mammal, 0.8, 0.9)
    atomspace:setTruthValue(animal, 0.7, 0.9)
    
    -- Create rooted hypershell
    local rhs = nn.RootedHypershell(atomspace, cat, {
        embeddingDim = 32,
        hiddenDim = 64,
        numShells = 3
    })
    
    -- Test forward pass
    local output = rhs:forward(nil)
    assertEq(output:size(1), 32, "Output should have embedding dimension")
    
    -- Test statistics
    local stats = rhs:getStats()
    assertTrue(stats.tree ~= nil, "Stats should include tree info")
    assertTrue(stats.shell ~= nil, "Stats should include shell info")
    assertEq(stats.embeddingDim, 32, "Stats should report correct embedding dim")
    
    -- Test tree and shell access
    local tree = rhs:getTree()
    assertTrue(tree ~= nil, "Should be able to get tree")
    assertEq(tree:getNodeCount() > 0, true, "Tree should have nodes")
    
    local hypershell = rhs:getHypershell()
    assertTrue(hypershell ~= nil, "Should be able to get hypershell")
    
    print("✓ RootedHypershell integration test passed")
end

-- Test attention spreading
local function testAttentionSpreading()
    print("Testing attention spreading...")
    
    local atomspace = nngraph.AtomSpace()
    
    local root = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "root")
    local child1 = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "child1")
    local child2 = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "child2")
    
    atomspace:addAtom(AtomTypes.LIST_LINK, nil, {root, child1})
    atomspace:addAtom(AtomTypes.LIST_LINK, nil, {root, child2})
    
    -- Set high attention on root
    atomspace:setAttentionValue(root, 100, 50, false)
    atomspace:setAttentionValue(child1, 0, 0, false)
    atomspace:setAttentionValue(child2, 0, 0, false)
    
    local rhs = nn.RootedHypershell(atomspace, root)
    
    -- Spread attention
    rhs:spreadAttention(2)
    
    -- Check that children received attention
    local sti1, _, _ = atomspace:getAttentionValue(child1)
    local sti2, _, _ = atomspace:getAttentionValue(child2)
    
    -- Children should have some attention now (but less than root)
    assertTrue(sti1 > 0 or sti2 > 0, "Children should receive some attention")
    
    print("✓ Attention spreading test passed")
end

-- Test rooted tree counts
local function testRootedTreeCounts()
    print("Testing rooted tree count calculations...")
    
    local counts = nn.RootedHypershell.getRootedTreeCounts(7)
    
    assertEq(#counts, 8, "Should have counts for 0-7")
    assertEq(counts[1], 0, "A000081(0) should be 0")
    assertEq(counts[2], 1, "A000081(1) should be 1")
    assertEq(counts[3], 1, "A000081(2) should be 1")
    assertEq(counts[4], 2, "A000081(3) should be 2")
    assertEq(counts[5], 4, "A000081(4) should be 4")
    
    print("✓ Rooted tree counts test passed")
end

-- Run all tests
local function runAllTests()
    print("\n========================================")
    print("Running Rooted Hypershell Architecture Tests")
    print("========================================\n")
    
    local tests = {
        testRootedTreeBasics,
        testRootedTreeTraversal,
        testA000081Sequence,
        testRootedTreeString,
        testHypershellCreation,
        testHypershellPropagation,
        testRootedHypershell,
        testAttentionSpreading,
        testRootedTreeCounts
    }
    
    local passed = 0
    local failed = 0
    
    for _, test in ipairs(tests) do
        local success, err = pcall(test)
        if success then
            passed = passed + 1
        else
            failed = failed + 1
            print("✗ Test failed: " .. tostring(err))
        end
    end
    
    print("\n========================================")
    print(string.format("Test Results: %d passed, %d failed", passed, failed))
    print("========================================\n")
    
    return failed == 0
end

-- Export test functions
return {
    runAllTests = runAllTests,
    testRootedTreeBasics = testRootedTreeBasics,
    testRootedTreeTraversal = testRootedTreeTraversal,
    testA000081Sequence = testA000081Sequence,
    testRootedTreeString = testRootedTreeString,
    testHypershellCreation = testHypershellCreation,
    testHypershellPropagation = testHypershellPropagation,
    testRootedHypershell = testRootedHypershell,
    testAttentionSpreading = testAttentionSpreading,
    testRootedTreeCounts = testRootedTreeCounts
}
