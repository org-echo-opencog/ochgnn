--[[
Test suite for OpenCog AtomSpace integration with hypergraph neural networks

This test suite validates the core functionality of the atomspace implementation,
including atom creation, truth value propagation, attention allocation, and
neural network integration.

Author: Neural Network Graph Package + OpenCog Integration
License: Same as parent project
]]--

require 'nngraph'
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

-- Test AtomSpace basic operations
local function testAtomSpaceBasics()
    print("Testing AtomSpace basic operations...")
    
    local atomspace = nngraph.AtomSpace()
    
    -- Test atom creation
    local conceptHandle = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "cat")
    assertEq(type(conceptHandle), "number", "Atom handle should be number")
    
    local predicateHandle = atomspace:addAtom(AtomTypes.PREDICATE_NODE, "likes")
    assertEq(type(predicateHandle), "number", "Predicate handle should be number")
    
    -- Test atom retrieval
    local conceptAtom = atomspace:getAtom(conceptHandle)
    assertEq(conceptAtom.type, AtomTypes.CONCEPT_NODE, "Atom type should match")
    assertEq(conceptAtom.name, "cat", "Atom name should match")
    
    -- Test atoms by type
    local concepts = atomspace:getAtomsByType(AtomTypes.CONCEPT_NODE)
    assertEq(#concepts, 1, "Should have one concept node")
    
    -- Test atoms by name
    local namedAtoms = atomspace:getAtomsByName("cat")
    assertEq(#namedAtoms, 1, "Should have one atom named 'cat'")
    
    print("✓ AtomSpace basic operations test passed")
end

-- Test truth value operations
local function testTruthValues()
    print("Testing truth value operations...")
    
    local atomspace = nngraph.AtomSpace()
    local handle = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "test")
    
    -- Test setting and getting truth values
    atomspace:setTruthValue(handle, 0.8, 0.9)
    local strength, confidence = atomspace:getTruthValue(handle)
    assertNear(strength, 0.8, 1e-6, "Truth value strength should match")
    assertNear(confidence, 0.9, 1e-6, "Truth value confidence should match")
    
    print("✓ Truth value operations test passed")
end

-- Test attention values
local function testAttentionValues()
    print("Testing attention value operations...")
    
    local atomspace = nngraph.AtomSpace()
    local handle = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "test")
    
    -- Test setting and getting attention values
    atomspace:setAttentionValue(handle, 100, 50, false)
    local sti, lti, vlti = atomspace:getAttentionValue(handle)
    assertEq(sti, 100, "STI should match")
    assertEq(lti, 50, "LTI should match")
    assertEq(vlti, false, "VLTI should match")
    
    print("✓ Attention value operations test passed")
end

-- Test link creation
local function testLinkCreation()
    print("Testing link creation...")
    
    local atomspace = nngraph.AtomSpace()
    
    -- Create some nodes
    local cat = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "cat")
    local animal = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "animal")
    
    -- Create inheritance link
    local inheritance = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {cat, animal})
    
    local inheritanceAtom = atomspace:getAtom(inheritance)
    assertEq(#inheritanceAtom.outgoing, 2, "Inheritance link should have 2 outgoing atoms")
    assertEq(inheritanceAtom.outgoing[1], cat, "First outgoing should be cat")
    assertEq(inheritanceAtom.outgoing[2], animal, "Second outgoing should be animal")
    
    -- Check incoming links
    local catAtom = atomspace:getAtom(cat)
    assertEq(#catAtom.incoming, 1, "Cat should have 1 incoming link")
    assertEq(catAtom.incoming[1], inheritance, "Incoming should be inheritance link")
    
    print("✓ Link creation test passed")
end

-- Test AtomNode functionality
local function testAtomNode()
    print("Testing AtomNode functionality...")
    
    local atomspace = nngraph.AtomSpace()
    local handle = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "test_node")
    atomspace:setTruthValue(handle, 0.7, 0.8)
    
    local atomNode = nngraph.AtomNode(atomspace, handle)
    
    -- Test embedding
    local embedding = atomNode:getEmbedding()
    assertEq(torch.isTensor(embedding), true, "Embedding should be tensor")
    assertEq(embedding:nElement() > 0, true, "Embedding should have elements")
    
    -- Test activation
    local activation = atomNode:getActivation()
    assertEq(type(activation), "number", "Activation should be number")
    
    -- Test forward pass
    local output = atomNode:forward()
    assertEq(type(output), "number", "Forward output should be number")
    
    print("✓ AtomNode functionality test passed")
end

-- Test HypergraphModule
local function testHypergraphModule()
    print("Testing HypergraphModule...")
    
    local atomspace = nngraph.AtomSpace()
    
    -- Create some test atoms
    local cat = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "cat")
    local dog = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "dog")
    local animal = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "animal")
    
    -- Set truth values
    atomspace:setTruthValue(cat, 0.9, 0.8)
    atomspace:setTruthValue(dog, 0.8, 0.9)
    atomspace:setTruthValue(animal, 0.7, 0.7)
    
    -- Create hypergraph module
    local hgModule = nn.HypergraphModule(atomspace)
    
    -- Test single atom forward pass
    local output1 = hgModule:forward(cat)
    assertEq(torch.isTensor(output1), true, "Output should be tensor")
    assertEq(output1:nElement(), 1, "Single atom output should have 1 element")
    
    -- Test multiple atoms forward pass
    local output2 = hgModule:forward({cat, dog, animal})
    assertEq(torch.isTensor(output2), true, "Multi-atom output should be tensor")
    assertEq(output2:nElement() >= 1, true, "Multi-atom output should have elements")
    
    print("✓ HypergraphModule test passed")
end

-- Test pattern matching
local function testPatternMatching()
    print("Testing pattern matching...")
    
    local atomspace = nngraph.AtomSpace()
    local hgModule = nn.HypergraphModule(atomspace)
    
    -- Create test structure
    local cat = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "cat")
    local dog = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "dog")
    
    -- Create pattern
    local pattern = {
        type = AtomTypes.CONCEPT_NODE,
        name = "cat"
    }
    
    -- Test pattern matching
    local matches = hgModule:matchPattern(pattern)
    assertEq(#matches >= 1, true, "Should find at least one match")
    
    print("✓ Pattern matching test passed")
end

-- Test atom type utilities
local function testAtomTypes()
    print("Testing AtomTypes utilities...")
    
    -- Test type checking
    assertEq(AtomTypes.isNodeType(AtomTypes.CONCEPT_NODE), true, "ConceptNode should be node type")
    assertEq(AtomTypes.isLinkType(AtomTypes.INHERITANCE_LINK), true, "InheritanceLink should be link type")
    assertEq(AtomTypes.canEmbed(AtomTypes.CONCEPT_NODE), true, "ConceptNode should be embeddable")
    
    -- Test arity validation
    local valid, msg = AtomTypes.validateArity(AtomTypes.INHERITANCE_LINK, 2)
    assertEq(valid, true, "InheritanceLink with arity 2 should be valid")
    
    local invalid, msg2 = AtomTypes.validateArity(AtomTypes.INHERITANCE_LINK, 1)
    assertEq(invalid, false, "InheritanceLink with arity 1 should be invalid")
    
    print("✓ AtomTypes utilities test passed")
end

-- Test neural network integration
local function testNeuralNetworkIntegration()
    print("Testing neural network integration...")
    
    local atomspace = nngraph.AtomSpace()
    
    -- Create atoms and links
    local cat = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "cat")
    local animal = atomspace:addAtom(AtomTypes.CONCEPT_NODE, "animal")
    local inheritance = atomspace:addAtom(AtomTypes.INHERITANCE_LINK, nil, {cat, animal})
    
    -- Set initial truth values
    atomspace:setTruthValue(cat, 0.9, 0.8)
    atomspace:setTruthValue(animal, 0.8, 0.9)
    atomspace:setTruthValue(inheritance, 0.7, 0.6)
    
    -- Create hypergraph module
    local hgModule = nn.HypergraphModule(atomspace, {
        embeddingDim = 32,
        hiddenDim = 64,
        outputDim = 16
    })
    
    -- Forward pass
    local input = {cat, animal, inheritance}
    local output = hgModule:forward(input)
    
    assertEq(torch.isTensor(output), true, "Output should be tensor")
    
    -- Backward pass
    local gradOutput = torch.ones(output:size())
    local gradInput = hgModule:backward(input, gradOutput)
    
    assertEq(torch.isTensor(gradInput), true, "Gradient input should be tensor")
    
    print("✓ Neural network integration test passed")
end

-- Run all tests
local function runAllTests()
    print("=" .. string.rep("=", 50))
    print("Running OpenCog AtomSpace Integration Tests")
    print("=" .. string.rep("=", 50))
    
    local tests = {
        testAtomSpaceBasics,
        testTruthValues,
        testAttentionValues,
        testLinkCreation,
        testAtomNode,
        testAtomTypes,
        testHypergraphModule,
        testPatternMatching,
        testNeuralNetworkIntegration
    }
    
    local passed = 0
    local failed = 0
    
    for i, test in ipairs(tests) do
        local success, error_msg = pcall(test)
        if success then
            passed = passed + 1
        else
            failed = failed + 1
            print("✗ Test failed: " .. tostring(error_msg))
        end
    end
    
    print("")
    print("=" .. string.rep("=", 50))
    print(string.format("Test Results: %d passed, %d failed", passed, failed))
    print("=" .. string.rep("=", 50))
    
    if failed == 0 then
        print("🎉 All tests passed!")
        return true
    else
        print("❌ Some tests failed!")
        return false
    end
end

-- Export test functions
return {
    runAllTests = runAllTests,
    testAtomSpaceBasics = testAtomSpaceBasics,
    testTruthValues = testTruthValues,
    testAttentionValues = testAttentionValues,
    testLinkCreation = testLinkCreation,
    testAtomNode = testAtomNode,
    testAtomTypes = testAtomTypes,
    testHypergraphModule = testHypergraphModule,
    testPatternMatching = testPatternMatching,
    testNeuralNetworkIntegration = testNeuralNetworkIntegration
}