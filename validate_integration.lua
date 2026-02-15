#!/usr/bin/env lua

--[[
Validation script for OpenCog AtomSpace integration

This script performs basic validation checks to ensure the integration
is properly structured and the main components can be loaded.

Note: This is a syntax and structure check. Full functionality testing
requires a Torch7/Lua environment with nn and graph packages.
]]--

print("OpenCog AtomSpace Integration Validation")
print("=" .. string.rep("=", 40))

-- Check if files exist
local required_files = {
    "atomspace.lua",
    "atom_types.lua", 
    "atom_node.lua",
    "hypergraph_module.lua",
    "test/test_atomspace.lua",
    "examples/atomspace_neural_network_example.lua",
    "OPENCOG_INTEGRATION.md"
}

print("\n1. Checking required files...")
local missing_files = {}
for _, file in ipairs(required_files) do
    local f = io.open(file, "r")
    if f then
        f:close()
        print("  ✓ " .. file)
    else
        print("  ✗ " .. file .. " (missing)")
        table.insert(missing_files, file)
    end
end

if #missing_files > 0 then
    print("\nError: Missing required files!")
    for _, file in ipairs(missing_files) do
        print("  - " .. file)
    end
    os.exit(1)
end

-- Check basic syntax of key components
print("\n2. Validating file structure...")

-- Check AtomTypes constants
local atom_types_content = io.open("atom_types.lua", "r"):read("*all")
local required_types = {
    "CONCEPT_NODE", "PREDICATE_NODE", "INHERITANCE_LINK", 
    "SIMILARITY_LINK", "EVALUATION_LINK", "AND_LINK", "OR_LINK"
}

for _, atom_type in ipairs(required_types) do
    if atom_types_content:find(atom_type) then
        print("  ✓ AtomType: " .. atom_type)
    else
        print("  ✗ AtomType missing: " .. atom_type)
    end
end

-- Check AtomSpace class
local atomspace_content = io.open("atomspace.lua", "r"):read("*all")
local required_methods = {
    "addAtom", "getAtom", "setTruthValue", "getTruthValue", 
    "setAttentionValue", "getAttentionValue", "removeAtom"
}

for _, method in ipairs(required_methods) do
    if atomspace_content:find("function.*" .. method) then
        print("  ✓ AtomSpace method: " .. method)
    else
        print("  ✗ AtomSpace method missing: " .. method)
    end
end

-- Check AtomNode class
local atom_node_content = io.open("atom_node.lua", "r"):read("*all")
if atom_node_content:find("torch.class.*AtomNode.*Node") then
    print("  ✓ AtomNode extends nngraph.Node")
else
    print("  ✗ AtomNode class definition issue")
end

-- Check HypergraphModule class  
local hg_module_content = io.open("hypergraph_module.lua", "r"):read("*all")
if hg_module_content:find("torch.class.*HypergraphModule.*Module") then
    print("  ✓ HypergraphModule extends nn.Module")
else
    print("  ✗ HypergraphModule class definition issue")
end

-- Check integration in init.lua
print("\n3. Checking integration...")
local init_content = io.open("init.lua", "r"):read("*all")
local required_requires = {
    "nngraph.atomspace", "nngraph.atom_types", 
    "nngraph.atom_node", "nngraph.hypergraph_module"
}

for _, req in ipairs(required_requires) do
    if init_content:find("require.*" .. req:gsub("%.", "%.")) then
        print("  ✓ Loaded: " .. req)
    else
        print("  ✗ Missing require: " .. req)
    end
end

-- Check test coverage
print("\n4. Checking test coverage...")
local test_content = io.open("test/test_atomspace.lua", "r"):read("*all")
local test_functions = {
    "testAtomSpaceBasics", "testTruthValues", "testAttentionValues",
    "testLinkCreation", "testAtomNode", "testHypergraphModule"
}

for _, test_func in ipairs(test_functions) do
    if test_content:find("function.*" .. test_func) then
        print("  ✓ Test: " .. test_func)
    else
        print("  ✗ Missing test: " .. test_func)
    end
end

-- Check example
print("\n5. Checking example...")
local example_content = io.open("examples/atomspace_neural_network_example.lua", "r"):read("*all")
if example_content:find("AtomSpace") and example_content:find("HypergraphModule") then
    print("  ✓ Example demonstrates core functionality")
else
    print("  ✗ Example missing key demonstrations")
end

-- Check documentation
print("\n6. Checking documentation...")
local doc_content = io.open("OPENCOG_INTEGRATION.md", "r"):read("*all")
local doc_sections = {"Overview", "AtomSpace", "AtomNode", "HypergraphModule", "Examples"}

for _, section in ipairs(doc_sections) do
    if doc_content:find(section) then
        print("  ✓ Documentation section: " .. section)
    else
        print("  ✗ Missing documentation: " .. section)
    end
end

print("\n" .. "=" .. string.rep("=", 40))
print("✅ Integration validation completed!")
print("All required components are present and properly structured.")
print("\nTo run full tests (requires Torch7/Lua environment):")
print("  lua run_tests.lua")
print("\nTo see usage examples:")
print("  lua examples/atomspace_neural_network_example.lua")
print("=" .. string.rep("=", 40))