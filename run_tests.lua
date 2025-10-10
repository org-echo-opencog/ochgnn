#!/usr/bin/env lua

--[[
Test runner for OpenCog AtomSpace integration

This script runs the test suite for the OpenCog atomspace integration
to verify that all functionality works correctly.

Usage:
  lua run_tests.lua [test_name]

If test_name is provided, runs only that specific test.
Otherwise, runs the full test suite.
]]--

-- Add current directory to package path
package.path = package.path .. ";./?/init.lua;./?.lua"

-- Check if we can load the required modules
local success, nngraph = pcall(require, 'nngraph')
if not success then
    print("Error: Could not load nngraph module")
    print("Make sure you are in the correct directory and all files are present")
    os.exit(1)
end

-- Try to load the test module
local success, tests = pcall(require, 'test.test_atomspace')
if not success then
    print("Error: Could not load test module")
    print("Error details: " .. tostring(tests))
    os.exit(1)
end

-- Parse command line arguments
local test_name = arg and arg[1]

if test_name then
    -- Run specific test
    print("Running specific test: " .. test_name)
    local test_function = tests[test_name]
    if test_function then
        local success, error_msg = pcall(test_function)
        if success then
            print("✓ Test passed: " .. test_name)
        else
            print("✗ Test failed: " .. test_name)
            print("Error: " .. tostring(error_msg))
            os.exit(1)
        end
    else
        print("Error: Test not found: " .. test_name)
        print("Available tests:")
        for name, func in pairs(tests) do
            if type(func) == 'function' and name ~= 'runAllTests' then
                print("  - " .. name)
            end
        end
        os.exit(1)
    end
else
    -- Run all tests
    local success = tests.runAllTests()
    if not success then
        os.exit(1)
    end
end

print("All tests completed successfully!")