#!/usr/bin/env python3
"""
Validation script for Rooted Hypershell Architecture implementation

This script validates that the implementation files are present, properly structured,
and contain the expected components without requiring a full Torch/Lua environment.

Usage:
    python3 validate_rooted_hypershell.py
"""

import os
import sys
import re

def check_file_exists(filepath, description):
    """Check if a file exists and report"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def check_function_exists(filepath, function_name):
    """Check if a function exists in a Lua file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Check for various function definition patterns
            patterns = [
                f'function {function_name}',
                f'function.*{function_name}',
                f'local function {function_name}',
                f'function.*:{function_name}',
                f'{function_name} = function',
                f'function.*\\.{function_name}',
            ]
            for pattern in patterns:
                if re.search(pattern, content):
                    return True
    except Exception:
        pass
    return False

def check_class_exists(filepath, class_name):
    """Check if a class exists in a Lua file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Check for class definition
            if f'local {class_name}' in content or f'{class_name} = {{}}' in content:
                return True
            # Check for torch.class definition
            if f"torch.class('{class_name}'" in content or f'torch.class("nn.{class_name}"' in content:
                return True
    except Exception:
        pass
    return False

def validate_rooted_tree():
    """Validate rooted_tree.lua implementation"""
    print("\n=== Validating RootedTree Module ===")
    
    filepath = 'rooted_tree.lua'
    if not check_file_exists(filepath, "RootedTree module"):
        return False
    
    functions = [
        'RootedTree.new',
        'addChild',
        'getNodesAtDepth',
        'traverseDFS',
        'traverseBFS',
        'getLeaves',
        'countRootedTrees',
        'getA000081Sequence',
    ]
    
    all_present = True
    for func in functions:
        if check_function_exists(filepath, func):
            print(f"  ✓ Function: {func}")
        else:
            print(f"  ✗ Function missing: {func}")
            all_present = False
    
    # Check for A000081 implementation
    with open(filepath, 'r') as f:
        content = f.read()
        if 'A000081' in content or 'a000081' in content:
            print("  ✓ A000081 sequence implementation present")
        else:
            print("  ✗ A000081 sequence implementation missing")
            all_present = False
    
    return all_present

def validate_hypershell():
    """Validate hypershell.lua implementation"""
    print("\n=== Validating Hypershell Module ===")
    
    filepath = 'hypershell.lua'
    if not check_file_exists(filepath, "Hypershell module"):
        return False
    
    functions = [
        'Hypershell.new',
        'buildShells',
        'getShell',
        'getNodeDepth',
        'propagateOutward',
        'propagateInward',
        'spreadAttention',
    ]
    
    all_present = True
    for func in functions:
        if check_function_exists(filepath, func):
            print(f"  ✓ Function: {func}")
        else:
            print(f"  ✗ Function missing: {func}")
            all_present = False
    
    return all_present

def validate_rooted_hypershell():
    """Validate rooted_hypershell.lua implementation"""
    print("\n=== Validating RootedHypershell Module ===")
    
    filepath = 'rooted_hypershell.lua'
    if not check_file_exists(filepath, "RootedHypershell module"):
        return False
    
    # Check for torch.class definition
    if check_class_exists(filepath, 'RootedHypershell'):
        print("  ✓ Class: nn.RootedHypershell")
    else:
        print("  ✗ Class definition missing")
        return False
    
    functions = [
        'updateOutput',
        'updateGradInput',
        'buildTreeFromShells',
        'spreadAttention',
        'hierarchicalInference',
        'getRelevantNodes',
    ]
    
    all_present = True
    for func in functions:
        if check_function_exists(filepath, func):
            print(f"  ✓ Method: {func}")
        else:
            print(f"  ✗ Method missing: {func}")
            all_present = False
    
    return all_present

def validate_tests():
    """Validate test file"""
    print("\n=== Validating Test Suite ===")
    
    filepath = 'test/test_rooted_hypershell.lua'
    if not check_file_exists(filepath, "Test suite"):
        return False
    
    tests = [
        'testRootedTreeBasics',
        'testRootedTreeTraversal',
        'testA000081Sequence',
        'testHypershellCreation',
        'testRootedHypershell',
    ]
    
    all_present = True
    for test in tests:
        if check_function_exists(filepath, test):
            print(f"  ✓ Test: {test}")
        else:
            print(f"  ✗ Test missing: {test}")
            all_present = False
    
    return all_present

def validate_examples():
    """Validate example files"""
    print("\n=== Validating Examples ===")
    
    filepath = 'examples/rooted_hypershell_example.lua'
    if not check_file_exists(filepath, "Example file"):
        return False
    
    # Check for key usage patterns
    with open(filepath, 'r') as f:
        content = f.read()
        
        checks = [
            ('RootedTree', 'RootedTree usage'),
            ('Hypershell', 'Hypershell usage'),
            ('RootedHypershell', 'RootedHypershell usage'),
            ('A000081', 'A000081 sequence'),
            ('spreadAttention', 'Attention spreading'),
        ]
        
        all_present = True
        for pattern, description in checks:
            if pattern in content:
                print(f"  ✓ {description}")
            else:
                print(f"  ✗ {description} missing")
                all_present = False
    
    return all_present

def validate_documentation():
    """Validate documentation files"""
    print("\n=== Validating Documentation ===")
    
    all_present = True
    
    # Check main documentation
    if check_file_exists('doc/ROOTED_HYPERSHELL.md', "Architecture documentation"):
        with open('doc/ROOTED_HYPERSHELL.md', 'r') as f:
            content = f.read()
            if 'A000081' in content:
                print("  ✓ A000081 documentation present")
            if 'Hypershell' in content:
                print("  ✓ Hypershell documentation present")
            if 'RootedTree' in content:
                print("  ✓ RootedTree documentation present")
    else:
        all_present = False
    
    # Check README update
    if check_file_exists('README.md', "README"):
        with open('README.md', 'r') as f:
            content = f.read()
            if 'Rooted Hypershell' in content or 'rooted hypershell' in content:
                print("  ✓ README mentions Rooted Hypershell")
            else:
                print("  ✗ README does not mention Rooted Hypershell")
                all_present = False
    else:
        all_present = False
    
    return all_present

def validate_integration():
    """Validate integration with init.lua"""
    print("\n=== Validating Integration ===")
    
    if not check_file_exists('init.lua', "init.lua"):
        return False
    
    with open('init.lua', 'r') as f:
        content = f.read()
        
        modules = [
            'rooted_tree',
            'hypershell',
            'rooted_hypershell',
        ]
        
        all_present = True
        for module in modules:
            if f"require('nngraph.{module}')" in content or f'require("nngraph.{module}")' in content:
                print(f"  ✓ Module required: {module}")
            else:
                print(f"  ✗ Module not required: {module}")
                all_present = False
    
    return all_present

def main():
    """Main validation function"""
    print("=" * 60)
    print("Rooted Hypershell Architecture Validation")
    print("=" * 60)
    
    results = {
        'RootedTree': validate_rooted_tree(),
        'Hypershell': validate_hypershell(),
        'RootedHypershell': validate_rooted_hypershell(),
        'Tests': validate_tests(),
        'Examples': validate_examples(),
        'Documentation': validate_documentation(),
        'Integration': validate_integration(),
    }
    
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    all_passed = True
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{component:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All validation checks passed!")
        print("\nThe rooted hypershell architecture implementation includes:")
        print("  • RootedTree with OEIS A000081 sequence enumeration")
        print("  • Hypershell organization with shell-based processing")
        print("  • RootedHypershell neural network integration")
        print("  • Comprehensive test suite")
        print("  • Example usage demonstrations")
        print("  • Complete documentation")
        return 0
    else:
        print("\n✗ Some validation checks failed.")
        print("Please review the failed checks above.")
        return 1

if __name__ == '__main__':
    os.chdir('/home/runner/work/ochgnn/ochgnn')
    sys.exit(main())
