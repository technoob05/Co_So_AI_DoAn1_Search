"""
Auto-fix installation script
Automatically installs compatible versions of dependencies
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

print("="*70)
print("AUTO-FIX INSTALLATION SCRIPT")
print("Swarm Intelligence Algorithms Project")
print("="*70)

print("\n📦 Step 1: Uninstalling conflicting packages...")
success, output = run_command(
    f"{sys.executable} -m pip uninstall -y numpy scipy seaborn"
)
if success:
    print("   ✓ Removed conflicting packages")
else:
    print("   ⚠ Warning:", output)

print("\n📦 Step 2: Installing compatible NumPy...")
success, output = run_command(
    f"{sys.executable} -m pip install numpy==1.26.4"
)
if success:
    print("   ✓ NumPy 1.26.4 installed")
else:
    print("   ✗ Error installing NumPy:", output)
    sys.exit(1)

print("\n📦 Step 3: Installing other dependencies...")
packages = [
    "matplotlib>=3.7.0",
    "pandas>=2.0.0",
    "tqdm>=4.65.0",
    "jupyter>=1.0.0",
    "notebook>=6.5.0"
]

for package in packages:
    print(f"   Installing {package}...")
    success, output = run_command(
        f"{sys.executable} -m pip install {package}"
    )
    if not success:
        print(f"   ⚠ Warning: Could not install {package}")

print("\n✅ Step 4: Verifying installation...")

# Test imports
try:
    import numpy
    print(f"   ✓ NumPy {numpy.__version__}")
except ImportError as e:
    print(f"   ✗ NumPy import failed: {e}")
    sys.exit(1)

try:
    import matplotlib
    print(f"   ✓ Matplotlib {matplotlib.__version__}")
except ImportError:
    print("   ⚠ Matplotlib not available")

try:
    import pandas
    print(f"   ✓ Pandas {pandas.__version__}")
except ImportError:
    print("   ⚠ Pandas not available")

# Test project imports
print("\n🧪 Step 5: Testing project modules...")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.test_functions import get_test_function
    print("   ✓ test_functions module works")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    from src.swarm_intelligence.pso import PSO
    print("   ✓ swarm_intelligence module works")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n🎉 Installation successful!")
print("="*70)
print("\n📚 Next steps:")
print("   1. Run simple test:  python run_simple_test.py")
print("   2. Run full demo:    python demo.py")
print("   3. Read guide:       QUICKSTART.md")
print("\n" + "="*70)

