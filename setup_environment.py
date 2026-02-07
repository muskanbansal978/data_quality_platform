"""
Data Quality Platform - Environment Setup and Verification

Automatically installs all required packages and verifies installation.
Handles the complete environment setup for the data quality platform.
"""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install all packages from requirements.txt using pip."""
    requirements_file = Path("requirements.txt")

    if not requirements_file.exists():
        print("Error: requirements.txt not found")
        return False

    print("=" * 60)
    print("Installing packages from requirements.txt...")
    print("This may take a few minutes...")
    print("=" * 60)

    try:
        # Run pip install with requirements.txt
        # Stream output instead of capturing it to show progress
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True
        )

        print("\n✓ All packages installed successfully!")
        return True

    except subprocess.CalledProcessError:
        print(f"\n✗ Error during installation")
        return False


def load_packages():
    """Load package names from requirements.txt"""
    packages = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name without version specifiers
                pkg = line.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
                packages.append(pkg)
    return packages


def verify_installation():
    """Verify that all packages can be imported"""
    print("\n" + "=" * 60)
    print("Verifying package installation...")
    print("=" * 60)

    packages = load_packages()
    installed = []
    missing = []

    for package in packages:
        # Handle special cases where package name != import name
        import_name = package.replace('-', '_')

        # Special mappings for common packages
        name_mappings = {
            'scikit-learn': 'sklearn',
            'python-dotenv': 'dotenv',
            'great-expectations': 'great_expectations',
        }

        if package in name_mappings:
            import_name = name_mappings[package]

        try:
            __import__(import_name)
            installed.append(package)
            print(f" {package}")
        except ImportError as e:
            missing.append(package)
            print(f" {package} - {e}")

    print("\n" + "=" * 60)
    print(f"Installation Summary: {len(installed)}/{len(packages)} packages verified")
    print("=" * 60)

    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("These packages may need manual installation or have import name mismatches.")
        return False
    else:
        print("\n All packages verified and ready to use!")
        return True


def main():
    """Main setup function"""
    print("\n" + "=" * 60)
    print("Data Quality Platform - Environment Setup")
    print("=" * 60 + "\n")

    # Step 1: Install packages
    install_success = install_requirements()

    if not install_success:
        print("\nSetup failed during installation")
        sys.exit(1)

    # Step 2: Verify installation
    verify_success = verify_installation()

    if verify_success:
        print("\n" + "=" * 60)
        print("🎉 Environment setup complete!")
        print("=" * 60)
        print("\nYou can now run:")
        print("  - python -m src.data_profiler")
        print("  - python -m src.anomaly_detector")
        print("  - python -m src.llm_explainer")
        print("  - streamlit run dashboard.py")
    else:
        print("\n  Setup completed with warnings")
        print("Some packages may not have been verified correctly.")
        print("Try running your scripts to see if everything works.")


if __name__ == "__main__":
    main()
