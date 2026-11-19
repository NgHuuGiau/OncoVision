import sys
import os
from pathlib import Path

def main():
    """Launcher that runs the actual application in-process"""
    # Get the directory of this executable
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_dir = os.path.dirname(sys.executable)
        # Add the _internal directory to sys.path for bundled modules
        internal_dir = os.path.join(base_dir, "_internal")
        if internal_dir not in sys.path:
            sys.path.insert(0, internal_dir)
    else:
        # Running as script — project root is parent of entrypoints/
        base_dir = str(Path(__file__).resolve().parent.parent)
    
    # Add project root to sys.path
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    
    # Change to the base directory
    os.chdir(base_dir)
    
    # Import and run the main module
    try:
        from entrypoints import run_chat
        return run_chat.main()
    except ImportError as e:
        print(f"Error importing entrypoints.run_chat: {e}")
        print(f"sys.path: {sys.path}")
        print(f"base_dir: {base_dir}")
        return 1
    except Exception as e:
        print(f"Error running run_chat: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())