import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Simple test to check if matplotlib is working
try:
    # Create simple data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title('Simple Sine Wave Test')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Save plot
    plt.savefig('simple_test_plot.png')
    plt.close()
    
    print("Simple plot test completed successfully. Plot saved as simple_test_plot.png")
except Exception as e:
    print(f"Error in simple plot test: {e}")
    import traceback
    traceback.print_exc()